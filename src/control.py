# Copyright 2023 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import time, sys, signal
from multiprocessing import Process, Lock, Value, Event
from multiprocessing.shared_memory import SharedMemory 
from multiprocessing.managers import BaseManager
from typing import Tuple
from dataclasses import dataclass
from collections import deque

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

import numpy as np 
from pyqtgraph.Qt import QtCore, QtWidgets 
from pyqtgraph.Qt.QtWidgets import QLabel
import pyqtgraph as pg 

from spectrometer import OceanSpectrometer

#https://stackoverflow.com/questions/66423081/is-it-possible-to-share-a-numpy-array-thats-not-empty-between-processes
class SHMArray(np.ndarray):
    def __new__(cls, input_array, shm=None):
        obj = np.asarray(input_array).view(cls)
        obj.shm = shm 
        return obj 

    def __array_finalize__(self, obj):
        if obj is None:
            return 
        self.shm = getattr(obj, 'shm', None)

@dataclass 
class SharedArrrayDescriptor:
    name:  str
    shape: Tuple[int,...]
    dtype: np.dtype
    lock:  Lock

class SpecManager(BaseManager): pass 

SpecManager.register('OceanSpectrometer', OceanSpectrometer,
                exposed=None)

#https://stackoverflow.com/questions/4938723/what-is-the-correct-way-to-make-my-pyqt-application-quit-when-killed-from-the-co
def sigint_handler(*args, **kwargs):
    kwargs['event'].set()
    sys.stderr.write('\r')
    QtWidgets.QApplication.quit()

class SpecApp(object):

    def __init__(self, spec, slock, desc, update_rate, derivative):
        self.spec  = spec
        self.slock = slock
        self.desc  = desc 
        self.update_rate = update_rate

        self.shm  = SharedMemory(name = desc.name)
        self.data = SHMArray(np.ndarray(desc.shape, buffer=self.shm.buf, dtype=desc.dtype), self.shm) 

        self.bg_data        = np.zeros(desc.shape[0], dtype=desc.dtype) 
        self.moving_average  = deque([])

        self.color_data = deque([], maxlen=1000)
        self.pca_data   = deque([], maxlen=1000)
        self.time   = deque([], maxlen=1000)

        self.color_max   = 0.0
        self.color_min   = 0.0
        self.threshold   = 2000
        self.threshold_up = False

        self.update = True

        self.lambdas = [None, None]

        self.bg_save_file = None
        self.smooth_level = 0.1

        self.save_interval_us = 500 * 1000
        self.saving = False
        self.save_elapsed = 0
        self.saving_average = False

        self.save_data = []

        self.slock.acquire()
        self.serial = self.spec.get_serial()
        self.slock.release()

        self.pca_components = None 
        self.pca_mean       = 0.0 
        self.pca_loaded     = False 
        self.display_pca    = False 

    def __del__(self):
        self.shm.close()

    def make_app(self):
        self.app = pg.mkQApp("Spectrometer Control")

        self.startstop_btn = pg.QtWidgets.QPushButton(text="Stop")

        self.int_time_txt  = pg.QtWidgets.QLineEdit("30")
        self.start_txt     = pg.QtWidgets.QLineEdit("200.0")
        self.stop_txt      = pg.QtWidgets.QLineEdit("1100.0")

        self.ymin_txt      = pg.QtWidgets.QLineEdit("0")
        self.ymax_txt      = pg.QtWidgets.QLineEdit("5000")
        self.spec_yscale   = pg.QtWidgets.QCheckBox(text="Autoscale Y")
        self.spec_yscale.setChecked(True)
        self.strip_yscale  = pg.QtWidgets.QPushButton(text="Autoscale Y")
        self.strip_avg     = pg.QtWidgets.QLineEdit(str(self.smooth_level))

        self.avg_txt       = pg.QtWidgets.QLineEdit("10")
        self.avg_enable    = pg.QtWidgets.QCheckBox(text="Average Enable")
        self.avg_restart   = pg.QtWidgets.QPushButton(text="Average Restart")
        self.avg_count_label = pg.QtWidgets.QLabel("Average Count: 0/0")

        self.background    = pg.QtWidgets.QPushButton(text="Record Background")
        self.sub_background= pg.QtWidgets.QCheckBox(text="Background Subtract")

        self.bg_set_file   = pg.QtWidgets.QPushButton(text="Background Save File")
        self.bg_save_file  = pg.QtWidgets.QPushButton(text="Save Background")
        self.bg_load_file  = pg.QtWidgets.QPushButton(text="Load Background")

        self.save_interval = pg.QtWidgets.QLineEdit(str(self.save_interval_us//1000))
        self.stopstart_save = pg.QtWidgets.QPushButton(text="Start Recording")

        self.bg_save_file.setEnabled(False)

        self.serial_label = QLabel(f"{self.serial}")
        if self.serial == 'FAKE':
            self.serial_label.setStyleSheet('font-weight: bold; color: red')
        else:
            self.serial_label.setStyleSheet('font-weight: bold; color: green')


        self.spectrum = pg.PlotWidget()
        self.color    = pg.PlotWidget()

        self.wl_text      = pg.QtWidgets.QLineEdit("703.9")
        self.clear_check = pg.QtWidgets.QCheckBox("Clear")

        self.use_pca        = pg.QtWidgets.QCheckBox("Use PCA")
        self.get_pca_coeffs = pg.QtWidgets.QPushButton("Load PCA Coeffs") 

        self.max_label   = pg.QtWidgets.QLabel("Max Value:")
        self.min_label   = pg.QtWidgets.QLabel("Min Value:")
        self.threshold_label = pg.QtWidgets.QLabel("Threshold:")
        self.threshold_txt = pg.QtWidgets.QLineEdit("2000")
        self.threshold_led = pg.QtWidgets.QLabel("Threshold Crossed")
        self.threshold_dir = pg.QtWidgets.QCheckBox("Greater Than")

        self.layout = pg.LayoutWidget()
        self.layout.addWidget(self.startstop_btn,           row=0, col=7)
        self.layout.addWidget(QLabel("Integration time (ms)"),
                                                            row=0, col=1)
        self.layout.addWidget(self.int_time_txt,            row=0, col=2)
        self.layout.addWidget(self.avg_count_label,
                                                            row=0, col=3)
        self.layout.addWidget(self.avg_txt,                 row=0, col=4)
        self.layout.addWidget(self.avg_enable,              row=0, col=5)
        self.layout.addWidget(self.avg_restart,             row=0, col=6)
        self.layout.addWidget(self.serial_label,            row=0, col=0)

        self.layout.addWidget(QLabel("X Limits"),           row=1, col=0)
        self.layout.addWidget(self.start_txt,               row=1, col=1)
        self.layout.addWidget(self.stop_txt,                row=1, col=2)
        self.layout.addWidget(QLabel("Y Limits"),           row=1, col=3)
        self.layout.addWidget(self.ymin_txt,                row=1, col=4)
        self.layout.addWidget(self.ymax_txt,                row=1, col=5)
        self.layout.addWidget(self.spec_yscale,             row=1, col=6)


        self.layout.addWidget(self.background,              row=2, col=0)
        self.layout.addWidget(self.sub_background,          row=2, col=1)
        self.layout.addWidget(self.bg_save_file,            row=2, col=2)
        self.layout.addWidget(self.bg_load_file,            row=2, col=3)
        self.layout.addWidget(self.bg_set_file,             row=2, col=4)

        self.layout.addWidget(QLabel("Save Interval (ms)"), row=2, col=5)
        self.layout.addWidget(self.save_interval,           row=2, col=6)
        self.layout.addWidget(self.stopstart_save,          row=2, col=7)

        self.layout.addWidget(self.spectrum,                row=3, col=0, colspan=8)

        self.layout.addWidget(QLabel("Wavelength (nm)"), 
                                                            row=4, col=0)
        self.layout.addWidget(self.wl_text,                 row=4, col=1)
        self.layout.addWidget(self.clear_check,             row=4, col=2)
        self.layout.addWidget(self.strip_yscale,            row=4, col=3)
        self.layout.addWidget(QLabel("Averaging:"),         row=4, col=4)
        self.layout.addWidget(self.strip_avg,               row=4, col=5)
        self.layout.addWidget(self.use_pca,                 row=4, col=6)
        self.layout.addWidget(self.get_pca_coeffs,          row=4, col=7)


        self.layout.addWidget(self.min_label,               row=5, col=0)
        self.layout.addWidget(self.max_label,               row=5, col=1)
        self.layout.addWidget(self.threshold_label,         row=5, col=2)
        self.layout.addWidget(self.threshold_txt,           row=5, col=3)
        self.layout.addWidget(self.threshold_led,           row=5, col=5)
        self.layout.addWidget(self.threshold_dir,           row=5, col=6)

        self.layout.addWidget(self.color,                  row=6, col=0, colspan=8)

        self.layout.resize(1800, 1200)
        self.layout.show()

    def _setup_plots(self):

        self.spec_curve = self.spectrum.plot(pen=pg.mkPen('r', width=1))
        self.avg_curve  = self.spectrum.plot(pen=pg.mkPen('w', width=1))
        self.spectrum.setLabel(axis='left', text='Counts')
        self.spectrum.setLabel(axis='bottom', text='Wavelength (nm)')
        self.spectrum.setXRange(200,1100,padding=0)

        self.color_curve = self.color.plot(pen=pg.mkPen('#fca503', width=1, 
                                                style=QtCore.Qt.PenStyle.DashLine))

        self.smoothed_curve = self.color.plot(pen=pg.mkPen('#fca503', width=2, 
                                                style=QtCore.Qt.PenStyle.SolidLine))


        self.color.setXRange(-30, 10, padding=0)

        self.color.setLabel(axis='left', text='Counts')
        self.color.setLabel(axis='bottom', text='Time (s)')

        self.color_vline = pg.InfiniteLine(pos=400, angle=90, movable=True, 
                                   pen=pg.mkPen('#fca503', width=2),
                                   hoverPen=pg.mkPen('w', width=2))

        self.min_hline   = pg.InfiniteLine(pos=0, angle=0, movable=False,
                                            pen=pg.mkPen("w", width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.max_hline   = pg.InfiniteLine(pos=5000, angle=0, movable=False, 
                                            pen=pg.mkPen("w", width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.t_hline     = pg.InfiniteLine(pos=self.threshold, angle=0, movable=True,
                                            pen=pg.mkPen("w", width=2))

        self.spectrum.addItem(self.color_vline, ignoreBounds=True)

        self.color.addItem(self.min_hline, ignoreBounds=True)
        self.color.addItem(self.max_hline, ignoreBounds=True)
        self.color.addItem(self.t_hline,   ignoreBounds=True)

    def _startstop(self):
        if self.update:
            self.startstop_btn.setText("Start")
            self.update = False 
        else:
            self.startstop_btn.setText("Stop")
            self.update = True

    def _load_background(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Background File", "", 
                                                   "NPZ Files (*.npz);;All Files (*)")

        if file_name:
            self.bg_data = np.load(file_name)['background']

    def _set_bg_savefile(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Background File", "", 
                                                   "NPZ Files (*.npz);;All Files (*)")

        if file_name:
            self.bg_save_file.setEnabled(True)
            self.bg_save_file = file_name

    def _load_pca_components(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, "PCA Data File", "",
                                                "NPZ Files (*.npz);;All Files (*)")
        if file_name:
            fd = np.load(file_name)
            self.pca_components = fd['pca_components']
            self.pca_mean       = fd['pca_mean']
            self.pca_loaded     = True

    def _use_pca(self):
        if self.use_pca.isChecked() and self.pca_loaded:
            self.display_pca = True 
        else:
            self.display_pca = False 
        self._clear_strips()

    def _save_background(self):
        if self.bg_save_file:

            self.desc.lock.acquire()
            wlens = self.data[:,0].copy()
            self.desc.lock.release()

            np.savez(self.bg_save_file, background=self.bg_data, wavelengths=wlens)

    def _avg_enable_curve(self):
        if self.avg_enable.isChecked():
            self.avg_curve.show()
        else:
            self.avg_curve.hide()
            self._avg_do_restart()
            self.avg_count_label.setText(f"Average Count: 0/{self.moving_average.maxlen}")

    def _avg_change_count(self):
        self.moving_average = deque(self.moving_average, maxlen=int(self.avg_txt.text()))

    def _avg_do_restart(self):
        self.moving_average.clear()

    def _set_integration_time(self):
        self.slock.acquire()
        self.spec.set_integration_time(1000*int(self.int_time_txt.text()))
        self.slock.release()

    def _get_wavelength(self, wlen):
        self.slock.acquire()
        index, actual_value = self.spec.get_index_at_wavelength(wlen)
        self.slock.release()
        return index, actual_value

    def _update_wavelength(self, n, textbox):
        if n == 1:
            line, box = self.color_vline, self.wl_text
            if self.clear_check.isChecked():
                self.color_data.clear()
                self.time.clear()
        else:
            raise ValueError()

        if textbox:
            new_value = float(box.text())
        else:
            new_value = line.value()
        index, actual_value = self._get_wavelength(new_value)
        self.lambdas[n-1] = (index, actual_value)
        line.setValue(actual_value)
        box.setText(f"{actual_value:.2f}")

    def _grab_background(self):
        if self.avg_enable.isChecked():
            self.bg_data = np.mean(self.moving_average, axis=0)
        else:
            self.desc.lock.acquire()
            self.bg_data = self.data[:,1].copy()
            self.desc.lock.release()
    
    def _clear_strips(self):
        self.color_data.clear()
        self.pca_data.clear()
        self.time.clear()

    def save_data_update(self):
        if self.saving:
            self.save_elapsed += self.time[-1]
            if self.save_elapsed >= self.save_interval_us:
                self.save_elapsed = 0
                if self.saving_average:
                    self.save_data.append(np.mean(self.moving_average, axis=0))
                else:
                    self.desc.lock.acquire()
                    self.save_data.append(self.data[:,1].copy())
                    self.desc.lock.release()

    def _saving(self):

        if self.saving:
            self.stopstart_save.setText("Start Recording")
            self.saving = False
            self.avg_enable.setEnabled(True)
            self.save_interval.setEnabled(True)

            self.desc.lock.acquire()
            wlens = self.data[:,0].copy()
            self.desc.lock.release()

            tarray = np.arange(0, len(self.save_data)) * self.save_interval_us/1e6

            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Background File", "", 
                                                   "NPZ Files (*.npz);;All Files (*)")

            if file_name:
                np.savez(file_name, spectrum_data=np.array(self.save_data), time=tarray, wavelengths=wlens)

            self.save_data = []
                
        else:
            self.stopstart_save.setText("Stop Recording")
            self.saving_average = self.avg_enable.isChecked()
            print(f"Saving average? {self.saving_average}")
            self.avg_enable.setEnabled(False)
            self.save_interval.setEnabled(False)
            self.saving=True

    def update_plots(self):
        if self.update:
            self.slock.acquire() #prevent deadlock on update_rate
            self.time.append(self.update_rate.value)
            self.slock.release()

            if self.sub_background.isChecked():
                bg_scale = 1.0 
            else:
                bg_scale = 0.0

            sdata = np.zeros(self.desc.shape[0], dtype=self.desc.dtype)
            xdata = np.zeros(self.desc.shape[0], dtype=self.desc.dtype)

            self.desc.lock.acquire()
            sdata[:] = self.data[:,1]
            xdata[:] = self.data[:,0] 
            self.desc.lock.release()

            self.color_data.append(sdata[self.lambdas[0][0]]  - bg_scale*self.bg_data[self.lambdas[0][0]])

            if self.pca_loaded:
                scaled_and_shifted = sdata - bg_scale*self.bg_data - self.pca_mean 
                self.pca_data.append(np.dot(sdata, self.pca_components.T)[0])
            else:
                self.pca_data.append(0.0)


            if self.avg_enable.isChecked():
                self.moving_average.append(sdata)
                self.avg_count_label.setText(f"Average Count: {len(self.moving_average)}/{self.moving_average.maxlen}")
                self.avg_curve.setData(xdata, np.mean(self.moving_average, axis=0)  - bg_scale*self.bg_data)


            self.spec_curve.setData(xdata, sdata  - bg_scale*self.bg_data)

            dt1 = np.cumsum(np.array(self.time))/1e6

            last_pt = None

            if len(self.color_data) > 3:
                color_smoothed = SimpleExpSmoothing(np.array(self.color_data)).fit(smoothing_level=self.smooth_level, 
                                                            optimized=False, 
                                                            use_brute=False).fittedvalues
                if self.pca_loaded:
                    pca_smoothed   = SimpleExpSmoothing(np.array(self.pca_data)).fit(smoothing_level=self.smooth_level,
                                                                optimized=False,
                                                                use_brute=False).fittedvalues
            else:
                color_smoothed = np.zeros(len(dt1))
                pca_smoothed   = np.zeros(len(dt1))             

            if self.display_pca:
                self.smoothed_curve.setData(dt1-dt1[-1], pca_smoothed)
                last_pt = pca_smoothed[-1]
                self.color_curve.setData(dt1-dt1[-1], np.array(self.pca_data))
            else:
                self.smoothed_curve.setData(dt1-dt1[-1], color_smoothed)
                last_pt = color_smoothed[-1]
                self.color_curve.setData(dt1-dt1[-1], np.array(self.color_data))

            self.save_data_update()

            self.color_min = min([self.color_min, np.min(self.color_data)])
            self.color_max = max([self.color_max, np.max(self.color_data)])

            self.min_label.setText(f"Min Value: {int(self.color_min)}")
            self.max_label.setText(f"Max Value: {int(self.color_max)}")
            self.min_hline.setValue(self.color_min)
            self.max_hline.setValue(self.color_max)

            if last_pt:
                if self.threshold_up:
                    if last_pt > self.threshold:
                        self._set_threshold_indicator(True)
                    else:
                        self._set_threshold_indicator(False)
                else:
                    if last_pt < self.threshold:
                        self._set_threshold_indicator(True)
                    else:
                        self._set_threshold_indicator(False)

    def _set_threshold_indicator(self, value):
        if value:
            self.threshold_led.setStyleSheet('background-color: green; font-weight: bold')
        else:
            self.threshold_led.setStyleSheet('background-color: red; font-weight: bold')

    def _set_spec_xlims(self):
        self.spectrum.setXRange(float(self.start_txt.text()), float(self.stop_txt.text()), padding=0)

    def _set_ylims(self):
        self.color.setYRange(float(self.ymin_txt.text()), float(self.ymax_txt.text()))
        if not self.spec_yscale.isChecked():
            self.spectrum.setYRange(float(self.ymin_txt.text()), float(self.ymax_txt.text()))

    def _autoscale_strips(self):
        if self.display_pca:
            self.color.setYRange(np.min(self.pca_data), np.max(self.pca_data), padding=0.2)
        else:
            self.color.setYRange(np.min(self.color_data), np.max(self.color_data), padding=0.2)

    def _set_smooth(self):
        new_smooth = float(self.strip_avg.text())
        if new_smooth < 0.01:
            self.smooth_level = 0.01
        elif new_smooth >= 0.99:
            self.smooth_level = 0.99
        else:
            self.smooth_level = new_smooth
        self.strip_avg.setText(f"{self.smooth_level:0.3f}")

    def _threshold_value(self, box):
        if box:
            self.threshold = int(self.threshold_txt.text())
            self.t_hline.setValue(self.threshold)
        else:
            self.threshold = int(self.t_hline.value())
            self.threshold_txt.setText(f"{self.threshold}")

    def _setup_values(self):
        self._set_spec_xlims()
        self._set_integration_time()
        self._avg_change_count()
        self._avg_enable_curve()
        self._update_wavelength(1, True)
        self._set_ylims()

        self._threshold_value(True)

    def _setup_callbacks(self):

        self.spec_yscale.stateChanged.connect(lambda: self.spectrum.enableAutoRange(axis='y', enable=self.spec_yscale.isChecked()))
        self.ymax_txt.editingFinished.connect(self._set_ylims)
        self.ymin_txt.editingFinished.connect(self._set_ylims)
        self.strip_yscale.clicked.connect(self._autoscale_strips)
        self.strip_avg.editingFinished.connect(self._set_smooth)

        self.startstop_btn.clicked.connect(self._startstop)
        self.stopstart_save.clicked.connect(self._saving)

        self.int_time_txt.editingFinished.connect(self._set_integration_time)

        self.avg_enable.stateChanged.connect(self._avg_enable_curve)
        self.avg_txt.editingFinished.connect(self._avg_change_count)
        self.avg_restart.clicked.connect(self._avg_do_restart)

        self.background.clicked.connect(self._grab_background)
        self.sub_background.stateChanged.connect(self._clear_strips)
        self.bg_set_file.clicked.connect(self._set_bg_savefile)
        self.bg_save_file.clicked.connect(self._save_background)
        self.bg_load_file.clicked.connect(self._load_background)

        self.start_txt.editingFinished.connect(self._set_spec_xlims)
        self.stop_txt.editingFinished.connect(self._set_spec_xlims)

        self.wl_text.editingFinished.connect(lambda: self._update_wavelength(1, True))

        self.color_vline.sigPositionChangeFinished.connect(lambda: self._update_wavelength(1, False))

        self.save_interval.editingFinished.connect(lambda: setattr(self, 'save_interval_us', 1000*int(self.save_interval.text())))

        self.threshold_txt.editingFinished.connect(lambda: self._threshold_value(True))
        self.t_hline.sigPositionChangeFinished.connect(lambda: self._threshold_value(False))

        self.threshold_dir.stateChanged.connect(lambda: setattr(self, 'threshold_up', self.threshold_dir.isChecked()))

        self.use_pca.stateChanged.connect(self._use_pca)
        self.get_pca_coeffs.clicked.connect(self._load_pca_components)

    def run(self, event=None):
        signal.signal(signal.SIGINT, lambda args: sigint_handler(args, event=event))

        self.make_app()
        self._setup_plots()

        self._setup_values()
        self._setup_callbacks()
            
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update_plots)
        timer.start(25)
        pg.exec()
        event.set()

def run_plotter_process(spec, slock, desc, update_rate, kill_event):
    app = SpecApp(spec, slock, desc, update_rate, False)
    app.run(event=kill_event)

def get_spectrum(spec, slock, desc, update_rate, kill_event) -> None:

    shm  = SharedMemory(name = desc.name)
    data = SHMArray(np.ndarray(desc.shape, buffer=shm.buf, dtype=desc.dtype), shm) 

    while not kill_event.is_set():
        slock.acquire()
        update_rate.value = spec.get_integration_time()
        spectrum = spec.get_spectrum()
        slock.release()
        desc.lock.acquire()
        data[:,1] = spectrum
        desc.lock.release()

    shm.close()
    return None


if __name__ == '__main__':

    manager = SpecManager()
    manager.start() 

    spec = manager.OceanSpectrometer()

    try:
        spec.connect('HR400347')
    except:
        spec.connect('FAKE')

    wavelns = spec.get_wavelengths()

    spec.set_integration_time(30000)

    shape = (len(wavelns), 2)
    dtype = 'f4'
    dummy = np.ndarray(shape, dtype=dtype)

    shm = SharedMemory(create=True, size=dummy.nbytes)
    arr = np.ndarray(shape, buffer=shm.buf, dtype=dtype)
    arr[:, 0] = wavelns
    arr[:, 1] = 0

    desc = SharedArrrayDescriptor(shm.name, shape, dtype, Lock())
    slock = Lock()

    update_rate_us = Value('i', 0)

    kill_event = Event()

    pp = Process(target=get_spectrum, args=(spec, slock, desc, update_rate_us, kill_event))
    pr = Process(target=run_plotter_process, args=(spec, slock, desc, update_rate_us, kill_event))

    pp.start()
    pr.start()
    pp.join()
    pr.join()

    shm.close()
    shm.unlink()
    manager.shutdown()
