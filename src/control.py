import time, sys, signal
from multiprocessing import Process, Lock, Value
from multiprocessing.shared_memory import SharedMemory 
from multiprocessing.managers import BaseManager
from typing import Tuple
from dataclasses import dataclass
from collections import deque

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

import numpy as np 
from pyqtgraph.Qt import QtCore, QtWidgets 
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
def sigint_handler(*args):
	sys.stderr.write('\r')
	QtWidgets.QApplication.quit()

def display_process(spec, slock, desc, update_rate) -> None:

	signal.signal(signal.SIGINT, sigint_handler)
	app = pg.mkQApp("Spectrometer Control")

	startstop_btn = pg.QtWidgets.QPushButton(text="Stop")
	background    = pg.QtWidgets.QPushButton(text="Record Background")
	sub_background= pg.QtWidgets.QCheckBox(text="Backgroun Subtract")
	start_label   = pg.QtWidgets.QLabel("Start (nm)")
	stop_label    = pg.QtWidgets.QLabel("Stop (nm)")
	start_txt     = pg.QtWidgets.QLineEdit("200.0")
	stop_txt      = pg.QtWidgets.QLineEdit("1100.0")

	spectrum = pg.PlotWidget()
	color1   = pg.PlotWidget()
	color2   = pg.PlotWidget()

	label1   = pg.QtWidgets.QLabel("Wavelength 1 (nm)")
	label2   = pg.QtWidgets.QLabel("Wavelength 2 (nm)")

	wl1      = pg.QtWidgets.QLineEdit("400.0")
	wl2      = pg.QtWidgets.QLineEdit("800.0")

	clear_check = pg.QtWidgets.QCheckBox("Clear")

	layout = pg.LayoutWidget()
	layout.addWidget(startstop_btn, row=0, col=0)
	layout.addWidget(background, row=0, col=1)
	layout.addWidget(sub_background, row=0, col=2)
	layout.addWidget(start_label, row=0, col=3)
	layout.addWidget(start_txt, row=0, col=4)
	layout.addWidget(stop_label, row=0, col=5)
	layout.addWidget(stop_txt, row=0, col=6)


	layout.addWidget(spectrum, row=1, colspan=7)

	layout.addWidget(label1, row=2, col=0)
	layout.addWidget(wl1, row=2, col=1)
	layout.addWidget(label2, row=2, col=2)
	layout.addWidget(wl2, row=2, col=3)
	layout.addWidget(clear_check, row=2, col=4)

	layout.addWidget(color1, row=3, col=0, colspan=7)

	layout.addWidget(color2, row=4, col=0, colspan=7)

	layout.resize(1800, 1200)
	layout.show()

	spec_curve = spectrum.plot(pen=pg.mkPen('r', width=1))
	spectrum.enableAutoRange('y', 1.0)
	spectrum.setLabel(axis='left', text='Counts')
	spectrum.setLabel(axis='bottom', text='Wavelength (nm)')
	spectrum.setXRange(200,1100,padding=0)

	color1_curve = color1.plot(pen=pg.mkPen('#fca503', width=1, 
											style=QtCore.Qt.PenStyle.DashLine))
	color2_curve = color2.plot(pen=pg.mkPen('#0356fc', width=1, 
											style=QtCore.Qt.PenStyle.DashLine))

	smoothed1_curve = color1.plot(pen=pg.mkPen('#fca503', width=2, 
											style=QtCore.Qt.PenStyle.SolidLine))
	smoothed2_curve = color2.plot(pen=pg.mkPen('#0356fc', width=2, 
											style=QtCore.Qt.PenStyle.SolidLine))

	color1.enableAutoRange('y', 1.0)
	color2.enableAutoRange('y', 1.0)

	color1.setXRange(-30, 10, padding=0)
	color2.setXRange(-30, 10, padding=0)

	color1.setLabel(axis='left', text='Counts')
	color1.setLabel(axis='bottom', text='Time (s)')

	color2.setLabel(axis='left', text='Counts')
	color2.setLabel(axis='bottom', text='Time (s)')


	c1_vline = pg.InfiniteLine(pos=400, angle=90, movable=True, 
		                       pen=pg.mkPen('#fca503', width=2),
		                       hoverPen=pg.mkPen('w', width=2))
	c2_vline = pg.InfiniteLine(pos=703,angle=90, movable=True, 
		                       pen=pg.mkPen('#0356fc', width=2),
		                       hoverPen=pg.mkPen('w', width=2))

	spectrum.addItem(c1_vline, ignoreBounds=True)
	spectrum.addItem(c2_vline, ignoreBounds=True)

	shm  = SharedMemory(name = desc.name)
	data = SHMArray(np.ndarray(desc.shape, buffer=shm.buf, dtype=desc.dtype), shm) 

	bg_data = np.zeros(desc.shape[0], dtype=desc.dtype) 

	c1_data = deque([], maxlen=1000)
	c2_data = deque([], maxlen=1000)
	time1   = deque([], maxlen=1000)
	time2   = deque([], maxlen=1000)

	update = True

	def startstop():
		nonlocal update
		if update:
			startstop_btn.setText("Start")
			update = False 
		else:
			startstop_btn.setText("Stop")
			update = True

	def get_wavelength(wlen):
		slock.acquire()
		index, actual_value = spec.get_index_at_wavelength(wlen)
		slock.release()
		return index, actual_value

	lambdas = [get_wavelength(400.0), get_wavelength(800.0)]

	def update_wavelength(n, textbox):
		if n == 1:
			line, box = c1_vline, wl1 
			if clear_check.isChecked():
				c1_data.clear()
				time1.clear()
		elif n == 2:
			line, box = c2_vline, wl2
			if clear_check.isChecked():
				c2_data.clear()
				time2.clear()
		else:
			raise ValueError()

		if textbox:
			new_value = float(box.text())
		else:
			new_value = line.value()
		index, actual_value = get_wavelength(new_value)
		lambdas[n-1] = (index, actual_value)
		line.setValue(actual_value)
		box.setText(f"{actual_value:.2f}")

	def grab_background():
		nonlocal bg_data
		desc.lock.acquire()
		bg_data = data[:,1].copy()
		desc.lock.release()
	
	def clear_strips():
		c1_data.clear()
		time1.clear()
		c2_data.clear()
		time2.clear()

	def update_plots():
		nonlocal bg_data
		if update:
			slock.acquire() #prevent deadlock on update_rate
			time1.append(update_rate.value)
			time2.append(update_rate.value)
			slock.release()

			if sub_background.isChecked():
				bg_scale = 1.0 
			else:
				bg_scale = 0.0

			desc.lock.acquire()
			spec_curve.setData(data[:, 0], data[:,1] - bg_scale*bg_data)
			c1_data.append(data[lambdas[0][0], 1] - bg_scale*bg_data[lambdas[0][0]])
			c2_data.append(data[lambdas[1][0], 1] - bg_scale*bg_data[lambdas[1][0]])
			desc.lock.release()

			dt1 = np.cumsum(np.array(time1))/1e6
			dt2 = np.cumsum(np.array(time2))/1e6


			if len(c1_data) > 3:
				sm1 = SimpleExpSmoothing(np.array(c1_data)).fit(smoothing_level=0.1, 
				                                            optimized=False, 
				                                            use_brute=True).fittedvalues
				smoothed1_curve.setData(dt1-dt1[-1], sm1)

			if len(c2_data) > 3:
				sm2 = SimpleExpSmoothing(np.array(c2_data)).fit(smoothing_level=0.1, 
				                                            optimized=False, 
				                                            use_brute=True).fittedvalues
				smoothed2_curve.setData(dt2-dt2[-1], sm2)

			color1_curve.setData(dt1-dt1[-1], np.array(c1_data))
			color2_curve.setData(dt2-dt2[-1], np.array(c2_data))

	def set_spec_xlims():
		spectrum.setXRange(float(start_txt.text()), float(stop_txt.text()), padding=0)

	startstop_btn.clicked.connect(startstop)
	background.clicked.connect(grab_background)
	sub_background.stateChanged.connect(clear_strips)

	start_txt.editingFinished.connect(set_spec_xlims)
	stop_txt.editingFinished.connect(set_spec_xlims)

	wl1.editingFinished.connect(lambda: update_wavelength(1, True))
	wl2.editingFinished.connect(lambda: update_wavelength(2, True))

	c1_vline.sigPositionChangeFinished.connect(lambda: update_wavelength(1, False))
	c2_vline.sigPositionChangeFinished.connect(lambda: update_wavelength(2, False))
		
	timer = QtCore.QTimer()
	timer.timeout.connect(update_plots)
	timer.start(25)
	pg.exec()
	shm.close()
	return None


def get_spectrum(spec, slock, desc, update_rate) -> None:

	shm  = SharedMemory(name = desc.name)
	data = SHMArray(np.ndarray(desc.shape, buffer=shm.buf, dtype=desc.dtype), shm) 

	while True:
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

	pp = Process(target=get_spectrum, args=(spec, slock, desc, update_rate_us))
	pr = Process(target=display_process, args=(spec, slock, desc, update_rate_us))

	pp.start()
	pr.start()
	pp.join()
	pr.join()

	shm.close()
	shm.unlink()
	manager.shutdown()