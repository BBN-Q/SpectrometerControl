# Form implementation generated from reading ui file 'control.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_SpectrumControl(object):
    def setupUi(self, SpectrumControl):
        SpectrumControl.setObjectName("SpectrumControl")
        SpectrumControl.resize(1093, 957)
        self.centralwidget = QtWidgets.QWidget(parent=SpectrumControl)
        self.centralwidget.setObjectName("centralwidget")
        self.spectrum = PlotWidget(parent=self.centralwidget)
        self.spectrum.setGeometry(QtCore.QRect(10, 10, 971, 401))
        self.spectrum.setObjectName("spectrum")
        self.color1 = PlotWidget(parent=self.centralwidget)
        self.color1.setGeometry(QtCore.QRect(10, 420, 971, 241))
        self.color1.setObjectName("color1")
        self.color1_2 = PlotWidget(parent=self.color1)
        self.color1_2.setGeometry(QtCore.QRect(0, 0, 971, 241))
        self.color1_2.setObjectName("color1_2")
        self.color2 = PlotWidget(parent=self.centralwidget)
        self.color2.setGeometry(QtCore.QRect(10, 670, 971, 241))
        self.color2.setObjectName("color2")
        self.wavelength1_input = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.wavelength1_input.setGeometry(QtCore.QRect(990, 440, 51, 22))
        self.wavelength1_input.setObjectName("wavelength1_input")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(990, 420, 71, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(990, 670, 71, 16))
        self.label_2.setObjectName("label_2")
        self.wavelength2_input = QtWidgets.QLineEdit(parent=self.centralwidget)
        self.wavelength2_input.setGeometry(QtCore.QRect(990, 690, 51, 22))
        self.wavelength2_input.setObjectName("wavelength2_input")
        SpectrumControl.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=SpectrumControl)
        self.statusbar.setObjectName("statusbar")
        SpectrumControl.setStatusBar(self.statusbar)

        self.retranslateUi(SpectrumControl)
        QtCore.QMetaObject.connectSlotsByName(SpectrumControl)

    def retranslateUi(self, SpectrumControl):
        _translate = QtCore.QCoreApplication.translate
        SpectrumControl.setWindowTitle(_translate("SpectrumControl", "MainWindow"))
        self.label.setText(_translate("SpectrumControl", "Wavelength"))
        self.label_2.setText(_translate("SpectrumControl", "Wavelength"))
from pyqtgraph import PlotWidget