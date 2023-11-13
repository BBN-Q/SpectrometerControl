import os, sys 
from pathlib import PurePath 
import numpy as np
import time
from numpy.typing import NDArray

try:
	od_path = PurePath(os.environ['OCEANDIRECT_HOME']) / 'Python/'
	sys.path.append(str(od_path))
	from oceandirect.OceanDirectAPI import OceanDirectAPI, OceanDirectError, Spectrometer

except Exception as e:
	raise ImportError("Could not find OceanDirect API. Is it installed?") from e 

class OceanSpectrometer(object):

	CONNECT_RETRY = 3

	def __init__(self, serial: str = ''):
		super().__init__()

		self.odapi = OceanDirectAPI()
		self.device_serials = []
		self.device_count   = 0

		self.device = None
		self.serial = None
		self.connected = False

		self.wavelengths = []

		self.get_serials()

		if serial:
			self.connect(serial)

	def __del__(self):
		self.disconnect()
		self.odapi.shutdown()

	def get_serials(self) -> None:

		try:
			self.device_count = self.odapi.find_usb_devices()
			if self.device_count > 0:
				dev_ids = self.odapi.get_device_ids()
				for did in dev_ids:
					device = self.odapi.open_device(did)
					self.device_serials.append(device.get_serial_number())  
					device.close_device()
		except OceanDirectError as err:
			[errorCode, errorMsg] = err.get_error_details()
			raise RuntimeError(f"ODAPI Error: {errorCode}, {errorMsg}") from err

	def get_connected_serials(self):
		return self.device_serials

	def disconnect(self) -> None:
		if self.connected:
			self.device.close_device()
		self.device = None 
		self.serial = None

	def connect(self, serial: str) -> None:

		if self.serial == serial:
			return

		self.disconnect()
		
		if self.odapi.find_usb_devices() != self.device_count:
			self.get_serials()
		
		if self.device_count == 0 or serial not in self.device_serials:
			raise ValueError(f"Device {serial} not connected!")

		try:
			retry = 0
			dev_ids = self.odapi.get_device_ids()
			while not self.connected and retry < self.CONNECT_RETRY:
				for did in dev_ids:
					device = self.odapi.open_device(did)
					if device.get_serial_number() == serial:
						self.device = device
						self.connected = True
						break 
					else:
						device.close_device()
				if self.connected:
					break 
				else:
					time.sleep(1)
					self.device_count = self.odapi.find_usb_devices()
					dev_ids = self.odapi.get_device_ids()
					retry += 1

		except OceanDirectError as err:
			[errorCode, errorMsg] = err.get_error_details()
			raise RuntimeError(f"ODAPI Error: {errorCode}, {errorMsg}") from err

		if not self.connected:
			raise RuntimeError(f"Could not connect to {serial}!")

		self.serial = serial

		self.integration_time_limits = (self.device.get_minimum_integration_time(),  
										self.device.get_maximum_integration_time()) 

		self.wavelengths = np.array(self.device.get_wavelengths())

		self.get_index_at_wavelength = self.device.get_index_at_wavelength 
		self.get_indices_at_wavelengths = self.device.get_indices_at_wavelengths

		self.device.use_nonlinearity(True)

	def get_integration_time(self) -> int:
		return self.device.get_integration_time()

	def set_integration_time(self, int_time: int) -> None:
		self.device.set_integration_time(int_time)

	def set_average_count(self) -> int:
		return self.device.get_scans_to_average()

	def get_average_count(self, count: int) -> None:
		self.device.set_scans_to_average(count)

	def get_spectrum(self) -> NDArray:
		return np.array(self.device.get_formatted_spectrum())

	def get_wavelengths(self) -> NDArray:
		return self.wavelengths

	def get_index_at_wavelength(self, wlen: float) -> tuple[int, float]:
		return self.device.get_index_at_wavelength(wlen)

	def get_indicies_at_wavelengths(self, wlens: list[float]) -> tuple[list[int], list[float]]:
		return self.device.get_indices_at_wavelengths(wlens)









