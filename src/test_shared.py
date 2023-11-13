from multiprocessing import Process, Lock
from multiprocessing.shared_memory import SharedMemory 
from typing import Tuple
from dataclasses import dataclass

import time

import numpy as np 


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

def writer_process(desc):
	shm = SharedMemory(name=desc.name)
	arr = SHMArray(np.ndarray(desc.shape, buffer=shm.buf, dtype=desc.dtype), shm)
	for j in range(10):
		desc.lock.acquire()
		arr[:] += 5
		desc.lock.release()
		time.sleep(0.1)
	shm.close()


def reader_process(desc):
	shm = SharedMemory(name=desc.name)
	arr = SHMArray(np.ndarray(desc.shape, buffer=shm.buf, dtype=desc.dtype), shm)
	for j in range(20):
		desc.lock.acquire()
		print(arr)
		desc.lock.release()
		time.sleep(0.05)
	shm.close()


if __name__ == "__main__":
	shape = (10,)
	dtype = 'f4'
	dummy_array = np.ndarray(shape, dtype=dtype)

	shm = SharedMemory(create=True, size=dummy_array.nbytes)
	arr = np.ndarray(shape, buffer=shm.buf, dtype=dtype)
	arr[:]  = 0 

	desc = SharedArrrayDescriptor(shm.name, shape, dtype, Lock())

	pp = Process(target=writer_process, args=(desc,))
	pr = Process(target=reader_process, args=(desc,))


	pp.start()
	pr.start()
	pp.join()
	pr.join()

	shm.close()
	shm.unlink()