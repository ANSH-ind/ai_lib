# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

cimport cython
from libc.math cimport exp
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange

ctypedef fused dtype:
	float
	double


cdef void _softmax_kernel(dtype[:,::1] array) noexcept nogil:

	cdef dtype dime
	cdef dtype max_val
	
	cdef Py_ssize_t row = array.shape[0]
	cdef Py_ssize_t col = array.shape[1]
	
	cdef Py_ssize_t i,j,x,z
	
	for i in prange(row,schedule='static'):
		max_val = array[i,0]
		dime = 0.0
		for j in range(col):
			if(max_val < array[i,j]):
				max_val = array[i,j]
		for x in range(col):
			dime = dime+exp(array[i,x]-max_val)
		
		for z in range(col):
			array[i, z] = exp(array[i,z]-max_val)/dime

cdef void _procces_float_kernel(cnp.ndarray array):
	cdef float[:,::1] float_array = array
	_softmax_kernel(float_array)

cdef void _procces_double_kernel(cnp.ndarray array):
	cdef double[:,::1] double_array = array
	_softmax_kernel(double_array)

cpdef cnp.ndarray softmax(cnp.ndarray array, axis = -1):
	cdef cnp.npy_intp* original_shape = array.shape
	cdef Py_ssize_t i
	ndim = array.ndim

	
	if axis >= ndim or axis < -ndim:
		raise ValueError(
    f"axis {axis} is out of bounds for array of dimension {array.ndim}"
    )
    
	if axis < 0:
		axis += ndim
	list_shape = []
	cdef Py_ssize_t j
	for i in range(ndim):
		list_shape.append(original_shape[i])
	
	if axis == array.ndim - 1 and array.flags["C_CONTIGUOUS"]:
		if array.dtype.num == cnp.NPY_FLOAT32:
			array = array.reshape(-1, array.shape[axis])
			_procces_float_kernel(array)
		elif array.dtype.num == cnp.NPY_FLOAT64:
			array = array.reshape(-1, array.shape[axis])
			_procces_double_kernel(array)
		else:
			raise TypeError(
			f"Unsupported dtype '{array.dtype}'."
		)
		array = array.reshape(tuple(list_shape))
		return array
	
	array = np.swapaxes(array, axis, -1)
	z = []
	for j in range(ndim):
		z.append(array.shape[j])
	
	if not array.flags["C_CONTIGUOUS"]:
		array = np.ascontiguousarray(array)
	
	array = array.reshape(-1, original_shape[axis])
	
	if array.dtype.num == cnp.NPY_FLOAT32:
		_procces_float_kernel(array)
	elif array.dtype.num == cnp.NPY_FLOAT64:
		_procces_double_kernel(array)
	else:
		raise TypeError(
        f"Unsupported dtype '{array.dtype}'."
    )
	
	array = array.reshape(tuple(z))
	array = np.swapaxes(array, -1, axis)
	return array

	
	
	
	
	
		

