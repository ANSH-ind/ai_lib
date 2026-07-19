# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

cimport cython
from libc.stddef cimport size_t
from libc.math cimport exp, sqrt, M_PI, log
#from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange


ctypedef fused dtype:
	float
	double

cdef inline dtype e_max(dtype a) noexcept nogil:
	cdef dtype b = <dtype>0
	if a > b:
		return a
	else:
		return b

cdef inline dtype e_min(dtype a) noexcept nogil:
	cdef dtype b = <dtype>6
	if a < b:
		return a
	else:
		return b

cdef inline dtype leaky(dtype x, dtype alpha) noexcept nogil:
	if x<0:
		return alpha*x
	else:
		return x

cdef inline dtype tanh_x(dtype x) noexcept nogil:
	return (exp(x)-exp(-x))/(exp(x)+exp(-x))

cdef inline dtype sigmoid_x(dtype a) noexcept nogil:
	return <dtype>1.0/(<dtype>1.0+exp(-a))

cdef void _ReLU_kernel(dtype[:,::1] array) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			array[i,j] = e_max(array[i,j])


cdef void _ReLU6_kernel(dtype[:,::1] array) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			array[i,j] = e_min(e_max(array[i,j]))
			
cdef void _LeakyReLU_kernel(dtype[:,::1] array, dtype alpha) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			array[i,j] = leaky(array[i,j], alpha)


cdef void _tanh_kernel(dtype[:,::1] array) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			array[i,j] = (exp(array[i,j])-exp(-array[i,j]))/(exp(array[i,j])+exp(-array[i,j]))


cdef void _GeLU_kernel(dtype[:,::1] array) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	cdef dtype x
	cdef dtype x2
	cdef dtype x3
	cdef dtype SQRT_2_PI = <dtype>0.7978845608028654
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			x = array[i, j]
			x2 = x * x
			x3 = x2 * x
			array[i,j] = <dtype>0.5 * x * (<dtype>1.0 + tanh_x(SQRT_2_PI * (x + <dtype>0.044715 * x3)))


cdef void _sigmoid_kernel(dtype[:,::1] array) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			array[i,j] = <dtype>1.0/(<dtype>1.0+exp(-array[i,j]))

cdef void _softplus_kernel(dtype[:,::1] array) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			array[i,j] = log(<dtype>1.0+exp(array[i,j]))

cdef void _swish_kernel(dtype[:,::1] array, dtype beta) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			array[i,j] = array[i,j]*sigmoid_x(beta * array[i,j])


cdef void _mish_kernel(dtype[:,::1] array) noexcept nogil:
	cdef size_t i,j
	cdef size_t row,col
	
	row = <size_t>array.shape[0]
	col = <size_t>array.shape[1]
	
	for i in prange(row):
		for j in range(col):
			array[i,j] = array[i,j] * tanh_x(log(<dtype>1+exp(array[i,j])))
			
#ReLU float32 processor
cdef void _process_kenrel_float(float[:,::1] array):
	_ReLU_kernel(array)
#ReLU float64 processor
cdef void _process_kenrel_double(double[:,::1] array):
	_ReLU_kernel(array)
	
#ReLU6 float32 processor	
cdef void _process_ReLU6_kenrel_float(float[:,::1] array):
	_ReLU6_kernel(array)

#ReLU6 float64 processor
cdef void _process_ReLU6_kenrel_double(double[:,::1] array):
	_ReLU6_kernel(array)

#LeakyReLU float32 processor
cdef void _process_LeakyReLU_kenrel_float(float[:,::1] array, float alpha):
	_LeakyReLU_kernel(array, alpha)

#LeakyReLU float64 processor
cdef void _process_LeakyReLU_kenrel_double(double[:,::1] array, float alpha):
	_LeakyReLU_kernel(array, alpha)

#tanh float32 processor
cdef void _process_tanh_kenrel_float(float[:,::1] array):
	_tanh_kernel(array)

#tanh float64 processor
cdef void _process_tanh_kenrel_double(double[:,::1] array):
	_tanh_kernel(array)
	
#GeLU float32 processor
cdef void _process_GeLU_kenrel_float(float[:,::1] array):
	_GeLU_kernel(array)
#GeLU float64 processor
cdef void _process_GeLU_kenrel_double(double[:,::1] array):
	_GeLU_kernel(array)

#sigmoid float32 processor
cdef void _process_sigmoid_kenrel_float(float[:,::1] array):
	_GeLU_kernel(array)
#sigmoid float64 processor
cdef void _process_sigmoid_kenrel_double(double[:,::1] array):
	_GeLU_kernel(array)


#softplus float32 processor
cdef void _process_softplus_kenrel_float(float[:,::1] array):
	_GeLU_kernel(array)
#softplus float64 processor
cdef void _process_softplus_kenrel_double(double[:,::1] array):
	_GeLU_kernel(array)


#swish float32 processor
cdef void _process_swish_kenrel_float(float[:,::1] array, float beta):
	_swish_kernel(array, beta)
#swish float64 processor
cdef void _process_swish_kenrel_double(double[:,::1] array, double beta):
	_swish_kernel(array, beta)
	

#mish float32 processor
cdef void _process_mish_kenrel_float(float[:,::1] array):
	_mish_kernel(array)
#mish float64 processor
cdef void _process_mish_kenrel_double(double[:,::1] array):
	_mish_kernel(array)
	

#ReLU Wrapper ---------ReLU------------
cpdef cnp.ndarray ReLU(cnp.ndarray array):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_kenrel_float(array)
		elif array.dtype == np.float64:
			_process_kenrel_double(array)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_kenrel_float(array)
	elif array.dtype == np.float64:
		_process_kenrel_double(array)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array

#RELU6 wrapper ---------ReLU6-------------
cpdef cnp.ndarray ReLU6(cnp.ndarray array):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_ReLU6_kenrel_float(array)
		elif array.dtype == np.float64:
			_process_ReLU6_kenrel_double(array)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_ReLU6_kenrel_float(array)
	elif array.dtype == np.float64:
		_process_ReLU6_kenrel_double(array)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array
	
	
#LeakyReLU
cpdef cnp.ndarray LEAKYReLU(cnp.ndarray array, float alpha = 0.01):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_LeakyReLU_kenrel_float(array, alpha)
		elif array.dtype == np.float64:
			_process_LeakyReLU_kenrel_double(array, alpha)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_LeakyReLU_kenrel_float(array, alpha)
	elif array.dtype == np.float64:
		_process_LeakyReLU_kenrel_double(array, alpha)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array

#tanh -------tanh activation------------


cpdef cnp.ndarray tanh(cnp.ndarray array):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_tanh_kenrel_float(array)
		elif array.dtype == np.float64:
			_process_tanh_kenrel_double(array)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_tanh_kenrel_float(array)
	elif array.dtype == np.float64:
		_process_tanh_kenrel_double(array)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array


#GeLU wrapper -------GeLU------------
cpdef cnp.ndarray GeLU(cnp.ndarray array):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_GeLU_kenrel_float(array)
		elif array.dtype == np.float64:
			_process_GeLU_kenrel_double(array)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_GeLU_kenrel_float(array)
	elif array.dtype == np.float64:
		_process_GeLU_kenrel_double(array)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array


#sigmoid wrapper ------sigmoid--------

cpdef cnp.ndarray sigmoid(cnp.ndarray array):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_sigmoid_kenrel_float(array)
		elif array.dtype == np.float64:
			_process_sigmoid_kenrel_double(array)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_sigmoid_kenrel_float(array)
	elif array.dtype == np.float64:
		_process_sigmoid_kenrel_double(array)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array

#softplus wrapper
cpdef cnp.ndarray softplus(cnp.ndarray array):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_softplus_kenrel_float(array)
		elif array.dtype == np.float64:
			_process_softplus_kenrel_double(array)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_softplus_kenrel_float(array)
	elif array.dtype == np.float64:
		_process_softplus_kenrel_double(array)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array
	
#swish wrappe	
cpdef cnp.ndarray swish(cnp.ndarray array, dtype beta = 1.0):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_swish_kenrel_float(array, beta)
		elif array.dtype == np.float64:
			_process_swish_kenrel_double(array, beta)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_swish_kenrel_float(array, beta)
	elif array.dtype == np.float64:
		_process_swish_kenrel_double(array, beta)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array

#mish wrapper
cpdef cnp.ndarray mish(cnp.ndarray array):
	cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
	
	original_shape = []
	for i in range(ndim):
		original_shape.append(array.shape[i])
		
	if array.flags['C_CONTIGUOUS']:
		array = array.reshape(-1, original_shape[ndim-1])
		if array.dtype == np.float32:
			_process_mish_kenrel_float(array)
		elif array.dtype == np.float64:
			_process_mish_kenrel_double(array)
		else:
			raise TypeError("input must be float32 or float64")
		array = array.reshape(tuple(original_shape))
		return array
	
	array = np.ascontiguousarray(array)
	array = array.reshape(-1, original_shape[ndim-1])
	
	if array.dtype == np.float32:
		_process_mish_kenrel_float(array)
	elif array.dtype == np.float64:
		_process_mish_kenrel_double(array)
	else:
		raise TypeError("input must be float32 or float64")
	
	array = array.reshape(tuple(original_shape))
	return array
	
	