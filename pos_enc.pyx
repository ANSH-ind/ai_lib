# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=True

cimport cython
from libc.stddef cimport size_t
from libc.math cimport sin, cos, exp, log
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange

cnp.import_array()

ctypedef fused dtype:
    double
    float

cdef void _kernel(dtype[:,::1] tokens, dtype base, int dtype_id, int actual_seq_len) noexcept nogil:
    cdef size_t total_rows = <size_t>tokens.shape[0]
    cdef size_t d_model = <size_t>tokens.shape[1]
    cdef size_t seq_len_st = <size_t>actual_seq_len
    cdef bint float_t = False
    
    if dtype_id == cnp.NPY_FLOAT32:
        float_t = True

    cdef double factor = -log(<double>base) / (<double>d_model)
    
    cdef double* div_term = <double*>malloc((d_model // 2 + 1) * sizeof(double))
    if div_term == NULL:
        return
        
    cdef size_t j, s
    for j in range(0, d_model, 2):
        div_term[j // 2] = exp(<double>j * factor)
        
    cdef dtype* pe_table = <dtype*>malloc(seq_len_st * d_model * sizeof(dtype))
    if pe_table == NULL:
        free(div_term)
        return

    for s in range(seq_len_st):
        for j in range(0, d_model, 2):
            if float_t:
                pe_table[s * d_model + j] = <float>sin((<float>s) * div_term[j // 2])
                if j + 1 < d_model:
                    pe_table[s * d_model + j + 1] = <float>cos((<float>s) * div_term[j // 2])
            else:
                pe_table[s * d_model + j] = <dtype>sin((<dtype>s) * div_term[j // 2])
                if j + 1 < d_model:
                    pe_table[s * d_model + j + 1] = <dtype>cos((<dtype>s) * div_term[j // 2])

    cdef size_t i, pos
    for i in prange(total_rows, schedule='static'):
        pos = i % seq_len_st
        for j in range(d_model):
            tokens[i, j] += pe_table[pos * d_model + j]

    free(div_term)
    free(pe_table)

cdef _process_double_kernel(double[:,::1] array, double base, int dtype_id, int seq_len):
    _kernel(array, base, dtype_id, seq_len)

cdef _process_float_kernel(float[:,::1] array, double base, int dtype_id, int seq_len):
    _kernel(array, base, dtype_id, seq_len)

cpdef cnp.ndarray pos_encoding(cnp.ndarray array, int axis=-1, dtype base=10000.0):
    cdef cnp.npy_intp* original_shape = array.shape
    cdef Py_ssize_t ndim = <Py_ssize_t>array.ndim
    cdef int dtype_id = array.dtype.num
    cdef int seq_len
    cdef Py_ssize_t safe_axis = <Py_ssize_t>axis

    if safe_axis >= ndim or safe_axis < -ndim:
        raise ValueError()

    if safe_axis < 0:
        safe_axis += ndim

    cdef list list_shape = []
    cdef Py_ssize_t i
    for i in range(ndim):
        list_shape.append(array.shape[i])

    if safe_axis == ndim - 1 and array.flags["C_CONTIGUOUS"]:
        if ndim >= 2:
            seq_len = <int>(list_shape[ndim - 2])
        else:
            seq_len = 1
            
        if dtype_id == cnp.NPY_FLOAT32:
            array = array.reshape(-1, array.shape[safe_axis])
            _process_float_kernel(array, base, dtype_id, seq_len)
        elif dtype_id == cnp.NPY_FLOAT64:
            array = array.reshape(-1, array.shape[safe_axis])
            _process_double_kernel(array, base, dtype_id, seq_len)
        else:
            raise TypeError()
            
        array = array.reshape(tuple(list_shape))
        return array

    array = np.swapaxes(array, safe_axis, ndim - 1)
    cdef list z = []
    cdef Py_ssize_t j
    for j in range(ndim):
        z.append(array.shape[j])

    if ndim >= 2:
        seq_len = <int>(z[ndim - 2])
    else:
        seq_len = 1

    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    array = array.reshape(-1, original_shape[safe_axis])

    if dtype_id == cnp.NPY_FLOAT32:
        _process_float_kernel(array, base, dtype_id, seq_len)
    elif dtype_id == cnp.NPY_FLOAT64:
        _process_double_kernel(array, base, dtype_id, seq_len)
    else:
        raise TypeError()

    array = array.reshape(tuple(z))
    array = np.swapaxes(array, ndim - 1, safe_axis)
    return array
    