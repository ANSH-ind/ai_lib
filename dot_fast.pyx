# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
# cython: infer_types=False

cimport cython
from libcpp cimport bool
from libc.stddef cimport size_t
from libc.math cimport sin, cos, exp, log
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange

ctypedef fused dtype:
    float
    double
    
cpdef cnp.ndarray dot(cnp.ndarray matrix_a, cnp.ndarray matrix_b):
    cdef size_t ndim = matrix_a.ndim
    
    out_shape = []
    cdef size_t x
    if ndim <= 2:
        out_shape.append(matrix_a.shape[0])
    else:
        for x in range(ndim-2):
            out_shape.append(matrix_a.shape[x])
    out_shape.append(matrix_b.shape[ndim-1])
    
    cdef float[:,::1] temp_f
    cdef double[:,::1] temp_d
    
    cdef float[:,::1] tempb_f
    cdef double[:,::1] tempb_d
    
    cdef bint float_type = False
    if matrix_a.dtype == np.float32 and matrix_b.dtype == np.float32:
        temp_f = np.copy(matrix_a)
        tempb_f = np.copy(matrix_b)
        float_type = True
    elif matrix_a.dtype == np.float64 and matrix_b.dtype == np.float64:
        temp_d = np.copy(matrix_a)
        tempb_d = np.copy(matrix_b)
    else:
        raise TypeError("both matrix dtype must be same (float32 or flaot64)")
    
    cdef cnp.ndarray output_np
    
    if matrix_a.dtype == np.float32:
        output_np = np.empty((out_shape[0], out_shape[1]), dtype=np.float32)
    else:
        output_np = np.empty((out_shape[0], out_shape[1]), dtype=np.float64)
    
    cdef float[:, ::1] out_f
    cdef double[:, ::1] out_d
    
    if matrix_a.dtype == np.float32:
        out_f = output_np
    else:
        out_d = output_np
    
    cdef size_t i,j,k
    cdef size_t row, col, col_a
    cdef float sum_f
    cdef double sum_d
    
    if not matrix_a.flags['C_CONTIGUOUS'] and matrix_b.flags['C_CONTIGUOUS']:
         matrix_a = np.ascontiguousarray(matrix_a)
         matrix_b =  np.ascontiguousarray(matrix_b)
    
    if matrix_a.flags['C_CONTIGUOUS'] and matrix_b.flags['C_CONTIGUOUS']:
        matrix_a = matrix_a.reshape(-1, matrix_a.shape[ndim-1])
        matrix_b = matrix_b.reshape(-1, matrix_b.shape[ndim-1])
        row = matrix_a.shape[0]
        col = matrix_b.shape[1]
        col_a = matrix_a.shape[1]
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError("error! matrix B row must be match to matrix B col")
        
        for i in prange(row, nogil = True):
            for j in range(col):
                if float_type == 1:
                    sum_f = 0
                else:
                    sum_d = 0
                for k in range(col_a):
                    if float_type == 1:
                        sum_f += temp_f[i,k]*tempb_f[k,j]
                    else:
                        sum_d += temp_d[i,k]*tempb_d[k,j]
                if float_type == 1:
                    out_f[i,j] = sum_f
                else:
                    out_d[i,j] = sum_d
                    
                sum_f = 0.0
                sum_d = 0.0
        if float_type == 1:
            matrix_a = np.asarray(out_f)
        else:
            matrix_a = np.asarray(out_d)
        
        matrix_a = matrix_a.reshape(tuple(out_shape))
        return matrix_a
    
                    
                
            