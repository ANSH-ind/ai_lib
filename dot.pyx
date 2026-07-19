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

ctypedef fused dtype:
    float
    double



cdef dtype[:,::1] _dot_kernal(dtype[:,::1] matrix_A, dtype[:,::1] matrix_B, dtype[:,::1] output) noexcept nogil:
    cdef size_t row, col, col_A
    cdef size_t i,j, k, x
    
    
    #cdef size_t ndim = <size_t>matrix_A.ndim
    row = <size_t>matrix_A.shape[0]
    col_A = <size_t>matrix_A.shape[1]
    col = <size_t>matrix_B.shape[1]
    cdef dtype sum
    
    for i in prange(row):
        for j in range(col):
            sum = <dtype>0
            for k in range(col_A):
                sum = sum + matrix_A[i,k]*matrix_B[k,j]
            output[i, j] = sum
            sum = <dtype>0
    return output
    
cdef float[:,::1] _dot_kernal_float(float[:,::1] matrix_A, float[:,::1] matrix_B):
    cdef cnp.ndarray output_np = np.empty((matrix_A.shape[0], matrix_B.shape[1]),dtype=np.float32)
    cdef float[:,::1] output = output_np
    
    cdef float[:,::1] out
    out = _dot_kernal(matrix_A, matrix_B, output)
    return out


cdef double[:,::1] _dot_kernal_double(double[:,::1] matrix_A, double[:,::1] matrix_B):
    cdef cnp.ndarray output_np = np.empty((matrix_A.shape[0], matrix_B.shape[1]),dtype=np.float64)
    cdef double[:,::1] output = output_np
    cdef double[:,::1] out
    out = _dot_kernal(matrix_A, matrix_B, output)
    return out


cpdef cnp.ndarray dot(cnp.ndarray matrix_A, cnp.ndarray matrix_B):
    
    cdef float[:,::1] final_output_f_ma
    cdef double[:,::1] final_output_dma
    
    cdef float[:,::1] final_output_fmb
    cdef double[:,::1] final_output_dmb
    
    cdef float[:,::1] final_output_f
    cdef double[:,::1] final_output_d
    cdef Py_ssize_t ndim = <Py_ssize_t>matrix_A.ndim
    A_original_shape = []
    
    for i in range(ndim):
        A_original_shape.append(matrix_A.shape[i])
    B_original_shape = []
    
    for i in range(ndim):
        B_original_shape.append(matrix_B.shape[i])
        
    output_shape = []
    if ndim == 2:
        output_shape.append(A_original_shape[0])
    else:
        for i in range(ndim-2):
            output_shape.append(A_original_shape[i])
    
    output_shape.append(B_original_shape[ndim-1])
    
    cdef cnp.ndarray np_output = np.empty((output_shape[0], output_shape[1]), dtype = np.float32 if matrix_A.dtype == np.float32 else np.float64)

    if matrix_A.flags['C_CONTIGUOUS'] and matrix_B.flags['C_CONTIGUOUS']:
        matrix_A = matrix_A.reshape(-1, A_original_shape[ndim-1])
        matrix_B = matrix_B.reshape(-1, B_original_shape[ndim-1])
        
        if matrix_A.shape[1] != matrix_B.shape[0]:
            raise ValueError(f"matrix A column and matrix B row must be same and matrix A row and matrix B column must be same {tuple(B_original_shape)} : {tuple(A_original_shape)}")
        
        if matrix_A.dtype == np.float32 and matrix_B.dtype == np.float32:
            final_output_f_ma = matrix_A
            final_output_fmb = matrix_B
            final_output_f = _dot_kernal_float(final_output_f_ma, final_output_fmb)
            np_output = np.asarray(final_output_f)
        elif matrix_A.dtype == np.float64 and matrix_B.dtype == np.float64:
            final_output_dma = matrix_A
            final_output_dmb = matrix_B
            final_output_d = _dot_kernal_double(final_output_dma, final_output_dmb)
            np_output = np.asarray(final_output_d)
        else:
            raise TypeError("input must be float32 or float64")
        np_output = np_output.reshape(tuple(output_shape))
        return np_output
        
    if not matrix_A.flags['C_CONTIGUOUS'] or not matrix_B.flags['C_CONTIGUOUS']:
        matrix_A = np.ascontiguousarray(matrix_A)
        matrix_B = np.ascontiguousarray(matrix_B)
    
    matrix_A = matrix_A.reshape(-1, A_original_shape[ndim-1])
    matrix_B = matrix_B.reshape(-1, B_original_shape[ndim-1])
    
    if matrix_A.dtype == np.float32 and matrix_B.dtype == np.float32:
        final_output_f_ma = matrix_A
        final_output_fmb = matrix_B
        final_output_f = _dot_kernal_float(final_output_f_ma, final_output_fmb)
        np_output = np.asarray(final_output_f)

    elif matrix_A.dtype == np.float64 and matrix_B.dtype == np.float64:
        final_output_dma = matrix_A
        final_output_dmb = matrix_B
        final_output_d = _dot_kernal_double(final_output_dma, final_output_dmb)
        np_output = np.asarray(final_output_d)
    else:
        raise ValueError("the dtype of matrix_A and matrix_B must be same or have float32 or float64")
    
    np_output = np_output.reshape(tuple(output_shape))
    
    return np_output
    