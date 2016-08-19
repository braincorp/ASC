# ============================================================================
# Copyright 2015 BRAIN Corporation. All rights reserved. This software is
# provided to you under BRAIN Corporation's Beta License Agreement and
# your use of the software is governed by the terms of that Beta License
# Agreement, found at http://www.braincorporation.com/betalicense.
# ============================================================================
import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel, threadid

import scipy.sparse

cdef extern from *:
    pass


cdef extern from "xmmintrin.h":
    void _mm_setcsr(unsigned int)
    unsigned int _mm_getcsr()


def set_flush_denormals():
    """
    This call will modify the Control Status Register (CSR) to instruct the CPU to flush denormals.

    Very small numbers are treated differently in order to gain a small amount of extra precision, however,
    this extra precision comes with very significant computational cost.  By flushing denormals to 0,
    we lose a small amount of precision but now all arithmetic operations run with consistent speed.

    This code only works for X86 SSE.  Flush to zero is default on ARM architecture.
    """
    _mm_setcsr((_mm_getcsr() & ~0x0040) | (0x0040))


# a is a vector of length N
# B is a matrix of MxN (C) or NxM (Fortran)
# o is a preallocated vector of length M
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void dot(const int N, const double *a, const int M, const double *B, double *o) nogil:
    cdef double dot1, dot2, dot3, dot4
    cdef int i, j
    cdef const double *Bj
    
    for j in range(M):
        Bj = &B[j*N]
        i = 0
        dot1 = dot2 = dot3 = dot4 = 0
        while i < N-3:
            dot1 += a[i] * Bj[i];
            dot2 += a[i + 1] * Bj[i + 1];
            dot3 += a[i + 2] * Bj[i + 2];
            dot4 += a[i + 3] * Bj[i + 3];
            i += 4
        
        while i < N:
            dot1 += a[i] * Bj[i];
            i += 1

        o[j] = dot1 + dot2 + dot3 + dot4;


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double single_ASC(const unsigned int K, const unsigned int N, const double* D, const double* DtD, const double* x, unsigned int target_l0, double* a, double* z, double ave_stop_l) nogil:
    cdef unsigned int i, j, num_nonzero, cnt, iK
    cdef double ai, ainew, l, aidelta, start_l
    
    for i in range(N):
        if x[i]:
            break
    else: # if x all zeros
        return ave_stop_l
    
    dot(N, x, K, D, z)

    l = 0.0
    for i in xrange(K):
        if l < z[i]:
            l = z[i]
    
    start_l = l
    
    if ave_stop_l <= 0.0:
        ave_stop_l = 1.0
    
    l *= ave_stop_l

    cnt = 0
    num_nonzero = 0
    while (num_nonzero != target_l0) and l > 0.0 and cnt < 25:
        num_nonzero = 0
        for i in xrange(K):
            ai = a[i]
            ainew = ai + z[i] - l
            if ainew < 0:
                ainew = 0
                
            if ai != ainew:
                a[i] = ainew
                iK = i*K
                aidelta = ainew-ai
                for j in xrange(K):
                    z[j] -= DtD[iK+j]*aidelta
                if ainew != 0:
                    num_nonzero += 1
        cnt += 1
        l *= (1.0+2.0/cnt) if num_nonzero > target_l0 else (1.0-0.75/cnt)
    return l/start_l if start_l > 0 else ave_stop_l


@cython.boundscheck(False)
@cython.wraparound(False)
def ASC(np.ndarray[double, ndim=2,] D, np.ndarray[double, ndim=2,] DtD, np.ndarray[double, ndim=2,] X, unsigned int target_l0, unsigned int num_threads=1, bint add_one=False, double ave_stop_l=1.0, np.ndarray inv_std=None):
    cdef unsigned int K, N, M, 
    cdef int i, tid, j, k, k_stop
    cdef np.ndarray[double, ndim=2,] A, Z, stop_ls
    cdef double* Ap
    cdef double* Zp
    cdef double* Xp
    cdef double* DtDp
    cdef double* Dp
    cdef double * stop_lsp
    cdef double cum_stop_ls = 0
    
    K = D.shape[1]
    N = X.shape[0]
    M = X.shape[1]
    
    cdef np.ndarray[double, ndim=1,] data = np.zeros((target_l0+1)*M)
    cdef np.ndarray[unsigned int, ndim=1,] rows = np.zeros((target_l0+1)*M, dtype=np.uint32)
    cdef np.ndarray[unsigned int, ndim=1,] indptr = np.arange(0,(target_l0+1)*M+1,(target_l0+1), dtype=np.uint32)
    
    cdef double* datap = &data[0]
    cdef unsigned int* rowsp = &rows[0]
    cdef double* inv_stdp = NULL
    
    if inv_std is not None:
        assert inv_std.size >= K
        inv_stdp = <double*>inv_std.data
    
    D=np.asfortranarray(D)
    X=np.asfortranarray(X)
    # DtD is symatric so C or Fortran ordering is fine
    
    A = np.zeros((X.shape[1], K))
    Z = np.zeros((num_threads, K))
    stop_ls = np.zeros((num_threads, 1))
    Ap=&A[0,0]
    Zp=&Z[0,0]
    Xp=&X[0,0]
    DtDp=&DtD[0,0]
    Dp=&D[0,0]
    stop_lsp = &stop_ls[0,0]
    with nogil, parallel(num_threads=num_threads):
        for i in prange(M,schedule='dynamic'):
            tid = threadid()
            stop_lsp[tid] = single_ASC(K, N, Dp, DtDp, &Xp[i*N], target_l0, &Ap[i*K], &Zp[tid*K], ave_stop_l)
            k = i*(target_l0+1)
            k_stop = k + target_l0
            for j in xrange(K):
                if Ap[i*K+j] != 0.0:
                    datap[k] = Ap[i*K+j] if inv_stdp == NULL else Ap[i*K+j] * inv_stdp[j]
                    rowsp[k] = j
                    k = k + 1
                    if k == k_stop:
                        break
            if add_one:
                datap[k] = 1.0
                rowsp[k] = K
    with nogil:
        for i in xrange(num_threads):
            cum_stop_ls += stop_lsp[i]
        ave_stop_l = cum_stop_ls/num_threads
    return scipy.sparse.csc_matrix((data, rows, indptr),shape=((K+1) if add_one else K, M)), ave_stop_l


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_dense_add_a_dot_at(np.ndarray[double, ndim=2,] A, np.ndarray[int, ndim=2,] rows, np.ndarray[double, ndim=2,] data, unsigned int num_threads=1):
    """ Computes A += D.dot(D.T), where D is a sparse matrix represented by matrices 'rows' (D.indices) and 'data' (D.data)."""
    cdef int num_rows = rows.shape[0]
    cdef int num_elements = rows.shape[1]
    cdef int K = A.shape[0]
    cdef int i, j, k, ind, tid
    cdef np.ndarray[int, ndim=2,] row_inds = np.zeros((num_threads, num_rows), dtype=np.int32)
    cdef double d_ind
    
    cdef int* rowsp = &rows[0, 0]
    cdef double* datap = &data[0, 0]
    cdef double* Ap = &A[0, 0]
    cdef int* row_indsp = &row_inds[0, 0]
    
    with nogil, parallel(num_threads=num_threads):
        for i in prange(K, schedule='dynamic'):
            tid = threadid()
            for j in range(num_rows):
                # find row i amongst the rows, exploit the fact that rows are sorted and appear only once
                ind = row_indsp[tid*num_rows+j]
                while rowsp[j*num_elements+ind] < i:
                    ind = ind + 1
                if rowsp[j*num_elements+ind] == i:
                    row_indsp[tid*num_rows+j] = ind + 1
                    
                    d_ind = datap[j*num_elements+ind]
                    for k in range(num_elements):
                        Ap[i*K + rowsp[j*num_elements+k]] += d_ind * datap[j*num_elements+k]


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_dense_add_dense_dot_at(np.ndarray[double, ndim=2,] A, np.ndarray[double, ndim=2,] X, np.ndarray[int, ndim=2,] rows, np.ndarray[double, ndim=2,] data, unsigned int num_threads=1):
    """ Computes A += D.dot(X.T).T, where D is a sparse matrix represented by matrices 'rows' (D.indices) and 'data' (D.data). """
    cdef int num_rows = rows.shape[0]
    cdef int num_elements = rows.shape[1]
    cdef int K = A.shape[1]
    cdef int M = X.shape[0]  # number of inputs
    cdef int i, j, k, ind, tid
    cdef np.ndarray[int, ndim=2,] row_inds = np.zeros((num_threads, num_rows), dtype=np.int32)
    cdef double d_ind
    
    assert A.shape[0] == M and A.shape[1] == K, "%d %d %d %d" % (A.shape[0], M, A.shape[1], K)
    assert X.shape[0] == M and X.shape[1] == num_rows, "%d %d %d %d" % (X.shape[0], M, X.shape[1], num_rows)
    
    X = np.asfortranarray(X)  # make sure the first dimension of X is contiguous so that for Xp[j*M+k], k indexes along the first dimension, instead of the second for normal numpy arrays
    A = np.asfortranarray(A)  # make sure the first dimension of A is contiguous so that for Ap[i*M+k], k indexes along the first dimension, instead of the second for normal numpy arrays
    
    cdef int* rowsp = &rows[0, 0]
    cdef double* datap = &data[0, 0]
    cdef double* Ap = &A[0, 0]
    cdef double* Xp = &X[0, 0]
    cdef int* row_indsp = &row_inds[0, 0]
    
    with nogil, parallel(num_threads=num_threads):
        for i in prange(K, schedule='dynamic'):
            tid = threadid()
            for j in range(num_rows):
                # find row i amongst the rows, exploit the fact that rows are sorted and appear only once
                ind = row_indsp[tid*num_rows+j]
                while rowsp[j*num_elements+ind] < i:
                    ind = ind + 1
                if rowsp[j*num_elements+ind] == i:
                    row_indsp[tid*num_rows+j] = ind + 1
                    
                    d_ind = datap[j*num_elements+ind]
                    for k in range(M):
                        Ap[i*M+k] += d_ind * Xp[j*M+k]
    return A

    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def parallel_dot(np.ndarray[double, ndim=2,] A, np.ndarray[double, ndim=2,] X, unsigned int num_threads=1, str AX_order='C', AX=None):
    """ Compute matrix multiply, A*X, in parallel and returns the solution in provided matrix AX.  if AX is None then a new matrix will be created."""
    cdef int An = A.shape[0]
    cdef int Am = A.shape[1]
    cdef int Xn = X.shape[0]
    cdef int Xm = X.shape[1]
    cdef int i, j, k
    cdef int A_order_C = not np.isfortran(A)
    cdef int X_order_C = not np.isfortran(X)
    
    cdef np.ndarray[double, ndim=2,] AX_local
    if AX is None:
        AX_local = np.zeros((An, Xm), order=AX_order)
    else:
        assert AX.shape[0] == An and AX.shape[1] == Xm
        AX_local = AX
    
    assert Am == Xn

    cdef double* Ap = &A[0, 0]
    cdef double* AXp = &AX_local[0, 0]
    cdef double* Xp = &X[0, 0]
    cdef double cum
    cdef int AX_order_C = AX_order == 'C'
    
    with nogil, parallel(num_threads=num_threads):
        for i in prange(An, schedule='dynamic'):
            for j in range(Xm):
                cum = 0
                if A_order_C:
                    if X_order_C:
                        for k in range(Xn):  # Xn == Am
                            cum = cum + Ap[i*Am+k] * Xp[j+k*Xm]
                    else:
                        for k in range(Xn):  # Xn == Am
                            cum = cum + Ap[i*Am+k] * Xp[j*Xn+k]
                else:
                    if X_order_C:
                        for k in range(Xn):  # Xn == Am
                            cum = cum + Ap[k*An+i] * Xp[j+k*Xm]
                    else:
                        for k in range(Xn):  # Xn == Am
                            cum = cum + Ap[k*An+i] * Xp[j*Xn+k]
                if AX_order_C:
                    AXp[i*Xm+j] = cum
                else:
                    AXp[j*An+i] = cum
    
    return AX_local
    
@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def add_scale(np.ndarray[double, ndim=2,] A, np.ndarray[double, ndim=2,] X, double scale, unsigned int num_threads=1):
    """ Computes: A += X*scale, where A and X are matrices of the same size and data order. """
    cdef double* Ap = &A[0, 0]
    cdef double* Xp = &X[0, 0]
    cdef int N = A.shape[0] * A.shape[1]

    assert np.isfortran(A) == np.isfortran(X)
    assert N == X.shape[0] * X.shape[1]
    
    with nogil:
        for i in range(N):
            Ap[i] += Xp[i]*scale
