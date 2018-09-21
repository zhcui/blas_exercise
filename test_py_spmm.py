#! /usr/bin/env python

import numpy as np
from ctypes import *
import scipy.sparse as spsp

libspmm = cdll.LoadLibrary("./libspmm.so")

def get_indptr(A):
    row = A.row
    indptr = np.zeros(A.shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(row, minlength=A.shape[0]), out=indptr[1:])
    return indptr

def mul_coo_coo(A, B):

    M, K = A.shape
    K_, N = B.shape
    assert(K == K_)

    # convert dtype to c_int and c_double
    rowIndex_A = get_indptr(A)
    columns_A = A.col.astype(np.int32)
    values_A = A.data.astype(np.double)
    
    rowIndex_B = get_indptr(B)
    columns_B = B.col.astype(np.int32)
    values_B = B.data.astype(np.double)

    # output variables
    pointerB_C = POINTER(c_int)()
    pointerE_C = POINTER(c_int)()
    columns_C = POINTER(c_int)()
    values_C = POINTER(c_double)()
    nnz = c_int(0)

    # calculation
    libspmm.spmm(byref(c_int(M)), byref(c_int(N)), byref(c_int(K)), \
            rowIndex_A.ctypes.data_as(c_void_p), columns_A.ctypes.data_as(c_void_p), values_A.ctypes.data_as(c_void_p), \
            rowIndex_B.ctypes.data_as(c_void_p), columns_B.ctypes.data_as(c_void_p), values_B.ctypes.data_as(c_void_p), \
            byref(pointerB_C), byref(pointerE_C), byref(columns_C), byref(values_C), byref(nnz))
    
    nnz = nnz.value
    
    #a = np.fromiter(values_C, dtype=np.double, count=nnz)
    #print "a"
    #print a
   
    # convert to numpy objects
    ArrayType = c_double*nnz
    addr = addressof(values_C.contents)
    values_C = np.frombuffer(ArrayType.from_address(addr), dtype = np.double, count = nnz)
    
    ArrayType = c_int*nnz
    addr = addressof(columns_C.contents)
    columns_C = np.frombuffer(ArrayType.from_address(addr), dtype = np.int32, count = nnz)
    
    ArrayType = c_int*M
    addr = addressof(pointerB_C.contents)
    pointerB_C = np.frombuffer(ArrayType.from_address(addr), dtype = np.int32, count = M)
    pointerB_C = np.append(pointerB_C, np.array(nnz, dtype = np.int32))
    # ZHC TODO NOTE Check if the memory needs deallocation!!!
    # e.g. if the numpy object is destoryed, what about the data?

    return pointerB_C, columns_C, values_C

if __name__ == '__main__':
    
    a = np.arange(16).reshape((4,4))
    b = a.T
    C_d =  a.dot(b) # dense reference
    
    print "a matrix"
    print a
    print "b matrix"
    print b
    print "c reference"
    print C_d

    A = spsp.coo_matrix(a)
    B = spsp.coo_matrix(b)

    indptr, col, data = mul_coo_coo(A, B)
    
    print "calc" 
    
    # ZHC NOTE the result can change if you comment the following three lines
    print indptr
    print col
    print data

    C = spsp.csr_matrix((data, col, indptr), shape = (4,4))
    print "c calculated from spmm"
    print C.todense()

    assert(np.allclose(C_d, C.todense()))
