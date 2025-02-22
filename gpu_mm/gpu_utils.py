import numpy as np
import ctypes
import os

mylib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "lib", "libgpu_mm.so"))

gpu_alloc=mylib.alloc_arr
gpu_alloc.argtypes=(ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p)

togpu=mylib.togpu
togpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p)

fromgpu=mylib.fromgpu
fromgpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p)

gpufree=mylib.free_ptr
#gpufree.argtypes=(ctypes.c_void_p)
gpufree.argtypes=(ctypes.c_void_p,)

sgemm_gpu=mylib.sgemm
sgemm_gpu.argtypes=(ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)
#void sgemm(int transa, int transb, int n, int m, int k, long *A, long *B, long *C)



class GPUvec:
    def __init__(self,arr=None,shape=None,dtype=None):
        if not(arr is None):
            shape=arr.shape
            dtype=arr.dtype
        self.shape=shape
        self.dtype=dtype
        self.nelem=np.prod(self.shape)
        self.nbyte=self.nelem*self.dtype.itemsize
        ptr=np.zeros(1,dtype='int64')
        self.failed=np.zeros(1,dtype='int64')
        gpu_alloc(ptr.ctypes.data,self.nbyte,self.failed.ctypes.data)
        if not(self.failed[0]==0):
            print('malloc failure on GPU')
            self.ptr=None
            return
        else:
            self.ptr=ptr

        if not(arr is None):
            self.togpu(arr)
    def togpu(self,arr):
        nbyte=np.prod(arr.shape)*arr.dtype.itemsize
        if nbyte==self.nbyte:
            togpu(self.ptr.ctypes.data,arr.ctypes.data,self.nbyte,self.failed.ctypes.data)
            if self.failed[0]:
                print('Failure in copying array to gpu.')
    def fromgpu(self):
        out=np.empty(self.shape,self.dtype)
        fromgpu(out.ctypes.data,self.ptr.ctypes.data,self.nbyte,self.failed.ctypes.data)
        if self.failed[0]:
            print("error copying data from gpu.")
            return None
        else:
            return out
        
    def clear(self):
        gpufree(self.ptr.ctypes.data)
    def __del__(self):
        #print('deleting self.')
        self.clear()
    
    
def sgemm(A,B,transa=0,transb=0):
    if (transa==0) and (transb==0):
        n=A.shape[0]
        m=B.shape[1]
        assert(A.shape[1]==B.shape[0])
        k=A.shape[1]
        C=GPUvec(shape=[n,m],dtype=np.dtype('float32'))
        sgemm_gpu(0,0,n,m,k,A.ptr.ctypes.data,B.ptr.ctypes.data,C.ptr.ctypes.data)
        return C
    if (transa) and (transb==0):
        n=A.shape[1]
        m=B.shape[1]
        assert(A.shape[0]==B.shape[0])
        k=A.shape[0]
        C=GPUvec(shape=[n,m],dtype=np.dtype('float32'))
        sgemm_gpu(1,0,n,m,k,A.ptr.ctypes.data,B.ptr.ctypes.data,C.ptr.ctypes.data)
        return C

    if (transa==0) and (transb):
        n=A.shape[0]
        m=B.shape[0]
        assert(A.shape[1]==B.shape[1])
        k=A.shape[1]
        C=GPUvec(shape=[n,m],dtype=np.dtype('float32'))
        sgemm_gpu(0,1,n,m,k,A.ptr.ctypes.data,B.ptr.ctypes.data,C.ptr.ctypes.data)
        return C
    if (transa) and (transb):
        n=A.shape[1]
        m=B.shape[0]
        assert(A.shape[0]==B.shape[1])
        k=A.shape[0]
        C=GPUvec(shape=[n,m],dtype=np.dtype('float32'))
        sgemm_gpu(1,1,n,m,k,A.ptr.ctypes.data,B.ptr.ctypes.data,C.ptr.ctypes.data)
        return C
