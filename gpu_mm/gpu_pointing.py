import numpy as np
import ctypes
import os

from . import gpu_utils
from .gpu_utils import GPUvec

mylib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "lib", "libgpu_mm.so"))

fillA=mylib.fillA_host
fillA.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p)


class gpuPoint:
    def __init__(self,ra_bore,dec_bore,ra_pars,dec_pars):
        self.ra_bore=GPUvec(np.asarray(ra_bore,dtype='float32'))
        self.dec_bore=GPUvec(np.asarray(dec_bore,dtype='float32'))
        self.ra_pars=GPUvec(np.asarray(ra_pars,dtype='float32'))
        self.dec_pars=GPUvec(np.asarray(dec_pars,dtype='float32'))
        self.ndet=ra_pars.shape[1]
        self.n=len(ra_bore)
        self.npar=13
        self.ra=None
        self.dec=None
    def eval_fit(self):
        A=GPUvec(shape=[self.npar,self.n],dtype=np.dtype('float32'))
        fillA(self.ra_bore.ptr.ctypes.data,self.dec_bore.ptr.ctypes.data,self.n,A.ptr.ctypes.data)
        dRA=gpu_utils.sgemm(self.ra_pars,A,1,0)
        ddec=gpu_utils.sgemm(self.dec_pars,A,1,0)
        return dRA,ddec
        #return A
    
