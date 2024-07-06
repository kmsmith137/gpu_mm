import numpy as np
import ctypes

raise RuntimeError("I think this script is bitrotted -- e.g. it expects cufft_c2r_host() to have 6 arguments (2 pointer + 4 int),"
                   + " and I didn't see a function with that signature in our cuda code. --KS")

mylib=ctypes.cdll.LoadLibrary("libtime_cufft.so")

#cufft_c2r=mylib.cufft_c2r_host
#cufft_c2r.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)

cufft_r2c=mylib.cufft_r2c_host
cufft_r2c.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int)

cufft_c2r=mylib.cufft_c2r_host
cufft_c2r.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)



def fft_c2r(datft):
    n=datft.shape[0]
    mm=datft.shape[1]
    m=(mm-1)*2
    out=np.zeros([n,m],dtype='float32')
    cufft_c2r(out.ctypes.data,datft.ctypes.data,n,mm,0,1)
    return out/m
    
def fft_r2c(dat):
    n=dat.shape[0]
    m=dat.shape[1]
    mm=m//2+1;

    out=np.zeros([n,mm],dtype='complex64')
    cufft_r2c(out.ctypes.data,dat.ctypes.data,n,m,1)

    return out

n=1000
m=2**18
x=np.random.randn(n,m)
x=np.asarray(x,dtype='float32')
xft=np.fft.rfft(x,axis=1)
xft2=fft_r2c(x)
xx=fft_c2r(xft2)
print('std of fft is ',np.std(xft-xft2)/np.std(xft))
print('std of return is ',np.std(xx-x)/np.std(x))
