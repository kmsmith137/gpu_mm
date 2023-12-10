import cupy as cp
import ctypes
mylib=ctypes.cdll.LoadLibrary("libpycufft.so")

# Sigurd changed the interface to match numpy/cupy better, but having the user
# pass in the output length instead of isodd. That means this file is currently
# broken will fix later. Also moved the implementation into libgpu_mm for now
# to make linking with the mapmaker easier. This will also be cleaned up
#
# Here's my new interface
#
#def rfft(dat, out=None, axis=-1):
#  # We apparently only support 2d input arrays
#  axis = axis % dat.ndim
#  if dat.ndim != 2:
#    raise ValueError("pycufft.rfft only supports 2d arrays")
#  if dat.dtype != np.float32:
#    raise ValueError("pycufft.rfft only supports single precision")
#  if out is None:
#    oshape = dat.shape[:axis] + (dat.shape[axis]//2+1,)+dat.shape[axis+1:]
#    out    = cp.empty(oshape, dtype=np.complex64)
#  elif out.dtype != np.complex64:
#    raise ValueError("pycufft.rfft only supports single precision")
#  cufft_r2c_gpu(out.data.ptr, dat.data.ptr, dat.shape[0], dat.shape[1], axis)
#  return out
#
#def irfft(dat, out=None, n=None, axis=-1):
#  # We apparently only support 2d input arrays
#  axis = axis % dat.ndim
#  if dat.ndim != 2:
#    raise ValueError("pycufft.rfft only supports 2d arrays")
#  if dat.dtype != np.complex64:
#    raise ValueError("pycufft.rfft only supports single precision")
#  if out is None:
#    if n is None: n = 2*(dat.shape[axis]-1)
#    oshape = dat.shape[:axis] + (n,)+dat.shape[axis+1:]
#    out    = cp.empty(oshape, dtype=np.float32)
#  elif out.dtype != np.float32:
#    raise ValueError("pycufft.rfft only supports single precision")
#  cufft_c2r_gpu(out.data.ptr, dat.data.ptr, out.shape[0], out.shape[1], axis)
#  return out

cufft_r2c_gpu=mylib.cufft_r2c_gpu
cufft_r2c_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int)

cufft_c2r_gpu=mylib.cufft_c2r_gpu
cufft_c2r_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)

def rfft(dat,out=None,axis=1):
    if not(dat.dtype=='float32'):
        print("warning - only float32 is supported in pycufft.rfft.  casting")
        x=cp.asarray(x,dtype='float32')
    
    n=dat.shape[0]
    m=dat.shape[1]
    if not(out is None):
        if not(out.dtype=='complex64'):
            print('warning - only complex64 is supported for rfft output in pycufft.rfft. allocating new storage')
            out=None
    if out is None:
        if axis==1:
            out=cp.empty([n,m//2+1],dtype='complex64')
        else:
            out=cp.empty([n//2+1,m],dtype='complex64')
    cufft_r2c_gpu(out.data.ptr,dat.data.ptr,n,m,axis)
    return out
def irfft(dat,out=None,axis=1,isodd=0):
    n=dat.shape[0]
    m=dat.shape[1]
    isodd=isodd%2
    if axis==0:
        if isodd:
            nn=2*n-1
        else:
            nn=2*(n-1)
        if out is None:
            out=cp.empty([nn,m],dtype='float32')
    else:
        if isodd:
            mm=2*m-1
        else:
            mm=2*(m-1)
        if out is None:
            out=cp.empty([n,mm],dtype='float32')
    
    cufft_c2r_gpu(out.data.ptr,dat.data.ptr,n,m,axis,isodd)
    return out
