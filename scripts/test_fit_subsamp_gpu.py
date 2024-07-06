import time
import numpy as np
import ctypes
import os

# from matplotlib import pyplot as plt
# plt.ion()

import gpu_mm
mylib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(gpu_mm.__file__), 'libgpu_mm.so'))

deval_fit=mylib.eval_fit
deval_fit.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p)

def eval_fit(fitp,ra_bore,dec_bore):
    ndet=fitp.shape[1]
    n=len(ra_bore)
    npar=fitp.shape[0]
    #A=np.zeros([npar,n],dtype='float32')-99
    out=np.zeros([ndet,n],dtype='float32')
    fitp=np.asarray(fitp,dtype='float32')
    ra_bore=np.asarray(ra_bore,dtype='float32')
    dec_bore=np.asarray(dec_bore,dtype='float32')
    deval_fit(out.ctypes.data,fitp.ctypes.data,n,ndet,ra_bore.ctypes.data,dec_bore.ctypes.data)
    return out

try:
    print(list(stuff.keys()))
except:
    rt='/home/sigurdkn/gpu_mapmaker/tods_full/'
    stuff=np.load(rt+'tod_1507611287.1507621748.ar5_f090.npz')
    point=stuff['pointing']
    ra=point[0,:,:]
    dec=point[1,:,:]
    th=point[2,:,:]
    ra[ra>np.pi]=ra[ra>np.pi]-2*np.pi


ra_bore=np.mean(ra,axis=0)
dec_bore=np.mean(dec,axis=0)
n=len(ra_bore)
t=np.linspace(-1,1,n)

ra_mat=np.vstack([ra_bore,ra_bore**2,ra_bore**3,ra_bore**4])
nra=ra_mat.shape[0]
dec_mat=np.vstack([dec_bore,dec_bore**2,dec_bore**3])
ndec=dec_mat.shape[0]
nt=2
mat=np.zeros([nt+4+nra+ndec,n])
for i in range(nt+1):
    mat[i,:]=t**i
mat[nt+1:nt+ndec+1,:]=dec_mat
mat[nt+ndec+1:nt+nra+ndec+1,:]=ra_mat
mat[-3,:]=ra_bore*t
mat[-2,:]=dec_bore*t
mat[-1,:]=dec_bore*ra_bore

mm=mat[:,::200]
rr=dec[:,::200]

t1=time.time()

lhs=mm@mm.T
rhs=mm@rr.T
fitp=np.linalg.inv(lhs)@rhs
t2=time.time()
print('elapsed time was ',t2-t1)
pred=(mat.T@fitp).T
print(np.std(pred-dec)*180/np.pi*3600)

A=eval_fit(fitp,ra_bore,dec_bore)
print('rms err is ',np.std(A-dec)*180/np.pi*3600)
