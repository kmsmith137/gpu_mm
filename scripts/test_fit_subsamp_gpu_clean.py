from gpu_mm import gpu_pointing

import time
import numpy as np
import ctypes

# from matplotlib import pyplot as plt
# plt.ion()

def fill_mat(ra_bore,dec_bore,nt,nra,ndec,cross=True,subsamp=200):
    n=len(ra_bore)
    tvec=np.linspace(-1,1,n)
    if subsamp>1:
        ra_bore=ra_bore[::subsamp]
        dec_bore=dec_bore[::subsamp]
        tvec=tvec[::subsamp]
        n=len(ra_bore)
    npar=nra+ndec+nt+1
    if cross:
        npar=npar+3
    mat=np.zeros([npar,n])
    for i in range(nt+1):
        mat[i,:]=tvec**i
    icur=nt+1
    for i in range(ndec):
        mat[icur,:]=dec_bore**(i+1)
        icur=icur+1
    for i in range(nra):
        mat[icur,:]=ra_bore**(i+1)
        icur=icur+1
    if cross:
        mat[icur,:]=ra_bore*tvec
        icur=icur+1
        mat[icur,:]=dec_bore*tvec
        icur=icur+1
        mat[icur,:]=ra_bore*dec_bore
        icur=icur+1
    assert(icur==npar)
    return mat
        
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

#do the parameter fit on the CPU
nt=2
nra=4
ndec=3
subsamp=200

t1=time.time()
mm=fill_mat(ra_bore,dec_bore,nt,nra,ndec,subsamp=subsamp)
lhs=mm@mm.T
rhs=mm@ra[:,::subsamp].T
fitp_ra=np.linalg.inv(lhs)@rhs
rhs=mm@dec[:,::subsamp].T
fitp_dec=np.linalg.inv(lhs)@rhs
t2=time.time()
print('cpu time to do fit was ',t2-t1)

point=gpu_pointing.gpuPoint(ra_bore,dec_bore,fitp_ra,fitp_dec)
dRA,dDec=point.eval_fit()
print('RMS RA err is ',np.std(dRA.fromgpu()-ra)*180/np.pi*3600,' arcseconds.')
print('RMS Dec err is ',np.std(dDec.fromgpu()-dec)*180/np.pi*3600,' arcseconds.')
