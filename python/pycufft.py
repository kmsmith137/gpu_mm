import cupy as cp
import ctypes
import numpy as np
mylib=ctypes.cdll.LoadLibrary("libpycufft.so")

cufft_r2c_gpu=mylib.cufft_r2c_gpu
cufft_r2c_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int)

cufft_r2c_gpu_wplan=mylib.cufft_r2c_gpu_wplan
cufft_r2c_gpu_wplan.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p)

cufft_c2r_gpu=mylib.cufft_c2r_gpu
cufft_c2r_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int)

cufft_c2r_gpu_wplan=mylib.cufft_c2r_gpu_wplan
cufft_c2r_gpu_wplan.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)

get_plan_r2c_gpu=mylib.get_plan_r2c
get_plan_r2c_gpu.argtypes=(ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int)
get_plan_c2r_gpu=mylib.get_plan_c2r
get_plan_c2r_gpu.argtypes=(ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int)

get_plan_size_gpu=mylib.get_plan_size
get_plan_size_gpu.argtypes=(ctypes.c_void_p,ctypes.c_void_p)
destroy_plan_gpu=mylib.destroy_plan
destroy_plan_gpu.argtypes=(ctypes.c_void_p,)
set_plan_scratch_gpu=mylib.set_plan_scratch
set_plan_scratch_gpu=mylib.set_plan_scratch
set_plan_scratch_gpu.argtypes=(ctypes.c_int,ctypes.c_void_p)

def get_plan_size(plan):
    sz=np.zeros(1,dtype='uint64')
    get_plan_size_gpu(plan.ctypes.data,sz.ctypes.data)
    return sz[0]
def set_plan_scratch(plan,buf):
    #print('plan in python is ',plan[0],plan.dtype)
    set_plan_scratch_gpu(plan[0],buf.data.ptr)
def get_plan_r2c(n,m,axis=1,alloc=1):
    plan=np.empty(1,dtype='int32') #I checked, and sizeof(plan) is 4 bytes
    get_plan_r2c_gpu(n,m,axis,plan.ctypes.data,alloc)
    return plan

def get_plan_c2r(n,m,axis=1,alloc=1):
    plan=np.empty(1,dtype='int32')
    get_plan_c2r_gpu(n,m,axis,plan.ctypes.data,alloc)
    return plan

def destroy_plan(plan):
    destroy_plan_gpu(plan.ctypes.data)

def rfft(dat,out=None,axis=1,plan=None,plan_cache=None):
    if not(dat.dtype=='float32'):
        print("warning - only float32 is supported in pycufft.rfft.  casting")
        x=cp.asarray(x,dtype='float32')    
    n=dat.shape[0]
    m=dat.shape[1]
    if not(plan_cache is None):
        plan=plan_cache.get_plan(n,m,True)
    if not(out is None):
        if not(out.dtype=='complex64'):
            print('warning - only complex64 is supported for rfft output in pycufft.rfft. allocating new storage')
            out=None
    if out is None:
        if axis==1:
            out=cp.empty([n,m//2+1],dtype='complex64')
        else:
            out=cp.empty([n//2+1,m],dtype='complex64')
    if plan is None:
        cufft_r2c_gpu(out.data.ptr,dat.data.ptr,n,m,axis)
    else:
        cufft_r2c_gpu_wplan(out.data.ptr,dat.data.ptr,n,m,axis,plan.ctypes.data)
    return out
def irfft(dat,out=None,axis=1,isodd=0,plan=None,plan_cache=None):
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
    if not(plan_cache is None):
        plan=plan_cache.get_plan(n,mm,False)
    if plan is None:
        cufft_c2r_gpu(out.data.ptr,dat.data.ptr,n,m,axis,isodd)
    else:
        cufft_c2r_gpu_wplan(out.data.ptr,dat.data.ptr,plan.ctypes.data)
    return out

def get_tag(n,m,forward):
    if forward:
        tag='r2c_'
    else:
        tag='c2r_'
    tag=tag+repr(n)+'_'+repr(m)
    return tag
class MultiPlan:
    def __init__(self,sizes):
        """Make a class that creates a bunch of cufft plans that share a common buffer"""
        self.plans={}
        maxbytes=0
        for sz in sizes:
            plan_r2c=get_plan_r2c(sz[0],sz[1],alloc=0)
            #print('plan r2c is ',plan_r2c[0])
            nb=get_plan_size(plan_r2c)
            if nb>maxbytes:
                maxbytes=nb
            nm=get_tag(sz[0],sz[1],True)
            self.plans[nm]=plan_r2c
            plan_c2r=get_plan_c2r(sz[0],sz[1],alloc=0)
            #print('plan c2r is ',plan_c2r[0])
            nb=get_plan_size(plan_c2r)
            if nb>maxbytes:
                maxbytes=nb
            nm=get_tag(sz[0],sz[1],False)
            self.plans[nm]=plan_c2r
        self.scratch=cp.empty(maxbytes,dtype='uint8')
        print('setting scratches with size ',maxbytes)
        for key in self.plans.keys():
            plan=self.plans[key]
            set_plan_scratch(plan,self.scratch)
    def get_plan(self,n,m,forward):
        nm=get_tag(n,m,forward)
        try:
            return self.plans[nm]
        except:
            print('failed in finding plan ',nm,' in get_plan')
            
                              
