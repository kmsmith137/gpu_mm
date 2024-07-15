import os
import time
import functools
import cupy as cp
import numpy as np

from . import gpu_mm
from . import gpu_mm_pybind11

####################################################################################################

from .gpu_mm_pybind11 import reference_map2tod, reference_tod2map

class OldPointingPlan(gpu_mm_pybind11.OldPointingPlan):
    def __init__(self, xpointing, ndec, nra, verbose=True):
        gpu_mm_pybind11.OldPointingPlan.__init__(self, xpointing, ndec, nra, verbose)
        self.plan_cltod_list = cp.asarray(self._plan_cltod_list)  # CPU -> GPU
        self.plan_quadruples = cp.asarray(self._plan_quadruples)  # CPU -> GPU

    def map2tod(self, tod, m, xpointing):
        gpu_mm_pybind11.old_map2tod(tod, m, xpointing)
            
    def tod2map(self, m, tod, xpointing):
        gpu_mm_pybind11.old_tod2map(m, tod, xpointing, self.plan_cltod_list, self.plan_quadruples)
    

####################################################################################################


def is_sorted(arr):
    assert arr.ndim == 1
    return np.all(arr[:-1] <= arr[1:])   # duplicates are allowed


def test_plan_iterator(niter=100):
    for _ in range(niter):
        min_nmt_per_cell = 1
        max_nmt_per_cell = 1000
        ncells = np.random.randint(100, 1000)
        nmt_per_block = np.random.randint(1, 1000)
        warps_per_threadblock = 4 * np.random.randint(1,5)
        plan_mt = gpu_mm_pybind11.make_random_plan_mt(ncells, min_nmt_per_cell=min_nmt_per_cell, max_nmt_per_cell=max_nmt_per_cell)
        print(f'test_plan_iterator({ncells=}, {min_nmt_per_cell=}, {max_nmt_per_cell=}, {nmt_per_block=}, {warps_per_threadblock=}')
        gpu_mm_pybind11.test_plan_iterator(plan_mt, nmt_per_block, warps_per_threadblock)

        
class PointingInstance:
    def __init__(self, xpointing_cpu, xpointing_gpu, nypix, nxpix, name):
        """Note: the xpointing arrays can be either shape (3,nsamp) or shape (3,ndet,nsamp)."""

        self.xpointing_cpu = xpointing_cpu
        self.xpointing_gpu = xpointing_gpu
        self.dtype = xpointing_gpu.dtype
        self.tod_shape = xpointing_gpu.shape[1:]
        self.nypix = nypix
        self.nxpix = nxpix
        self.name = name

        # FIXME temporary convenience
        self.lpix = gpu_mm.LocalPixelization.make_rectangle(nypix, nxpix)

    @classmethod
    def from_toy_pointing(cls, ndet, nt, nypix, nxpix, scan_speed, total_drift):
        tp = gpu_mm.ToyPointing(ndet, nt, nypix, nxpix, scan_speed, total_drift)
        return PointingInstance(tp.xpointing_cpu, tp.xpointing_gpu, tp.nypix, tp.nxpix, str(tp))        

    
    @classmethod
    def make_random(cls, nsamp_max):
        tp = gpu_mm.ToyPointing.make_random(nsamp_max, noisy=False)
        return PointingInstance(tp.xpointing_cpu, tp.xpointing_gpu, tp.nypix, tp.nxpix, str(tp))        

    
    @classmethod
    def from_file(cls, filename):
        print(f'Reading xpointing file {filename}')
        
        f = np.load(filename)
        xp = f['xpointing']
        assert (xp.ndim == 3) and (xp.shape[0] == 3)   # ({x,y,a}, ndet, ntod)
        ndet = xp.shape[1]
        ntu = xp.shape[2]            # unpadded
        ntp = 32 * ((ntu+31) // 32)  # padded to multiple of 32

        # FIXME should convert dtype here, to whatever has been compiled.
        xpointing_cpu = np.zeros((3,ndet,ntp), dtype=xp.dtype)
        xpointing_cpu[0,:,:ntu] = xp[1,:,:]    # FIXME (0,1) index swap here is horrible
        xpointing_cpu[1,:,:ntu] = xp[0,:,:]    # FIXME (0,1) index swap here is horrible
        xpointing_cpu[2,:,:ntu] = xp[2,:,:]
        xpointing_cpu[:,:,ntu:] = xpointing_cpu[:,:,ntu-1].reshape((3,-1,1))

        ymin = np.min(xpointing_cpu[0,:])
        ymax = np.max(xpointing_cpu[0,:])
        xmin = np.min(xpointing_cpu[1,:])
        xmax = np.max(xpointing_cpu[1,:])
        print(f'{filename}: {xpointing_cpu.shape=}, ymin={float(ymin)} ymax={float(ymax)} xmin={float(xmin)} xmax={float(xmax)}')
        
        assert ymin >= 0
        assert xmin >= 0
        
        return PointingInstance(
            xpointing_cpu = xpointing_cpu,
            xpointing_gpu = cp.asarray(xpointing_cpu),
            nypix = 64*int(ymax//64) + 64,      # FIXME should be in npy fileshould be in npy file
            nxpix = 64*int(xmax//64) + 64,      # FIXME 
            name = filename
        )

    
    @classmethod
    def generate_test_instances(cls):
        for _ in range(10):
            yield cls.make_random(1024*1024)
        for _ in range(10):
            yield cls.make_random(16*1024*1024)
        for _ in range(3):
            yield cls.make_random(256*1024*1024)
        for t in cls.generate_act_instances():
            yield t

            
    @classmethod
    def generate_timing_instances(cls):
        yield cls.from_toy_pointing(
            ndet = None,
            nt = 256*1024*1024,
            nypix = 8*1024,
            nxpix = 32*1024,
            scan_speed = 0.5,    # pixels per TOD sample
            total_drift = 1024   # x-pixels
        )
        
        for t in cls.generate_act_instances():
            yield t

    
    @classmethod
    def generate_act_instances(cls):
        if 'HOME' not in os.environ:
            print("Environment variable HOME not defined, can't look for ACT xpointing files")
            return

        d = os.path.join(os.environ['HOME'], 'xpointing')
        if not os.path.isdir(d):
            print(f"Directory {d} not found, ACT xpointing files will not be analyzed")

        flag = False
        for f in sorted(os.listdir(d)):
            if f.startswith('xpointing') and f.endswith('.npz'):
                flag = True
                yield cls.from_file(os.path.join(d,f))

        if not flag:
            print(f"No xpointing files found in directory {d}")
            

    @functools.cached_property
    def preplan(self):
        return gpu_mm.PointingPrePlan(self.xpointing_gpu, self.nypix, self.nxpix)

    @functools.cached_property
    def plan(self):
        return gpu_mm.PointingPlan(self.preplan, self.xpointing_gpu)

    @functools.cached_property
    def reference_plan(self):
        return gpu_mm.ReferencePointingPlan(self.preplan, self.xpointing_gpu)

    @functools.cached_property
    def old_plan(self):
        print('    Making OldPointingPlan (slow, done on CPU)')
        return  OldPointingPlan(self.xpointing_cpu, self.nypix, self.nxpix)
    
    
    def _compare_arrays(self, arr1, arr2):
        num = cp.sum(cp.abs(arr1-arr2))
        den = cp.sum(cp.abs(arr1)) + cp.sum(cp.abs(arr2))
        assert den > 0
        return float(num/den)   # convert zero-dimensional array -> scalar

    
    def test_pointing_preplan(self):
        nmt_cumsum_fast = self.preplan.get_nmt_cumsum()
        nmt_cumsum_slow = self.reference_plan.nmt_cumsum
        
        assert nmt_cumsum_fast.shape == nmt_cumsum_slow.shape
        assert np.all(nmt_cumsum_fast == nmt_cumsum_slow)
        print('    test_pointing_preplan: pass')


    def test_pointing_plan(self):
        mt_fast = self.plan.get_plan_mt()
        mt_slow = self.reference_plan.sorted_mt
        assert mt_slow.shape == mt_fast.shape

        # Reminder: mt bit layout is
        #   Low 10 bits = Global xcell index
        #   Next 10 bits = Global ycell index
        #   Next 26 bits = Primary TOD cache line index
        #   Next bit = zflag
        #   Next bit = aflag

        # Lowest 20 bits of 'mt_fast' array should be sorted.
        assert is_sorted(mt_fast & ((1 << 20) - 1))

        # Lowest 46 bits of 'mt_fast' array should agree with 'mt_slow' (after sorting).
        mt_sorted = np.sort(mt_fast & ((1 << 46) - 1))
        assert np.all(mt_sorted == mt_slow)
        print('    test_pointing_preplan: pass')


    def test_plan_iterator(self):
        plan_mt = self.plan.get_plan_mt()   # FIXME extra copy GPU -> CPU -> GPU
        nmt_per_threadblock = 32 * np.random.randint(1, 100)   # FIXME revisit
        warps_per_threadblock = 4 * np.random.randint(1,5)     # FIXME revisit
        gpu_mm_pybind11.test_plan_iterator(plan_mt, nmt_per_threadblock, warps_per_threadblock)
        print('    test_plan_iterator: pass')


    def test_map2tod(self, test_old=False):
        m = cp.random.normal(size=(3, self.nypix, self.nxpix), dtype=self.dtype)
        
        tod_ref = np.random.normal(size=self.tod_shape)
        tod_ref = np.asarray(tod_ref, dtype=self.dtype)
        reference_map2tod(tod_ref, cp.asnumpy(m), self.xpointing_cpu, self.lpix, allow_outlier_pixels=False)
        tod_ref = cp.array(tod_ref)  # CPU -> GPU

        tod_unplanned = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        gpu_mm_pybind11.unplanned_map2tod(tod_unplanned, m, self.xpointing_gpu, self.lpix)
        cp.cuda.runtime.deviceSynchronize()
        
        epsilon = self._compare_arrays(tod_ref, tod_unplanned)
        print(f'    test_map2tod_unplanned: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold
        del tod_unplanned

        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        self.plan.map2tod(tod, m, self.xpointing_gpu, self.lpix, debug=True)
        cp.cuda.runtime.deviceSynchronize()
        
        epsilon = self._compare_arrays(tod_ref, tod)
        print(f'    test_map2tod: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold
        del tod

        if not test_old:
            return

        tod_old = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        self.old_plan.map2tod(tod_old, m, self.xpointing_gpu)
        cp.cuda.runtime.deviceSynchronize()
        
        epsilon = self._compare_arrays(tod_ref, tod_old)
        print(f'    test_map2tod_old: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold

        
    def test_tod2map(self, test_old=False):
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m0 = cp.random.normal(size=(3, self.nypix, self.nxpix), dtype=self.dtype)

        m_ref = cp.asnumpy(m0)
        reference_tod2map(m_ref, cp.asnumpy(tod), self.xpointing_cpu, self.lpix, allow_outlier_pixels=False)
        m_ref = cp.asarray(m_ref)  # CPU -> GPU

        m_unplanned = cp.copy(m0)
        gpu_mm_pybind11.unplanned_tod2map(m_unplanned, tod, self.xpointing_gpu, self.lpix)
        cp.cuda.runtime.deviceSynchronize()

        epsilon = self._compare_arrays(m_ref, m_unplanned)
        print(f'    test_tod2map_unplanned: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold
        del m_unplanned

        m = cp.copy(m0)
        self.plan.tod2map(m, tod, self.xpointing_gpu, self.lpix, debug=True)
        cp.cuda.runtime.deviceSynchronize()

        epsilon = self._compare_arrays(m_ref, m)
        print(f'    test_tod2map: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold
        del m

        if not test_old:
            return

        m_old = cp.copy(m0)
        self.old_plan.tod2map(m_old, tod, self.xpointing_gpu)
        cp.cuda.runtime.deviceSynchronize()

        epsilon = self._compare_arrays(m_ref, m_old)
        print(f'    test_old_tod2map: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold
        del m_old

        
    def test_all(self):
        test_old = True   # FIXME define command-line flag
        print(f'    {self.plan}')
        self.test_pointing_preplan()
        self.test_pointing_plan()
        self.test_plan_iterator()
        self.test_map2tod(test_old=test_old)
        self.test_tod2map(test_old=test_old)

        
    def time_pointing_preplan(self):
        print()
        for _ in range(10):
            t0 = time.time()
            pp = gpu_mm.PointingPrePlan(self.xpointing_gpu, self.nypix, self.nxpix)
            print(f'    time_pointing_preplan: {1000*(time.time()-t0)} ms')
            del pp

            
    def time_pointing_plan(self):
        pp = self.preplan
        buf = cp.zeros(pp.plan_nbytes, dtype=np.uint8)
        tmp_buf = cp.zeros(pp.plan_constructor_tmp_nbytes, dtype=np.uint8)        
        print()
        
        for _ in range(10):
            t0 = time.time()
            p = gpu_mm.PointingPlan(pp, self.xpointing_gpu, buf, tmp_buf)
            print(f'    time_pointing_plan: {1000*(time.time()-t0)} ms')
            del p
    

    def time_map2tod(self, time_old=False):
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m = cp.zeros((3, self.nypix, self.nxpix), dtype=self.dtype)
        print()
        
        for _ in range(10):
            t0 = time.time()
            gpu_mm_pybind11.unplanned_map2tod(tod, m, self.xpointing_gpu, self.lpix)
            cp.cuda.runtime.deviceSynchronize()
            print(f'    time_unplanned_map2tod: {1000*(time.time()-t0)} ms')

        plan = self.plan        
        print()
        
        for _ in range(10):
            t0 = time.time()
            plan.map2tod(tod, m, self.xpointing_gpu, self.lpix, debug=False)
            cp.cuda.runtime.deviceSynchronize()
            print(f'    time_map2tod: {1000*(time.time()-t0)} ms')

        if not time_old:
            return
        
        plan = self.old_plan
        print()
        
        for _ in range(10):
            t0 = time.time()
            plan.map2tod(tod, m, self.xpointing_gpu)
            cp.cuda.runtime.deviceSynchronize()
            print(f'    time_old_map2tod: {1000*(time.time()-t0)} ms')


    def time_tod2map(self, time_old=False):
        # Note: we don't use a zeroed timestream, since tod2map() contains an optimization
        # which may give artificially fast timings when run on an all-zeroes timestream.
        
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m = cp.zeros((3, self.nypix, self.nxpix), dtype=self.dtype)
        print()
        
        for _ in range(10):
            t0 = time.time()
            gpu_mm_pybind11.unplanned_tod2map(m, tod, self.xpointing_gpu, self.lpix)
            cp.cuda.runtime.deviceSynchronize()
            print(f'    time_unplanned_tod2map: {1000*(time.time()-t0)} ms')

        plan = self.plan
        print()

        for _ in range(10):
            t0 = time.time()
            plan.tod2map(m, tod, self.xpointing_gpu, self.lpix, debug=False)
            cp.cuda.runtime.deviceSynchronize()
            print(f'    time_tod2map: {1000*(time.time()-t0)} ms')

        if not time_old:
            return

        plan = self.old_plan
        print()

        for _ in range(10):
            t0 = time.time()
            plan.tod2map(m, tod, self.xpointing_gpu)
            cp.cuda.runtime.deviceSynchronize()
            print(f'    time_old_tod2map (uses plan computed on cpu): {1000*(time.time()-t0)} ms')


    def time_all(self, ):
        time_old = True    # FIXME define command-line flag
        self.time_pointing_preplan()
        self.time_pointing_plan()
        self.time_map2tod(time_old=time_old)
        self.time_tod2map(time_old=time_old)

