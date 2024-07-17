import os
import time
import functools
import cupy as cp
import numpy as np

from . import gpu_mm
from . import gpu_mm_pybind11   # make_random_plan_mt, test_plan_iterator


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
    def __init__(self, xpointing_cpu, xpointing_gpu, nypix_global, nxpix_global, name, debug_plan=False):
        """Note: the xpointing arrays can be either shape (3,nsamp) or shape (3,ndet,nsamp)."""

        self.xpointing_cpu = xpointing_cpu
        self.xpointing_gpu = xpointing_gpu
        self.dtype = xpointing_gpu.dtype
        self.tod_shape = xpointing_gpu.shape[1:]
        self.nypix_global = nypix_global
        self.nxpix_global = nxpix_global
        self.periodic_xcoord = False    # FIXME
        self.name = name
        self.debug_plan = debug_plan

        # FIXME temporary convenience
        self.lpix = gpu_mm.LocalPixelization.make_rectangle(nypix_global, nxpix_global, self.periodic_xcoord)

    @classmethod
    def from_toy_pointing(cls, ndet, nt, nypix_global, nxpix_global, scan_speed, total_drift, debug_plan=False):
        tp = gpu_mm.ToyPointing(ndet, nt, nypix_global, nxpix_global, scan_speed, total_drift)
        return PointingInstance(tp.xpointing_cpu, tp.xpointing_gpu, tp.nypix_global, tp.nxpix_global, str(tp), debug_plan)        

    
    @classmethod
    def make_random(cls, nsamp_max, debug_plan=False):
        tp = gpu_mm.ToyPointing.make_random(nsamp_max, noisy=False)
        return PointingInstance(tp.xpointing_cpu, tp.xpointing_gpu, tp.nypix_global, tp.nxpix_global, str(tp), debug_plan)

    
    @classmethod
    def from_file(cls, filename, debug_plan=False):
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
            nypix_global = 64*int(ymax//64) + 64,      # FIXME should be in npy fileshould be in npy file
            nxpix_global = 64*int(xmax//64) + 64,      # FIXME 
            name = filename,
            debug_plan = debug_plan
        )

    
    @classmethod
    def generate_test_instances(cls):
        for _ in range(10):
            yield cls.make_random(1024*1024, debug_plan=True)
        for _ in range(10):
            yield cls.make_random(16*1024*1024, debug_plan=True)
        for _ in range(3):
            yield cls.make_random(256*1024*1024, debug_plan=True)
        for t in cls.generate_act_instances(debug_plan=True):
            yield t

            
    @classmethod
    def generate_timing_instances(cls):
        yield cls.from_toy_pointing(
            ndet = None,
            nt = 256*1024*1024,
            nypix_global = 8*1024,
            nxpix_global = 32*1024,
            scan_speed = 0.5,    # pixels per TOD sample
            total_drift = 1024   # x-pixels
        )
        
        for t in cls.generate_act_instances():
            yield t

    
    @classmethod
    def generate_act_instances(cls, debug_plan=False):
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
                yield cls.from_file(os.path.join(d,f), debug_plan)

        if not flag:
            print(f"No xpointing files found in directory {d}")
            

    @functools.cached_property
    def preplan(self):
        return gpu_mm.PointingPrePlan(self.xpointing_gpu, self.nypix_global, self.nxpix_global,
                                      periodic_xcoord=self.periodic_xcoord, debug=self.debug_plan)

    @functools.cached_property
    def plan(self):
        return gpu_mm.PointingPlan(self.preplan, self.xpointing_gpu, debug=self.debug_plan)

    @functools.cached_property
    def plan_tester(self):
        return gpu_mm.PointingPlanTester(self.preplan, self.xpointing_gpu)

    @functools.cached_property
    def old_plan(self):
        print('    Making OldPointingPlan (slow, done on CPU)')
        return  gpu_mm.OldPointingPlan(self.xpointing_cpu, self.nypix_global, self.nxpix_global)
    
    
    def _compare_arrays(self, arr1, arr2):
        num = cp.sum(cp.abs(arr1-arr2))
        den = cp.sum(cp.abs(arr1)) + cp.sum(cp.abs(arr2))
        assert den > 0
        return float(num/den)   # convert zero-dimensional array -> scalar

    
    def test_pointing_preplan(self):
        nmt_cumsum_fast = self.preplan.get_nmt_cumsum()
        nmt_cumsum_slow = self.plan_tester.nmt_cumsum
        
        assert nmt_cumsum_fast.shape == nmt_cumsum_slow.shape
        assert np.all(nmt_cumsum_fast == nmt_cumsum_slow)
        print('    test_pointing_preplan: pass')


    def test_pointing_plan(self):
        mt_fast = self.plan.get_plan_mt()
        mt_slow = self.plan_tester.sorted_mt
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


    def _test_map2tod(self, m, tod_ref, plan, suffix):
        """Helper method called by test_map2tod()."""
        
        local_map = gpu_mm.LocalMap(self.lpix, m)
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        gpu_mm.map2tod(tod, local_map, self.xpointing_gpu, plan, debug=True)  # note debug=True here
        cp.cuda.runtime.deviceSynchronize()
        
        epsilon = self._compare_arrays(tod_ref, tod)
        print(f'    test_map2tod{suffix}: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold
        
            
    def test_map2tod(self, test_old=False):
        m = cp.random.normal(size=(3, self.nypix_global, self.nxpix_global), dtype=self.dtype)
        
        tod_ref = np.random.normal(size=self.tod_shape)
        tod_ref = np.asarray(tod_ref, dtype=self.dtype)
        lmap_ref = gpu_mm.LocalMap(self.lpix, cp.asnumpy(m))  # GPU -> CPU
        gpu_mm.map2tod(tod_ref, lmap_ref, self.xpointing_cpu, plan='reference', debug=True)
        tod_ref = cp.array(tod_ref)  # CPU -> GPU

        self._test_map2tod(m, tod_ref, plan=None, suffix='_unplanned')
        self._test_map2tod(m, tod_ref, plan=self.plan, suffix='')

        if test_old:
            self._test_map2tod(m, tod_ref, plan='old', suffix='_old')


    def _test_tod2map(self, tod, m0, m_ref, plan, suffix):
        """Helper method called by test_tod2map()."""
        
        local_map = gpu_mm.LocalMap(self.lpix, cp.copy(m0))
        gpu_mm.tod2map(local_map, tod, self.xpointing_gpu, plan, debug=True)   # note debug=True here
        cp.cuda.runtime.deviceSynchronize()

        epsilon = self._compare_arrays(local_map.arr, m_ref)
        print(f'    test_tod2map{suffix}: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold
        
        
    def test_tod2map(self, test_old=False):
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m0 = cp.random.normal(size=(3, self.nypix_global, self.nxpix_global), dtype=self.dtype)

        lmap_ref = gpu_mm.LocalMap(self.lpix, cp.asnumpy(m0))  # GPU -> GPU
        gpu_mm.tod2map(lmap_ref, cp.asnumpy(tod), self.xpointing_cpu, plan='reference')
        m_ref = cp.asarray(lmap_ref.arr)  # CPU -> GPU

        self._test_tod2map(tod, m0, m_ref, plan=None, suffix='_unplanned')
        self._test_tod2map(tod, m0, m_ref, plan=self.plan, suffix='')

        if test_old:
            self._test_tod2map(tod, m0, m_ref, plan=self.old_plan, suffix='_old')

        
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
            pp = gpu_mm.PointingPrePlan(self.xpointing_gpu, self.nypix_global, self.nxpix_global)
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


    def _time_map2tod(self, tod, local_map, plan, label):
        print()
        
        for _ in range(10):
            t0 = time.time()
            gpu_mm.map2tod(tod, local_map, self.xpointing_gpu, plan, debug=False)
            cp.cuda.runtime.deviceSynchronize()
            dt = time.time()-t0
            print(f'    {label}: {1000*dt} ms')
        

    def time_map2tod(self, time_old=False):
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m = cp.zeros((3, self.nypix_global, self.nxpix_global), dtype=self.dtype)
        local_map = gpu_mm.LocalMap(self.lpix, m)

        self._time_map2tod(tod, local_map, plan=None, label='unplanned_map2tod')
        self._time_map2tod(tod, local_map, plan=self.plan, label='map2tod')

        if time_old:
            self._time_map2tod(tod, local_map, plan='old', label='old_map2tod (does not use a LocalPixelization)')


    def _time_tod2map(self, local_map, tod, plan, label):
        print()
        
        for _ in range(10):
            t0 = time.time()
            gpu_mm.tod2map(local_map, tod, self.xpointing_gpu, plan, debug=False)
            cp.cuda.runtime.deviceSynchronize()
            dt = time.time() - t0
            print(f'    {label}: {1000*dt} ms')

        
    def time_tod2map(self, time_old=False):
        # Note: we don't use a zeroed timestream, since tod2map() contains an optimization
        # which may give artificially fast timings when run on an all-zeroes timestream.
        
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m = cp.zeros((3, self.nypix_global, self.nxpix_global), dtype=self.dtype)
        local_map = gpu_mm.LocalMap(self.lpix, m)

        self._time_tod2map(local_map, tod, plan=None, label='unplanned_tod2map')
        self._time_tod2map(local_map, tod, plan=self.plan, label='tod2map')

        if time_old:
            self._time_tod2map(local_map, tod, plan=self.old_plan, label='old_tod2map (uses plan precomputed on cpu)')

    def time_all(self, ):
        time_old = True    # FIXME define command-line flag
        self.time_pointing_preplan()
        self.time_pointing_plan()
        self.time_map2tod(time_old=time_old)
        self.time_tod2map(time_old=time_old)

