import os
import time
import gpu_mm
import functools
import cupy as cp
import numpy as np


def is_sorted(arr):
    assert arr.ndim == 1
    return np.all(arr[:-1] <= arr[1:])   # duplicates are allowed


class PointingInstance:
    def __init__(self, xpointing_gpu, nypix, nxpix, name):
        self.xpointing_gpu = xpointing_gpu
        self.nypix = nypix
        self.nxpix = nxpix
        self.name = name


    @classmethod
    def from_toy_pointing(cls, nsamp, nypix, nxpix, scan_speed, total_drift):
        tp = gpu_mm.gpu_mm.ToyPointing(nsamp, nypix, nxpix, scan_speed, total_drift)
        return PointingInstance(tp.xpointing_gpu, tp.nypix, tp.nxpix, str(tp))        

    
    @classmethod
    def make_random(cls, nsamp_max):
        tp = gpu_mm.gpu_mm.ToyPointing.make_random(nsamp_max, noisy=False)
        return PointingInstance(tp.xpointing_gpu, tp.nypix, tp.nxpix, str(tp))        

    
    @classmethod
    def from_file(cls, filename):
        print(f'Reading xpointing file {filename}')
        
        f = np.load(filename)
        xp = f['xpointing']
        assert (xp.ndim == 3) and (xp.shape[0] == 3)   # ({x,y,a}, ndet, ntod)

        # FIXME this swap is horrible
        xp = np.array([ xp[1], xp[0], xp[2] ])

        ymin = np.min(xp[0,:])
        ymax = np.max(xp[0,:])
        xmin = np.min(xp[1,:])
        xmax = np.max(xp[1,:])
        print(f'{filename}: {xp.shape=}, ymin={float(ymin)} ymax={float(ymax)} xmin={float(xmin)} xmax={float(xmax)}')

        # Flatten (ndet, ntod) indices and round up to multiple of 32
        ns0 = xp.shape[1] * xp.shape[2]    # before padding
        nsamp = 32 * ((ns0 + 31) // 32)    # after padding
        xp2 = np.zeros((3, nsamp), dtype=xp.dtype)
        xp2[:,:ns0] = xp.reshape((3,ns0))
        xp2[:,ns0:] = xp[:,-1,-1].reshape((3,1))
        
        assert ymin >= 0
        assert xmin >= 0
        
        return PointingInstance(
            xpointing_gpu = cp.asarray(xp2),  # copy CPU -> CPU
            nypix = 64*(ymax//64) + 64,       # FIXME should be in npy file
            nxpix = 128*(xmax//128) + 128,    # FIXME can be 64 after removing periodicity
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
            nsamp = 256*1024*1024,
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
        return gpu_mm.gpu_mm.PointingPrePlan(self.xpointing_gpu, self.nypix, self.nxpix)

    @functools.cached_property
    def plan(self):
        return gpu_mm.gpu_mm.PointingPlan(self.preplan, self.xpointing_gpu)

    @functools.cached_property
    def reference_plan(self):
        return gpu_mm.gpu_mm.ReferencePointingPlan(self.preplan, self.xpointing_gpu)

    
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
        #   High 18 bits = Secondary TOD cache line index, 1-based (relative to start of block)

        # Lowest 20 bits of 'mt_fast' array should be sorted.
        assert is_sorted(mt_fast & ((1 << 20) - 1))

        # Lowest 46 bits of 'mt_fast' array should agree with 'mt_slow'.
        mt_sorted = np.sort(mt_fast & ((1 << 46) - 1))
        assert np.all(mt_sorted == mt_slow)
        print('    test_pointing_preplan: pass')

        
    def time_pointing_preplan(self):
        for _ in range(10):
            t0 = time.time()
            pp = gpu_mm.gpu_mm.PointingPrePlan(self.xpointing_gpu, self.nypix, self.nxpix)
            print(f'    time_pointing_preplan: {time.time()-t0} seconds')
            del pp

            
    def time_pointing_plan(self):
        pp = self.preplan
        buf = cp.zeros(pp.plan_nbytes, dtype=np.uint8)
        tmp_buf = cp.zeros(pp.plan_constructor_tmp_nbytes, dtype=np.uint8)
        
        for _ in range(10):
            t0 = time.time()
            p = gpu_mm.gpu_mm.PointingPlan(pp, self.xpointing_gpu, buf, tmp_buf)
            print(f'    time_pointing_plan: {time.time()-t0} seconds')
            del p
