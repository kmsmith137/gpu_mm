import os
import gpu_mm
import cupy as cp
import numpy as np


class TestPointing:
    def __init__(self, xpointing_gpu, nypix, nxpix, name):
        self.xpointing_gpu = xpointing_gpu
        self.nypix = nypix
        self.nxpix = nxpix
        self.name = name

        
    @staticmethod
    def make_random(nsamp_max):
        tp = gpu_mm.gpu_mm.ToyPointing.make_random(nsamp_max, noisy=False)
        return TestPointing(tp.xpointing_gpu, tp.nypix, tp.nxpix, str(tp))

    
    @staticmethod
    def from_file(filename):
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
        
        return TestPointing(
            xpointing_gpu = cp.asarray(xp2),  # copy CPU -> CPU
            nypix = 64*(ymax//64) + 64,       # FIXME should be in npy file
            nxpix = 128*(xmax//128) + 128,    # FIXME can be 64 after removing periodicity
            name = filename
        )

    
    @classmethod
    def generate(cls):
        for _ in range(10):
            yield cls.make_random(1024*1024)
        for _ in range(10):
            yield cls.make_random(16*1024*1024)
        for _ in range(3):
            yield cls.make_random(256*1024*1024)

        if 'HOME' not in os.environ:
            print("Environment variable HOME not defined, can't look for xpointing files")
            return

        d = os.path.join(os.environ['HOME'], 'xpointing')
        if not os.path.isdir(d):
            print(f"Directory {d} not found, xpointing files will not be analyzed")

        flag = False
        for f in sorted(os.listdir(d)):
            if f.startswith('xpointing') and f.endswith('.npz'):
                flag = True
                yield cls.from_file(os.path.join(d,f))

        if not flag:
            print(f"No xpointing files found in directory {d}")
            
        
def test_pointing_preplan():
    for tp in TestPointing.generate():
        print(f'test_pointing_preplan: start: {tp.name}')
        
        pp = gpu_mm.gpu_mm.PointingPrePlan(tp.xpointing_gpu, tp.nypix, tp.nxpix)
        qp = gpu_mm.gpu_mm.QuantizedPointing(tp.xpointing_gpu, tp.nypix, tp.nxpix)
        
        nmt_cumsum_fast = pp.get_nmt_cumsum()
        nmt_cumsum_slow = qp.compute_nmt_cumsum(pp._rk)

        assert nmt_cumsum_fast.shape == nmt_cumsum_slow.shape
        assert np.all(nmt_cumsum_fast == nmt_cumsum_slow)
        print('test_pointing_preplan: pass')
