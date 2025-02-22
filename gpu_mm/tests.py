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
        self.nypix_padded = (self.nypix_global + 63) & ~63   # round up to multiple of 64
        self.nxpix_padded = (self.nxpix_global + 63) & ~63   # round up to multiple of 64

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
        xpointing_cpu = gpu_mm.read_xpointing_npzfile(filename)

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
            nypix_global = 64*int(ymax//64) + 64,      # FIXME should be in npy file
            nxpix_global = 64*int(xmax//64) + 64,      # FIXME should be in npy file
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
        
            
    def test_map2tod(self):
        m = cp.random.normal(size=(3, self.nypix_padded, self.nxpix_padded), dtype=self.dtype)
        
        tod_ref = np.random.normal(size=self.tod_shape)
        tod_ref = np.asarray(tod_ref, dtype=self.dtype)
        lmap_ref = gpu_mm.LocalMap(self.lpix, cp.asnumpy(m))  # GPU -> CPU
        gpu_mm.map2tod(tod_ref, lmap_ref, self.xpointing_cpu, plan='reference', debug=True)
        tod_ref = cp.array(tod_ref)  # CPU -> GPU

        self._test_map2tod(m, tod_ref, plan=None, suffix='_unplanned')
        self._test_map2tod(m, tod_ref, plan=self.plan, suffix='')


    def _test_tod2map(self, tod, m0, m_ref, plan, suffix):
        """Helper method called by test_tod2map()."""
        
        local_map = gpu_mm.LocalMap(self.lpix, cp.copy(m0))
        gpu_mm.tod2map(local_map, tod, self.xpointing_gpu, plan, debug=True)   # note debug=True here
        cp.cuda.runtime.deviceSynchronize()

        epsilon = self._compare_arrays(local_map.arr, m_ref)
        print(f'    test_tod2map{suffix}: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold

        
    def test_tod2map(self):
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m0 = cp.random.normal(size=(3, self.nypix_padded, self.nxpix_padded), dtype=self.dtype)

        lmap_ref = gpu_mm.LocalMap(self.lpix, cp.asnumpy(m0))  # GPU -> CPU
        gpu_mm.tod2map(lmap_ref, cp.asnumpy(tod), self.xpointing_cpu, plan='reference')
        m_ref = cp.asarray(lmap_ref.arr)  # CPU -> GPU

        self._test_tod2map(tod, m0, m_ref, plan=None, suffix='_unplanned')
        self._test_tod2map(tod, m0, m_ref, plan=self.plan, suffix='')


    def test_dynamic_map(self):
        """Heler method called by test_tod2map()."""

        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        mref = np.zeros(shape=(3, self.nypix_padded, self.nxpix_padded), dtype=self.dtype)
        lmap_ref = gpu_mm.LocalMap(self.lpix, mref)
        gpu_mm.tod2map(lmap_ref, cp.asnumpy(tod), self.xpointing_cpu, plan='reference')
        
        # Reminder: the TOD can either be 1-d (time,) or 2-d (detector, time).
        nt = tod.shape[-1]

        # Let's divide the TOD into chunks along its time axis.
        nt_chunks = [ ]
        nt_remaining = nt
        
        while sum(nt_chunks) < nt:
            n = 32 * np.min(np.random.randint(1, nt//16 + 2, size=6))
            n = int(min(n, nt_remaining))
            nt_chunks.append(n)
            nt_remaining -= n
        
        print(f'    test_dynamic_map: {nt} -> {nt_chunks}')
    
        # Loop over chunks (FIXME test cell_mask).
        
        # Note initial_ncells=1 here, in order to exercise reallocation mechanism.
        dmap = gpu_mm.DynamicMap(self.nypix_global, self.nxpix_global, self.dtype, cell_mask=None, periodic_xcoord=self.periodic_xcoord, initial_ncells=1)
        nt_cumul = 0
        
        for nt_chunk in nt_chunks:
            tod_chunk = cp.copy(tod[..., nt_cumul:(nt_cumul+nt_chunk)])
            xpointing_chunk = cp.copy(self.xpointing_gpu[..., nt_cumul:(nt_cumul+nt_chunk)])
            
            preplan_chunk = gpu_mm.PointingPrePlan(xpointing_chunk, self.nypix_global, self.nxpix_global,
                                                   periodic_xcoord=self.periodic_xcoord, debug=self.debug_plan)
            
            plan_chunk = gpu_mm.PointingPlan(preplan_chunk, xpointing_chunk, debug=self.debug_plan)
            
            gpu_mm.tod2map(dmap, tod_chunk, xpointing_chunk, plan_chunk, debug=True)
            nt_cumul += nt_chunk

        # Randomly permute the cells (tests DynamicMap.permute())
        dmap.randomly_permute()
        
        # DynamicMap -> LocalMap -> (padded global map).
        lmap = dmap.finalize()
        mdyn = np.zeros((3, self.nypix_padded, self.nxpix_padded), dtype=self.dtype)
        mdyn[:, :self.nypix_global, :self.nxpix_global] = lmap.to_global()

        # Compare 'mref' and 'mdyn'.
        num = np.sum(np.abs(mref-mdyn))
        den = np.sum(np.abs(mref)) + np.sum(np.abs(mdyn))
        assert den > 0
        epsilon = float(num/den)   # convert zero-dimensional array -> scalar
        print(f'    test_dynamic_map: {epsilon=}')
        assert epsilon < 1.0e-6   # FIXME dtype=dependent threshold
        
        
    def test_all(self):
        print(f'    {self.plan}')
        self.test_pointing_preplan()
        self.test_pointing_plan()
        self.test_plan_iterator()
        self.test_map2tod()
        self.test_tod2map()
        self.test_dynamic_map()

        
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
        

    def time_map2tod(self):
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m = cp.zeros((3, self.nypix_global, self.nxpix_global), dtype=self.dtype)
        local_map = gpu_mm.LocalMap(self.lpix, m)

        self._time_map2tod(tod, local_map, plan=None, label='unplanned_map2tod')
        self._time_map2tod(tod, local_map, plan=self.plan, label='map2tod')


    def _time_tod2map(self, local_map, tod, plan, label):
        print()
        
        for _ in range(10):
            t0 = time.time()
            gpu_mm.tod2map(local_map, tod, self.xpointing_gpu, plan, debug=False)
            cp.cuda.runtime.deviceSynchronize()
            dt = time.time() - t0
            print(f'    {label}: {1000*dt} ms')

        
    def time_tod2map(self):
        # Note: we don't use a zeroed timestream, since tod2map() contains an optimization
        # which may give artificially fast timings when run on an all-zeroes timestream.
        
        tod = cp.random.normal(size=self.tod_shape, dtype=self.dtype)
        m = cp.zeros((3, self.nypix_global, self.nxpix_global), dtype=self.dtype)
        local_map = gpu_mm.LocalMap(self.lpix, m)

        self._time_tod2map(local_map, tod, plan=None, label='unplanned_tod2map')
        self._time_tod2map(local_map, tod, plan=self.plan, label='tod2map')

        
    def time_all(self):
        self.time_pointing_preplan()
        self.time_pointing_plan()
        self.time_map2tod()
        self.time_tod2map()


####################################################################################################


def make_random_npix(ncells_max=1024):
    """
    Usage:
       nypix_global, nycells = make_random_npix()
       nxpix_global, nxcells = make_random_npix()
    """

    # Randomly choose ncells.
    if np.random.uniform() < 0.05:
        ncells = 1
    elif np.random.uniform() < 0.1:
        ncells = ncells_max
    else:
        ncells = np.random.randint(1, ncells_max+1)

    # Randomly choose npix_global.
    if np.random.uniform() < 0.05:
        npix_global = 64*ncells - 63
    elif np.random.uniform() < 0.1:
        npix_global = 64*ncells
    elif np.random.uniform() < 0.4:
        npix_global = np.random.randint(64*ncells-63, 64*ncells+1)
    elif np.random.uniform() < 0.05:
        npix_global = 64*ncells_max
    else:
        npix_global = np.random.randint(64*ncells-63, 64*ncells_max+1)

    return npix_global, ncells


def make_random_plan_mt(nypix_global, nxpix_global, nmt=None):
    """Returns a 'plan_mt' array with no associated xpointing. 

    If 'nmt' is None, then nmt will be randomly generated.
    Currently only used by test_expand_dynamic_map()."""
    
    nycells = (nypix_global + 63) // 64
    nxcells = (nxpix_global + 63) // 64
    ncells_tot = nycells * nxcells
    ncells_hit_max = ncells_tot if (nmt is None) else min(ncells_tot,nmt)

    # ncells_hit
    if np.random.uniform() < 0.05:
        ncells_hit = 1
    elif np.random.uniform() < 0.05:
        ncells_hit = ncells_hit_max
    else:
        ncells_hit = np.random.randint(1, ncells_hit_max+1)

    # nmt
    if nmt is None:
        if np.random.uniform() < 0.1:
            nmt = ncells_hit
        else:
            nmt = np.random.randint(ncells_hit, 1024*1024)
            
    # print(f'{ncells_hit=}')
    # print(f'{nmt=}')

    # icell = 1-d array of length 'ncells_hit', containing 20-bit uints.
    iycell, ixcell = np.mgrid[:nycells, :nxcells]
    icell = (iycell << 10) + ixcell
    icell = np.random.permutation(icell.flatten())[:ncells_hit]

    # Reminder: mt bit layout is
    #   Low 10 bits = Global xcell index
    #   Next 10 bits = Global ycell index
    #   Next 26 bits = Primary TOD cache line index
    #   Next bit = mflag (does cache line overlap multiple map cells?)
    #   Next bit = zflag (mflag && first appearance of cache line)
    #   Total: 48 bits
    
    plan_mt = np.zeros(nmt, dtype=np.uint)
    plan_mt[:ncells_hit] = icell
    plan_mt[ncells_hit:] = icell[np.random.randint(0, ncells_hit, size=nmt-ncells_hit)]
    plan_mt = np.sort(plan_mt)

    # Higher bits just junk for now
    plan_mt |= np.random.randint(0, 1<<28, size=nmt, dtype=np.uint) << 20

    return plan_mt


def make_random_cell_offsets(nycells, nxcells):
    """Returns (cell_offsets_cpu, ncells_curr). Currently only used by test_expand_dynamic_map()."""
    
    # Randomly initialize 'cell_offsets' to either (-1) or (-2)
    p0 = np.random.uniform(-0.2, 1.2)
    x = np.random.uniform(size=(nycells,nxcells))
    cell_offsets = np.where(x < p0, -1, -2)

    # Randomly generated an unordered set of current cells.
    #  ncells_curr = integer between 0 and (nycells*nxcells), inclusive
    #  iycells = 1-d integer-valued array of length ncells_curr
    #  ixcells = 1-d integer-valued array of length ncells_curr
    
    p1 = np.random.uniform(-0.2, 1.2)
    x = np.random.uniform(size=(nycells,nxcells))
    mask = np.where(x < p1)
    iycells, ixcells = np.mgrid[:nycells,:nxcells]
    iycells = iycells[mask]
    ixcells = ixcells[mask]
    ncells_curr = len(iycells)    # also equal to len(ixcells)

    # Assign an ordering to the current cells
    cell_perm = np.random.permutation(ncells_curr)
    cell_offsets[iycells,ixcells] = cell_perm * (3*64*64)
    
    return cell_offsets, ncells_curr


def make_test_map(pixelization, dtype):
    assert isinstance(pixelization, gpu_mm.LocalPixelization)
    assert pixelization.polstride == 64*64
    assert pixelization.ystride == 64

    ncells = pixelization.npix // (64*64)
    n = 3*64*64
        
    y, x = np.where(pixelization.cell_offsets_cpu >= 0)   # 1-d arrays of length (ncells)
    cell_off = pixelization.cell_offsets_cpu[y,x]         # 1-d array of length (ncells)
    cell_ix = cell_off // n
    assert y.shape == x.shape == cell_ix.shape == (ncells,)
    assert np.all(cell_off == cell_ix*n)

    m0 = np.zeros((ncells,n), dtype=int)
    m0 += np.reshape(np.arange(n), (1,-1))
    m0 += np.reshape(137*y, (-1,1))
    m0 += np.reshape(2323*x, (-1,1))
    
    m1 = np.zeros((ncells,n), dtype=dtype)
    m1[cell_ix,:] = m0[:,:]   # permutes cells and converts dtype
    
    return gpu_mm.LocalMap(pixelization, m1.reshape(ncells*n))


def make_random_map(pixelization, dtype):
    """Helper for test()."""

    assert isinstance(pixelization, gpu_mm.LocalPixelization)
    m = np.random.uniform(-1.0, 1.0, size=3*pixelization.npix)
    return gpu_mm.LocalMap(pixelization, np.asarray(m,dtype))


####################################################################################################


def test_one_expand_dynamic_map():
    
    ### Part 1: make random test instance (plan_nmt_cpu, cell_offsets_cpu, global_ncells_cpu) ###
    
    nypix_global, nycells = make_random_npix()
    nxpix_global, nxcells = make_random_npix()
    
    plan_mt_cpu = make_random_plan_mt(nypix_global, nxpix_global)
    cell_offsets_cpu, global_ncells_cpu = make_random_cell_offsets(nycells, nxcells)

    print(f'test_expand_dynamic_map: npix_global={(nypix_global,nxpix_global)}, ncells={(nycells,nxcells)}, nmt={len(plan_mt_cpu)}, ncells_curr={global_ncells_cpu}')

    ### Part 2: run GPU kernel ###

    plan_mt_gpu = cp.array(plan_mt_cpu)
    cell_offsets_gpu = cp.array(cell_offsets_cpu)
    global_ncells_gpu = cp.full((1,), global_ncells_cpu, dtype=cp.uint32)
    
    new_ncells_gpu = gpu_mm_pybind11.expand_dynamic_map(global_ncells_gpu, cell_offsets_gpu, plan_mt_gpu)
    cell_offsets_gpu = cp.asnumpy(cell_offsets_gpu)   # GPU -> CPU

    ### Part 3: figure out which cell_indices should have been updated ###
    
    # Output of this part is a pair of 1-d arrays (ixcells, iycells).
    # The two arrays have the same length (equal to the number of new cells)
    #
    # Reminder: mt bit layout is
    #   Low 10 bits = Global xcell index
    #   Next 10 bits = Global ycell index
    #     ...
    
    icells = plan_mt_cpu & ((1 << 20) - 1)
    icells = np.unique(icells)
    ixcells = icells & ((1<<10) - 1)
    iycells = icells >> 10

    mask = np.logical_and(ixcells < nxcells, iycells < nycells)
    ixcells = ixcells[mask]
    iycells = iycells[mask]
    
    mask = (cell_offsets_cpu[iycells,ixcells] == -1)
    ixcells = ixcells[mask]
    iycells = iycells[mask]

    ### Part 4: check outputs of GPU kernel ###

    nnew = len(ixcells)
    # print(f'{new_ncells_gpu=} (expected {global_ncells_cpu} + {nnew} = {global_ncells_cpu+nnew})')
    assert(new_ncells_gpu == global_ncells_cpu + nnew)
    assert(cp.asarray(global_ncells_gpu)[0] == global_ncells_cpu + nnew)

    new_offsets = np.sort(cell_offsets_gpu[iycells,ixcells])  # note np.sort() here
    expected_offsets = (global_ncells_cpu + np.arange(nnew)) * (3*64*64)
    assert np.all(new_offsets == expected_offsets)
                
    mask_2d = (cell_offsets_gpu == cell_offsets_cpu)
    mask_2d[iycells,ixcells] = True
    assert np.all(mask_2d)


def test_expand_dynamic_map():
    for _ in range(100):
        test_one_expand_dynamic_map()
