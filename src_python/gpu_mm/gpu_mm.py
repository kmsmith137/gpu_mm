# This source file will eventually become a proper python module that 'pip' can build.
#
# Currently, it contains some hackery (e.g. hardcoded pathname ../lib/libgpu_mm.so).
# To use it, you'll need to build the C++ library ('make -j' from the toplevel gpu_mm dir),
# and then do 'import gpu_mm' from the gpu_mm/scripts directory.

import ctypes, os
import numpy as np
import cupy as cp

from . import gpu_mm_pybind11

# Currently we wrap either float32 or float64, determined at compile time!
# FIXME eventually, should wrap both.
mm_dtype = np.float64 if (gpu_mm_pybind11._get_tsize() == 8) else np.float32


def map2tod(tod, local_map, xpointing, plan, partial_pixelization=False, debug=False):
    assert isinstance(local_map, LocalMap)
    lpix = local_map.pixelization
    larr = local_map.arr

    if isinstance(plan, PointingPlan):
        gpu_mm_pybind11.planned_map2tod(tod, larr, xpointing, lpix, plan, partial_pixelization, debug)
    elif plan is None:
        max_blocks = 2048  # ad hoc
        errflag = cp.zeros(max_blocks, dtype=cp.uint32)
        gpu_mm_pybind11.unplanned_map2tod(tod, larr, xpointing, lpix, errflag, partial_pixelization)
    elif plan == 'reference':
        gpu_mm_pybind11.reference_map2tod(tod, larr, xpointing, lpix, partial_pixelization)
    elif (plan == 'old') or isinstance(plan, OldPointingPlan):
        assert lpix.is_simple_rectangle()
        assert larr.shape == (3, 64*lpix.nycells, 64*lpix.nxcells)
        assert not lpix.periodic_xcoord
        if isinstance(plan, OldPointingPlan):
            assert (plan.ndec, plan.nra) == (64*lpix.nycells, 64*lpix.nxcells)
        gpu_mm_pybind11.old_map2tod(tod, larr, xpointing)
    else:
        raise RuntimeError(f"Bad 'plan' argument to map2tod(): {plan}")


def tod2map(local_map, tod, xpointing, plan, partial_pixelization=False, debug=False):
    assert isinstance(local_map, LocalMap)
    lpix = local_map.pixelization
    larr = local_map.arr

    if isinstance(plan, PointingPlan):
        gpu_mm_pybind11.planned_tod2map(larr, tod, xpointing, lpix, plan, partial_pixelization, debug)
    elif plan is None:
        max_blocks = 2048  # ad hoc
        errflag = cp.zeros(max_blocks, dtype=cp.uint32)
        gpu_mm_pybind11.unplanned_tod2map(larr, tod, xpointing, lpix, errflag, partial_pixelization)
    elif plan == 'reference':
        gpu_mm_pybind11.reference_tod2map(larr, tod, xpointing, lpix, partial_pixelization)
    elif isinstance(plan, OldPointingPlan):
        assert lpix.is_simple_rectangle()
        assert larr.shape == (3, 64*lpix.nycells, 64*lpix.nxcells)
        assert not lpix.periodic_xcoord
        assert (plan.ndec, plan.nra) == (64*lpix.nycells, 64*lpix.nxcells)
        gpu_mm_pybind11.old_tod2map(larr, tod, xpointing, plan.plan_cltod_list, plan.plan_quadruples)
    else:
        raise RuntimeError(f"Bad 'plan' argument to tod2map(): {plan}")


####################################################################################################


class LocalPixelization(gpu_mm_pybind11.LocalPixelization):
    def __init__(self, nypix_global, nxpix_global, cell_offsets, ystride, polstride, periodic_xcoord = True):
        self.cell_offsets_cpu = cp.asnumpy(cell_offsets)   # numpy array
        self.cell_offsets_gpu = cp.asarray(cell_offsets)   # cupy array
        gpu_mm_pybind11.LocalPixelization.__init__(self, nypix_global, nxpix_global, self.cell_offsets_cpu, self.cell_offsets_gpu, ystride, polstride, periodic_xcoord)


    # Note: inherits the following members from C++
    #
    #   nypix_global      int
    #   nxpix_global      int
    #   periodic_xcoord   bool
    #   ystride           int
    #   polstride         int
    #   nycells           int
    #   nxcells           int
    #   npix              int
    
    def is_simple_rectangle(self):
        rect_offsets = self._make_rectangular_cell_offsets(self.nycells, self.nxcells)
        return np.array_equal(self.cell_offsets_cpu, rect_offsets)


    @classmethod
    def _make_rectangular_cell_offsets(cls, nycells, nxcells):
        cell_offsets = np.empty((nycells, nxcells), dtype=int)
        cell_offsets[:,:] = 64 * np.arange(nxcells).reshape((1,-1))
        cell_offsets[:,:] += 64*64 * nxcells * np.arange(nycells).reshape((-1,1))
        return cell_offsets
        
        
    @classmethod
    def make_rectangle(cls, nypix_global, nxpix_global, periodic_xcoord = True):
        assert nypix_global > 0
        assert nxpix_global > 0
        assert nypix_global % 64 == 0
        assert nxpix_global % 64 == 0

        nycells = (nypix_global + 63) // 64
        nxcells = (nxpix_global + 63) // 64

        return LocalPixelization(
            nypix_global,
            nxpix_global,
            cell_offsets = cls._make_rectangular_cell_offsets(nycells, nxcells),
            ystride = 64 * nxcells,
            polstride = 64 * 64 * nycells * nxcells,
            periodic_xcoord = periodic_xcoord
        )


class LocalMap:
    def __init__(self, pixelization, arr):
        assert isinstance(pixelization, LocalPixelization)
        self.pixelization = pixelization
        self.arr = arr


class PointingPrePlan(gpu_mm_pybind11.PointingPrePlan):
    """
    PointingPrePlan(xpointing_gpu, nypix_global, nxpix_global, buf=None, tmp_buf=None)

    Constructor arguments:
        preplan             instance of type gpu_mm.PointingPrePlan
        xpointing_gpu       shape (3, preplan.nsamp) array, on GPU
        buf                 1-d uint32 array with length PointingPrePlan.preplan_size
        tmp_buf             1-d uint32 array with length PointingPrePlan.preplan_size

    FIXME explain difference between 'buf' and 'tmp_buf'.

    Inherits from C++ base class:
        self.nsamp
        self.nypix_global
        self.nxpix_global
        self.plan_nbytes
        self.plan_constructor_tmp_nbytes
        self.overhead
        self.ncl_per_threadblock
        self.planner_nblocks
        self.nmt_per_threadblock
        self.pointing_nblocks
        self.plan_nmt
        self.cub_nbytes
        self.get_nmt_cumsum()     intended for unit tests
    """

    # static member
    preplan_size = gpu_mm_pybind11.PointingPrePlan._get_preplan_size()
    
    def __init__(self, xpointing_gpu, nypix_global, nxpix_global, buf=None, tmp_buf=None, periodic_xcoord=True, debug=False):
        if buf is None:
            buf = cp.empty(self.preplan_size, dtype=cp.uint32)
        if tmp_buf is None:
            tmp_buf = cp.empty(self.preplan_size, dtype=cp.uint32)

        gpu_mm_pybind11.PointingPrePlan.__init__(self, xpointing_gpu, nypix_global, nxpix_global, buf, tmp_buf, periodic_xcoord, debug)


class PointingPlan(gpu_mm_pybind11.PointingPlan):
    """
    PointingPlan(preplan, xpointing_gpu, buf=None, tmp_buf=None)

    Constructor arguments:
        preplan             instance of type gpu_mm.PointingPrePlan
        xpointing_gpu       shape (3, preplan.nsamp) array, on GPU
        buf                 1-d uint8 array with length >= preplan.plan_nbytes
        tmp_buf             1-d uint8 array with length >= preplan.plan_constructor_tmp_nbytes

    FIXME explain difference between 'buf' and 'tmp_buf'.

    Inherits from C++ base class:
        self.nsamp          int
        self.nypix_global          int
        self.nxpix_global          int
        self.get_plan_mt()  returns 1-d uint64 array of length preplan.get_nmt_cumsum()[-1]
        self.__str__()
    """
    
    def __init__(self, preplan, xpointing_gpu, buf=None, tmp_buf=None, debug=False):
        if buf is None:
            buf = cp.empty(preplan.plan_nbytes, dtype=cp.uint8)
        if tmp_buf is None:
            tmp_buf = cp.empty(preplan.plan_constructor_tmp_nbytes, dtype=cp.uint8)

        gpu_mm_pybind11.PointingPlan.__init__(self, preplan, xpointing_gpu, buf, tmp_buf, debug)

            
####################################################################################################


class OldPointingPlan(gpu_mm_pybind11.OldPointingPlan):
    def __init__(self, xpointing, ndec, nra, verbose=True):
        assert (ndec % 64) == 0
        assert (nra % 64) == 0
        gpu_mm_pybind11.OldPointingPlan.__init__(self, xpointing, ndec, nra, verbose)
        self.plan_cltod_list = cp.asarray(self._plan_cltod_list)  # CPU -> GPU
        self.plan_quadruples = cp.asarray(self._plan_quadruples)  # CPU -> GPU
        self.ndec = ndec
        self.nra = nra

    def map2tod(self, tod, m, xpointing):
        gpu_mm_pybind11.old_map2tod(tod, m, xpointing)
            
    def tod2map(self, m, tod, xpointing):
        gpu_mm_pybind11.old_tod2map(m, tod, xpointing, self.plan_cltod_list, self.plan_quadruples)


class ToyPointing(gpu_mm_pybind11.ToyPointing):
    """
    ToyPointing(nsamp, nypix_global, nxpix_global, scan_speed, total_drift)
    Scans currently go at 45 degrees, and cover the full ypix-range.

    Constructor arguments:
      scan_speed = map pixels per TOD sample
      total_drift = total drift over full TOD, in x-pixels

    Python members:
      self.xpointing_cpu    shape (3,nsamp) numpy array
      self.xpointing_gpu    shape (3,nsamp) numpy array
      self.ndet             can be None
      self.nt               int

    Inherits from C++ base class:
        self.nsamp          int
        self.nypix_global          int
        self.nxpix_global          int
        self.scan_speed     float
        self.total_drift    float
        self.drift_speed    float
        self.__str__()
    """

    def __init__(self, ndet, nt, nypix_global, nxpix_global, scan_speed, total_drift, noisy=True):
        xpointing_shape = (3,ndet,nt) if (ndet is not None) else (3,nt)
        self.xpointing_cpu = np.zeros(xpointing_shape, mm_dtype)
        self.xpointing_gpu = cp.zeros(xpointing_shape, mm_dtype)
        gpu_mm_pybind11.ToyPointing.__init__(self, nypix_global, nxpix_global, scan_speed, total_drift, self.xpointing_cpu, self.xpointing_gpu, noisy)


    @staticmethod
    def make_random(nsamp_max, noisy=True):
        assert nsamp_max >= 64*1024
        
        if np.random.randint(0,2):
            # Case 1: TOD arrays have shape (nsamp)
            ndet = None
            nt = 32 * np.random.randint(nsamp_max//64, nsamp_max//32)
        else:
            # Case 2: TOD arrays have shape (ndet,nt)
            r = np.random.randint(0,6)
            s = int((nsamp_max/32.)**0.5)
            ndet = 2**r * np.random.randint(s//2,s+1)
            nt = 2**(5-r) * np.random.randint(s//2,s+1)
            
        npix_max = min(nsamp_max//128, 16384)
        nypix_global = 64 * np.random.randint(npix_max//512, npix_max//128)
        nxpix_global = 128 * np.random.randint(nypix_global//128 + 2, npix_max//128)
        scan_speed = np.random.uniform(0.1, 0.5)
        total_drift = np.random.uniform(0.1*(nxpix_global-nypix_global), (nxpix_global-nypix_global)-2)
                    
        return ToyPointing(ndet, nt, nypix_global, nxpix_global, scan_speed, total_drift, noisy=noisy)


class PointingPlanTester(gpu_mm_pybind11.PointingPlanTester):
    """
    PointingPlanTester(preplan, xpointing_gpu).

    Only used in unit tests.
    Inherited from C++:

      self.nsamp
      self.nypix_global
      self.nxpix_global
      self.rk
      self.nblocks

      self.iypix         shape (nsamp,)
      self.ixpix         shape (self.nsamp,)
      self.nmt_cumsum    shape (self.nblocks,)
      self.sorted_mt     shape (self.nmt,)
    """

    def __init__(self, preplan, xpointing_gpu):
        assert isinstance(preplan, PointingPrePlan)
        
        tmp_nbytes = gpu_mm_pybind11.PointingPlanTester.get_constructor_tmp_nbytes(preplan)
        tmp = cp.empty(tmp_nbytes, dtype=cp.uint8)
        
        gpu_mm_pybind11.PointingPlanTester.__init__(self, preplan, xpointing_gpu, tmp)
        
        
####################################################################################################

_libgpu_mm = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libgpu_mm.so"))

# All function arguments are pointers, ints, longs, or floats
_i = ctypes.c_int
_l = ctypes.c_long
_p = ctypes.c_void_p
_f = ctypes.c_float

_clip = _libgpu_mm.clip
_clip.argtypes = (_p, _l, _f, _f)

_extract_ranges = _libgpu_mm.extract_ranges
_extract_ranges.argtypes = (_p, _i, _p, _p, _i, _p, _p, _p)

_insert_ranges = _libgpu_mm.insert_ranges
_insert_ranges.argtypes = (_p, _i, _p, _p, _i, _p, _p, _p)

_clear_ranges = _libgpu_mm.clear_ranges
_clear_ranges.argtypes = (_p, _i, _i, _p, _p, _p)


# Cuts
def insert_ranges(tod, junk, offs, dets, starts, lens):
    _insert_ranges(tod.data.ptr, tod.shape[1], junk.data.ptr, offs.data.ptr, len(lens), dets.data.ptr, starts.data.ptr, lens.data.ptr)

def extract_ranges(tod, junk, offs, dets, starts, lens):
    _extract_ranges(tod.data.ptr, tod.shape[1], junk.data.ptr, offs.data.ptr, len(lens), dets.data.ptr, starts.data.ptr, lens.data.ptr)

def clear_ranges(tod, dets, starts, lens):
    _clear_ranges(tod.data.ptr, tod.shape[1], len(lens), dets.data.ptr, starts.data.ptr, lens.data.ptr)

# ----------
# Misc stuff

def clip(arr, vmin, vmax):
    """In-place clip. Necessary because cupy's clip makes a copy, even when
    the out argument is used."""
    assert arr.dtype == np.float32, "In-place clip only supports float32"
    _clip(arr.data.ptr, arr.size, vmin, vmax)
