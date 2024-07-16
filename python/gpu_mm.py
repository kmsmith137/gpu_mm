# This source file will eventually become a proper python module that 'pip' can build.
#
# Currently, it contains some hackery (e.g. hardcoded pathname ../lib/libgpu_mm.so).
# To use it, you'll need to build the C++ library ('make -j' from the toplevel gpu_mm dir),
# and then do 'import gpu_mm' from the gpu_mm/scripts directory.

import ctypes, os
import numpy as np
import cupy as cp

from . import gpu_mm_pybind11
from .gpu_mm_pybind11 import PointingPrePlan

# Currently we wrap either float32 or float64, determined at compile time!
# FIXME eventually, should wrap both.
mm_dtype = np.float64 if (gpu_mm_pybind11._get_tsize() == 8) else np.float32

        
class LocalPixelization(gpu_mm_pybind11.LocalPixelization):
    def __init__(self, nypix_global, nxpix_global, cell_offsets, ystride, polstride, periodic_xcoord = True):
        cell_offsets_cpu = cp.asnumpy(cell_offsets)
        cell_offsets_gpu = cp.asarray(cell_offsets)
        gpu_mm_pybind11.LocalPixelization.__init__(self, nypix_global, nxpix_global, cell_offsets_cpu, cell_offsets_gpu, ystride, polstride, periodic_xcoord)

        
    @staticmethod
    def make_rectangle(nypix_global, nxpix_global, periodic_xcoord = True):
        assert nypix_global > 0
        assert nxpix_global > 0
        assert nypix_global % 64 == 0
        assert nxpix_global % 64 == 0

        nycells = nypix_global // 64
        nxcells = nxpix_global // 64

        cell_offsets = np.empty((nycells, nxcells), dtype=int)
        cell_offsets[:,:] = 64 * np.arange(nxcells).reshape((1,-1))
        cell_offsets[:,:] += 64 * nxpix_global * np.arange(nycells).reshape((-1,1))

        return LocalPixelization(
            nypix_global, nxpix_global, cell_offsets,
            ystride = nxpix_global,
            polstride = nypix_global*nxpix_global,
            periodic_xcoord = periodic_xcoord
        )


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

        self.map2tod(tod, map, xpointing_gpu, debug=False)
           -> overwrites output 'tod' array

        self.tod2map(map, tod, xpointing_gpu, debug=False)
           -> accumulates into output 'map' array
    """
    
    def __init__(self, preplan, xpointing_gpu, buf=None, tmp_buf=None, debug=False):
        if buf is None:
            buf = cp.empty(preplan.plan_nbytes, dtype=cp.uint8)
        if tmp_buf is None:
            tmp_buf = cp.empty(preplan.plan_constructor_tmp_nbytes, dtype=cp.uint8)

        gpu_mm_pybind11.PointingPlan.__init__(self, preplan, xpointing_gpu, buf, tmp_buf, debug)

            
####################################################################################################


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


class ReferencePointingPlan(gpu_mm_pybind11.ReferencePointingPlan):
    """
    ReferencePointingPlan(preplan, xpointing_gpu).

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
        
        tmp_nbytes = gpu_mm_pybind11.ReferencePointingPlan.get_constructor_tmp_nbytes(preplan)
        tmp = cp.empty(tmp_nbytes, dtype=cp.uint8)
        
        gpu_mm_pybind11.ReferencePointingPlan.__init__(self, preplan, xpointing_gpu, tmp)
        
        
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
