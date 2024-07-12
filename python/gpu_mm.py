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


# PointingPrePlan: imported from gpu_mm_pybind11 (for details, see src_pybind11/gpu_mm_pybind11.cu,
# or read docstrings in the python interpreter).
#
# class PointingPrePlan:
#     self.__init__(xpointing_gpu, nypix, nxpix)
#     self.nsamp
#     self.nypix
#     self.nxpix
#     self.plan_nbytes
#     self.plan_constructor_tmp_nbytes
#     self.rk
#     self.nblocks
#     self.plan_nmt
#     self.cub_nbytes
#     self.get_nmt_cumsum()


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
        self.nypix          int
        self.nxpix          int
        self.get_plan_mt()  returns 1-d uint64 array of length preplan.get_nmt_cumsum()[-1]
        self.__str__()

        self.map2tod(tod, map, xpointing_gpu, debug=False)
           -> overwrites output 'tod' array

        self.tod2map(map, tod, xpointing_gpu, debug=False)
           -> accumulates into output 'map' array
    """
    
    def __init__(self, preplan, xpointing_gpu, buf=None, tmp_buf=None):
        if buf is None:
            buf = cp.empty(preplan.plan_nbytes, dtype=cp.uint8)
        if tmp_buf is None:
            tmp_buf = cp.empty(preplan.plan_constructor_tmp_nbytes, dtype=cp.uint8)

        gpu_mm_pybind11.PointingPlan.__init__(self, preplan, xpointing_gpu, buf, tmp_buf)

            
    

####################################################################################################


class ToyPointing(gpu_mm_pybind11.ToyPointing):
    """
    ToyPointing(nsamp, nypix, nxpix, scan_speed, total_drift)
    Scans currently go at 45 degrees, and cover the full ypix-range.

    Constructor arguments:
      scan_speed = map pixels per TOD sample
      total_drift = total drift over full TOD, in x-pixels

    Python members:
      self.xpointing_cpu    shape (3,nsamp) numpy array
      self.xpointing_gpu    shape (3,nsamp) numpy array

    Inherits from C++ base class:
        self.nsamp          int
        self.nypix          int
        self.nxpix          int
        self.scan_speed     float
        self.total_drift    float
        self.drift_speed    float
        self.__str__()
    """

    def __init__(self, nsamp, nypix, nxpix, scan_speed, total_drift, noisy=True):
        assert nsamp > 0
        self.xpointing_cpu = np.zeros((3,nsamp), mm_dtype)
        self.xpointing_gpu = cp.zeros((3,nsamp), mm_dtype)
        gpu_mm_pybind11.ToyPointing.__init__(self, nsamp, nypix, nxpix, scan_speed, total_drift, self.xpointing_cpu, self.xpointing_gpu, noisy)


    @staticmethod
    def make_random(nsamp_max, noisy=True):
        assert nsamp_max >= 64*1024
        npix_max = min(nsamp_max//128, 16384)
        nsamp = 32 * np.random.randint(nsamp_max//64, nsamp_max//32)
        nypix = 64 * np.random.randint(npix_max//512, npix_max//128)
        nxpix = 128 * np.random.randint(nypix//128 + 2, npix_max//128)
        scan_speed = np.random.uniform(0.1, 0.5)
        total_drift = np.random.uniform(0.1*(nxpix-nypix), (nxpix-nypix)-2)
        return ToyPointing(nsamp, nypix, nxpix, scan_speed, total_drift, noisy=noisy)


class ReferencePointingPlan(gpu_mm_pybind11.ReferencePointingPlan):
    """
    ReferencePointingPlan(preplan, xpointing_gpu).

    Only used in unit tests.
    Inherited from C++:

      self.nsamp
      self.nypix
      self.nxpix
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
