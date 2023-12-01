# This source file will eventually become a proper python module that 'pip' can build.
#
# Currently, it contains some hackery (e.g. hardcoded pathname ../lib/libgpu_mm.so).
# To use it, you'll need to build the C++ library ('make -j' from the toplevel gpu_mm dir),
# and then do 'import gpu_mm' from the gpu_mm/scripts directory.

import ctypes
import numpy as np
import cupy as cp

_libgpu_mm = ctypes.cdll.LoadLibrary("../lib/libgpu_mm.so")

# py_reference_map2tod(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra);
_reference_map2tod = _libgpu_mm.py_reference_map2tod
_reference_map2tod.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

# py_reference_map2tod(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra);
_gpu_map2tod = _libgpu_mm.py_gpu_map2tod
_gpu_map2tod.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)


def reference_map2tod(tod_out, map_in, xpointing):
    """
    Slow (single-threaded CPU) reference implementation of map2tod().

    tod_out: float32 numpy array of shape (ndetectors, ntimes).
        This array gets overwritten by reference_map2tod().
        Current implementation asserts (ntimes % 32) == 0.

    map_in: float32 numpy array of shape (3, ndec, nra).
        Current implementation asserts (ndec % 64) == (nra % 64) == 0.
        The length-3 axis is {I,Q,U}.

    xpointing: float32 numpy array of shape (3, ndetectors, ntimes).
   
        The length-3 axis is {px_dec, px_ra, alpha}, where:

            - px_dec is declination in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - px_ra is right ascension in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - alpha is detector angle in radians.

        ('xpointing' is short for "exploded pointing", i.e. per-detector pointing)
    """
    
    assert isinstance(tod_out, np.ndarray)
    assert tod_out.dtype == np.float32
    assert tod_out.flags.c_contiguous
    assert tod_out.ndim == 2
    ndet, nt = tod_out.shape
    
    assert isinstance(map_in, np.ndarray)
    assert map_in.dtype == np.float32
    assert map_in.flags.c_contiguous
    assert map_in.ndim == 3
    assert map_in.shape[0] == 3
    ndec, nra = map_in.shape[1], map_in.shape[2]
    
    assert isinstance(xpointing, np.ndarray)
    assert xpointing.dtype == np.float32
    assert xpointing.flags.c_contiguous
    assert xpointing.ndim == 3
    assert xpointing.shape[0] == 3
    assert xpointing.shape[1] == tod_out.shape[0]
    assert xpointing.shape[2] == tod_out.shape[1]

    assert nt > 0
    assert ndet > 0
    assert ndec > 0
    assert nra > 0

    # Assumed in current implementation, could be relaxed if needed.
    assert (nt % 32) == 0
    assert (ndec % 64) == 0
    assert (nra % 64) == 0

    _reference_map2tod(tod_out.ctypes.data, map_in.ctypes.data, xpointing.ctypes.data, ndet, nt, ndec, nra)


def gpu_map2tod(tod_out, map_in, xpointing):
    """
    GPU implementation of map2tod().

    tod_out: float32 cupy array of shape (ndetectors, ntimes).
        This array gets overwritten by reference_map2tod().
        Current implementation asserts (ntimes % 32) == 0.

    map_in: float32 cupy array of shape (3, ndec, nra).
        Current implementation asserts (ndec % 64) == (nra % 64) == 0.
        The length-3 axis is {I,Q,U}.

    xpointing: float32 cupy array of shape (3, ndetectors, ntimes).
   
        The length-3 axis is {px_dec, px_ra, alpha}, where:

            - px_dec is declination in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - px_ra is right ascension in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - alpha is detector angle in radians.

        ('xpointing' is short for "exploded pointing", i.e. per-detector pointing)

    FIXME: currently calls cudaDeviceSynchronize() before and after kernel launch,
    which could slow things down.
    """

    assert isinstance(tod_out, cp.ndarray)
    assert tod_out.dtype == cp.float32
    assert tod_out.flags.c_contiguous
    assert tod_out.ndim == 2
    ndet, nt = tod_out.shape
    
    assert isinstance(map_in, cp.ndarray)
    assert map_in.dtype == cp.float32
    assert map_in.flags.c_contiguous
    assert map_in.ndim == 3
    assert map_in.shape[0] == 3
    ndec, nra = map_in.shape[1], map_in.shape[2]
    
    assert isinstance(xpointing, cp.ndarray)
    assert xpointing.dtype == cp.float32
    assert xpointing.flags.c_contiguous
    assert xpointing.ndim == 3
    assert xpointing.shape[0] == 3
    assert xpointing.shape[1] == tod_out.shape[0]
    assert xpointing.shape[2] == tod_out.shape[1]

    assert nt > 0
    assert ndet > 0
    assert ndec > 0
    assert nra > 0

    # Assumed in current implementation, could be relaxed if needed.
    assert (nt % 32) == 0
    assert (ndec % 64) == 0
    assert (nra % 64) == 0

    _gpu_map2tod(tod_out.data.ptr, map_in.data.ptr, xpointing.data.ptr, ndet, nt, ndec, nra)
