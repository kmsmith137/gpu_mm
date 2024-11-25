"""
==============================
MAP2TOD AND TOD2MAP OPERATIONS
==============================

Most of this file is concerned with defining two functions:

  map2tod(timestream, local_map, xpointing, plan, partial_pixelization=False, debug=False)
  tod2map(local_map, timestream, xpointing, plan, partial_pixelization=False, debug=False)

The map2tod() function overwrites the output TOD array, and the tod2map() function accumulates
its output to the existing map array.

In this top-level docstring, we define central data structures ("global" and "local" maps,
"xpointing", pointing plans, etc.)

Please also refer to this example script: scripts/gpu_mm_example.py

=========================
"GLOBAL" PIXEL-SPACE MAPS
=========================

A "global" map is a 3-d array of shape 

   float32 global_map[3][nypix_global][nxpix_global]   # first coord is {I,Q,U}

where the y-coordinate might be declination, and the x-coordinate might be RA.
Note that maps are indexed as map[y][x], not map[x,y].

The "global" pixelization is specified by three parameters:  

  nypix_global (int)
  nxpix_global (int)
  periodic_xcoord (bool)

The global pixelization might represent the entire sky, even in a situation
where the map-maker is running on a small sky fraction. (The "local map
data structure can be used to avoid storing the full global map -- there's 
no requirement that global maps be stored either on CPU or GPU. See below!)

The 'periodic_xcoord' parameter is intended to offer the choice between
putting "wrapping" logic on the CPU or the GPU. If 'periodic_xcoord' is
true, then the GPU is aware that the global pixelization is periodic, and
will automatically "wrap" the xpointing (see below) in map2tod/tod2map
operations. If 'periodic_xcoord' is false, then the GPU won't perform
"wrapping", and the CPU caller is responsible for wrapping.

===========
TIMESTREAMS
===========

Currently, a timestream is a either a 1-d or a 2-d array
 
   float32 tod[nsamp];
   float32 tod[ndet][nt];

(Internally, the 2-d case is handled by "flattening" to a 1-d array.)

================
XPOINTING ARRAYS
================

The map2tod/tod2map operations use "exploded pointing" throughout,
represented as either a 2-d or a 3-d array:

   float32 xpointing[3][nsamp];     // axis 0 is {y,x,alpha}
   float32 xpointing[3][ndet][nt];  // axis 0 is {y,x,alpha}

That is, xpointing[0,:] contains (global) y-coordinates and xpointing[1,:]
contains x-coordinates. These coordinates must satisfy the following
constraints:

  1 <= y <= (nypix_global-2)
  1 <= x <= (nxpix_global-2)                  if periodic_xcoord == False
  -nxpix_global+1 <= x <= 2*nxpix_global-2    if periodic_xcoord == True

That is, the global pixelization must be "at least large enough to contain the
xpointing", even if the map-maker is running on a smaller sky fraction (see
next item, "Local maps").

The 'alpha' values in xpointing[2,:] are detector angles, defined by the
statement that detectors measure:

   T + cos(2*alpha)*Q + sin(2*alpha)*U

========================
"LOCAL" PIXEL-SPACE MAPS
========================

This is the most complicated part, since it's designed to represent a
variety of data layouts.

Consider a situation where we don't want to store an entire global map in
memory. (This could either be on the GPU or CPU.) We define a "local map"
data structure, to store a partial sky map, as follows.

We divide the global map into 64-by-64 "cells". A local map is an arbitrary
set of cells, packed into a contiguous memory region of length (3*64*64*ncells).

(If nxpix_global or nypix_global is not divisible by 64, then the local map may
include some "boundary" pixels which are not in the global map. This is allowed,
and boundary pixels are ignored by the tod2map() and map2tod() operations.)

We define the following data structure (the "local pixelization") which describes 
which cells are stored, and how cells are mapped to memory addresses:

  cell_offsets   2-d integer-valued array indexed by (iycell, ixcell)
  ystride        integer
  polstride      integer

These members have the following meaning. Consider one cell in the _global_ 
map, i.e. pixel index ranges

   64*iycell <= iypix_global < 64*(iycell+1)
   64*iycell <= iypix_global < 64*(iycell+1)

If cell_offsets[iycell,ixcell] is negative, then cell (iycell, ixcell)
is not stored as part of the local map.

If cell_offsets[iycell,ixcell] is >= 0, then the cell is a logical array
of shape (3,64,64), and array element (pol,iy,ix) is stored at the
following memory address:

  cell_offsets[iycell,ixcell] + (pol * polstride) + (iy * ystride) + ix

Here are some examples of local maps, just to illustrate the flexibility:

 - Consider a rectangular subset of the global map, represented as a 3-d
   contiguous array with shape:

     float32 local_map[3][64*nycells][64*nxcells]
    
   This could be represented as a local map with:

     ystride = 64*nxcells
     polstride = 64*64*nycells*nxcells
      (cell_offsets chosen appropriately)

 - Consider a 1-d array of cells, where each cell is contiguous in memory,
   represented as a 4-d array:

     float32 local_map[ncells][3][64][64]

   This could be represented as a local map with:

     ystride = 64
     polstride = 64*64
      (cell_offsets chosen appropriately)

===============================
POINTING "PLANS" AND "PREPLANS"
===============================

The most complicated part of the code is creating a "PointingPlan" on the GPU.
This is a parallel data structure that is initialized from an xpointing array,
and accelerates map2tod/tod2map operations.

I won't describe the plan in detail, except to say that we factor plan creation
into two steps:

   - Create PointingPrePlan from xpointing array (takes ~5 milliseconds
     and needs a few KB to store)

   - Create PointingPlan from PointingPrePlan + xpointing array (takes ~5
     milliseconds and needs ~100 MB to store)

The idea is that PointingPrePlans are small and can be retained (per-TOD)
for the duration of the map-maker, whereas PointingPlans are large and 
should be created/destroyed on the fly.

=========
TODO LIST
=========

  - Right now the code is not very well tested! I think testing is my 
    next priority.

  - Currently, the number of TOD samples 'nsamp' must be a multiple of 32.
    (I'd like to change this, but it's not totally trivial, and there are a
     few minor issues I'd like to chat about.)

  - Helper functions for converting maps between different pixelizations
    (either local or global, with or without wrapping logic).

  - An MPIPixelization class with all-to-all logic for distirbuting/reducing
    maps across GPUs.

  - Kernels should be launched on current cupy stream (I'm currently launching
    on the default cuda stream).

  - Support both float32 and float64.

  - There are still some optimizations I'd like to explore, for making map2tod()
    and tod2map() even faster, but I put this on the back-burner since they're
    pretty fast already.

  - Not currently pip-installable (or conda-installable). This turned out to 
    be a real headache. I put it "on pause" but I plan to go back to it later.

None of these todo items should be a lot of work individually, but I'm not sure 
how to prioritize.
"""


import ctypes, os
import numpy as np
import cupy as cp

from . import gpu_mm_pybind11

# Currently we wrap either float32 or float64, determined at compile time!
# FIXME eventually, should wrap both.
mm_dtype = np.float64 if (gpu_mm_pybind11._get_tsize() == 8) else np.float32


def map2tod(tod, local_map, xpointing, plan, partial_pixelization=False, debug=False):
    """
    Arguments:
    
      - tod: output array (1-dimensional, length-nsamp).
         (See "timestreams" in top-level gpu_mm docstring)
         Will be overwritten by map2tod().

      - local_map: input array (instance of class LocalMap).
         (See "local maps" in top-level gpu_mm docstring)

      - xpointing: shape (3,nsamp) array.
         (See "xpointing arrays" in top-level gpu_mm docstring)
    
      - plan: either instance of class PointingPlan, or one of
         the following special values:
    
          - None: use plan-free map2tod code (slower)
          - 'reference': use CPU-based reference code (intended for testing)
    
       - partial_pixelization: does caller intend to run the map-maker
          on a subset of the sky covered by the observations?

          - if False, then an exception will be thrown if any xpointing
            values are outside the local_map.
    
          - if True, then xpointing values outside the local_map are
            treated as zeroed pixels.
    """

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
    else:
        raise RuntimeError(f"Bad 'plan' argument to map2tod(): {plan}")


def tod2map(local_map, tod, xpointing, plan, partial_pixelization=False, debug=False):
    """
    Arguments:
    
      - local_map: output array (instance of class LocalMap or DynamicMap).
         (See "local maps" and "dynamic maps" in top-level gpu_mm docstring)
         tod2map() will accumulate its output (i.e. add to existing array contents)

      - tod: input array (1-dimensional, length-nsamp).
         (See "timestreams" in top-level gpu_mm docstring)

      - xpointing: shape (3,nsamp) array.
         (See "xpointing arrays" in top-level gpu_mm docstring)
    
      - plan: either instance of class PointingPlan, or one of
         the following special values:
    
          - None: use plan-free map2tod code (slower)
          - 'reference': use CPU-based reference code (intended for testing)
    
       - partial_pixelization: does caller intend to run the map-maker
          on a subset of the sky covered by the observations?

          - if False, then an exception will be thrown if any xpointing
            values are outside the local_map.
    
          - if True, then xpointing values outside the local_map are
            treated as zeroed pixels.
    """

    if isinstance(local_map, LocalMap):
        lpix = local_map.pixelization
        larr = local_map.arr        
    elif isinstance(local_map, DynamicMap):
        if not isinstance(plan, PointingPlan):
            raise RuntimeError("tod2map(): if output is a DynamicMap, then the 'plan' argument must be a PointingPlan")
        local_map.extend(plan)
        lpix = local_map._unstable_pixelization  # okay to use temporarily in this function
        larr = local_map._unstable_arr           # okay to use temporarily in this function
    else:
        raise RuntimeError(f"tod2map(): Bad 'local_map' argument to tod2map(): expected LocalMap or DynamicMap, got: {local_map}")

    if isinstance(plan, PointingPlan):
        gpu_mm_pybind11.planned_tod2map(larr, tod, xpointing, lpix, plan, partial_pixelization, debug)
    elif plan is None:
        max_blocks = 2048  # ad hoc
        errflag = cp.zeros(max_blocks, dtype=cp.uint32)
        gpu_mm_pybind11.unplanned_tod2map(larr, tod, xpointing, lpix, errflag, partial_pixelization)
    elif plan == 'reference':
        gpu_mm_pybind11.reference_tod2map(larr, tod, xpointing, lpix, partial_pixelization)
    else:
        raise RuntimeError(f"Bad 'plan' argument to tod2map(): {plan}")


####################################################################################################


class LocalPixelization(gpu_mm_pybind11.LocalPixelization):
    """
    This class contains the following members, which describe the global
    pixelization (see "global maps" in the gpu_mm docstring):

       nypix_global (int)
       nxpix_global (int)
       periodic_xcoord (bool)

    and the following members, which describe a local pixelization (i.e.
    subset of 64-by-64 cells in the global pixelization, see "local maps"
    in the gpu_mm docstring):

      cell_offsets   2-d integer-valued array indexed by (iycell, ixcell)
      ystride        integer
      polstride      integer
    """
    
    def __init__(self, nypix_global, nxpix_global, cell_offsets, ystride, polstride, periodic_xcoord = True):
        self.cell_offsets_cpu = cp.asnumpy(cell_offsets)   # numpy array
        self.cell_offsets_gpu = cp.asarray(cell_offsets)   # cupy array
        gpu_mm_pybind11.LocalPixelization.__init__(self, nypix_global, nxpix_global, self.cell_offsets_cpu, self.cell_offsets_gpu, ystride, polstride, periodic_xcoord)


    # Note: inherits the following members from C++ (via pybind11)
    #
    #   nypix_global      int
    #   nxpix_global      int
    #   periodic_xcoord   bool
    #   ystride           int
    #   polstride         int
    #   nycells           int (same as cell_offsets.shape[0])
    #   nxcells           int (same as cell_offsets.shape[0])
    #   npix              int (counts only local pixels, does not include factor 3 from TQU)
    
    def is_simple_rectangle(self):
        """
        The simplest case of a local pixelization is a 3-d contiguous array of shape
        (3, 64*nycells, 64*nxcells). This function returns True if the LocalPixelization
        is of this simple type.
        """
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
        """
        Returns a trivial LocalPixelization which covers the entire global sky.
        The "local" map will represented as a 3-d contiguous array of shape 
        (3, nypix_padded, nxpix_padded), where:
        
           nypix_padded = (nypix_global + 63) & ~63   # round up to multiple of 64
           nxpix_padded = (nxpix_global + 63) & ~63   # round up to multiple of 64

        This is useful because map2tod() and tod2map() operate on local maps.
        If you want to operate on a global map instead, you can convert a
        global map to a local map as follows:

           global_map = ....    # shape (3, nypix_global, nxpix_global)

           # Pad global map.
           # This step is only needed if nypix_global or nxpix_global is not a multiple of 64.

           nypix_padded = (nypix_global + 63) & ~63   # round up to multiple of 64
           nxpix_padded = (nxpix_global + 63) & ~63   # round up to multiple of 64
           padded_map = cp.zeros((3, nypix_padded, nxpix_padded), dtype = cp.float32)
           padded_map[:,:nypix_global,:nxpix_global] = global_map

           rect_pix = LocalPixelization.make_rectangle(nypix_global, nxpix_global)
           local_map = LocalMap(rect_pix, global_map)
        """
        
        assert nypix_global > 0
        assert nxpix_global > 0

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
    """
    A LocalMap consists of:
        
       - A LocalPixelization, which describes which 64-by-64 map cells are in
         the local map, and their layout in memory.

       - An array 'arr' which contains the actual map. (Can be either a numpy
         array on the CPU, or a cupy array on the GPU.)

    LocalMaps are output arrays to tod2map(), and input arrays to map2tod().
    (If you want these functions to operate on global maps instead, see the
    LocalPixelization.make_rectangle() docstring.)
    """
        
    def __init__(self, pixelization, arr):
        assert isinstance(pixelization, LocalPixelization)
        self.pixelization = pixelization
        self.arr = arr
        
        # FIXME should add some error-checking on arr.shape, arr.dtype


####################################################################################################


class DynamicMap:
    def __init__(self, nypix_global, nxpix_global, dtype, cell_mask=None, periodic_xcoord=True, initial_ncells=1024):
        """cell_mask: shape (nycells, nxcells) boolean array"""

        # FIXME more argument checking
        assert initial_ncells > 0
        
        self.nypix_global = nypix_global
        self.nxpix_global = nxpix_global
        self.dtype = dtype
        self.ncells_allocated = initial_ncells
        self.reallocation_factor = 1.5
        self.kernel_nblocks = 1024

        # Make 'cell_offsets' array (see LocalPixelization docstring)
        # Initialize self.max_cells.
        
        if cell_mask is None:
            nycells = (nypix_global + 63) // 64
            nxcells = (nxpix_global + 63) // 64
            cell_offsets = np.full((nycells,nxcells), -1, dtype=int)
        else:
            cell_mask = np.asarray(cell_mask)
            assert cell_mask.ndim == 2
            assert cell_mask.dtype == bool
            cell_offsets = np.where(cell_mask, -1, -2)
        
        # Make LocalPixelization (initially empty)
        self._unstable_pixelization = LocalPixelization(
            nypix_global = nypix_global,
            nxpix_global = nxpix_global,
            cell_offsets = cell_offsets,
            ystride = 64,
            polstride = 64*64,
            periodic_xcoord = periodic_xcoord)

        # Make local map (initially empty)
        self._padded_arr = cp.zeros(self.ncells_allocated * 3*64*64, dtype=self.dtype)
        self._unstable_arr = self._padded_arr[:0]   # shape (0,)

        # The cuda kernel (expand_dynamic_map()) needs an auxiliary one-element array, to track
        # current number of cells. (We allocate a 32-element cache line and truncate.)
        self.global_ncells = cp.zeros(32, dtype=cp.uint32)
        self.global_ncells = self.global_ncells[:1]
        
        self.ncells_curr = 0
        self.is_finalized = False
        

    def extend(self, plan):
        """I think this will be moved to a non-member function."""

        # More argument checking!
        # E.g. should check consistency of (nypix_global, nxpix_global, periodic_xcoord)
        assert isinstance(plan, PointingPlan)
        assert not self.is_finalized

        self.ncells_curr = gpu_mm_pybind11.expand_dynamic_map2(self.global_ncells, self._unstable_pixelization, plan)

        if self.ncells_curr > self.ncells_allocated:
            self.ncells_allocated = int(self.reallocation_factor * self.ncells_allocated)
            self.ncells_allocated = max(self.ncells_allocated, self.ncells_curr)
            self._padded_arr = cp.zeros(self.ncells_allocated * 3*64*64, dtype=self.dtype)
            self._padded_arr[:len(self._unstable_arr)] = self._unstable_arr[:]

        self._unstable_pixelization.npix = self.ncells_curr * (64*64)
        self._unstable_arr = self._padded_arr[:(self.ncells_curr * 3*64*64)]


    def finalize(self):
        """Returns a LocalMap."""
        
        assert not self.is_finalized

        self._unstable_pixelization.copy_gpu_offsets_to_cpu()
        assert self._unstable_pixelization.npix == (self.ncells_curr * 64 * 64)

        # Shrink overallocated map, and delete original.
        arr = cp.copy(self._unstable_arr)
        self._padded_arr = self._unstable_arr = None
        self.is_finalized = True
        
        return LocalMap(self._unstable_pixelization, arr)
        

####################################################################################################


class PointingPrePlan(gpu_mm_pybind11.PointingPrePlan):
    """
    The GPU pointing 'plan' is an argument to map2tod() or tod2map().
    You'll construct the PointingPrePlan before constructing the PointingPlan.
    (See "plans and preplans" in the gpu_mm docstring.)

    Constructor arguments:
     - xpointing_gpu       shape (3, nsamp) array, see "xpointing" in gpu_mm docstring
     - nypix_global        see "global maps" in gpu_mm docstring
     - nxpix_global        see "global maps" in gpu_mm docstring
     - periodic_xcoord     see "global maps" in gpu_mm docstring
     - buf                 1-d uint32 array with length PointingPrePlan.preplan_size (see below)
     - tmp_buf             1-d uint32 array with length PointingPrePlan.preplan_size (see below)

    The 'buf' and 'tmp_buf' arrays are allocated from cupy, and populated by the C++ constructor.
    If these arrays are None (the default), then they'll be allocated and freed on-the-fly.
    However, you may find it more efficient to use preallocated buffers, to avoid the overhead
    of this on-the-fly allocation.
    
    IMPORTANT: the PointingPrePlan keeps a reference to the 'buf' array and assumes that it
    has exclusive access, whereas the 'tmp_buf' array is only used by the constructor temporarily.
    Therefore, if you're constructing multiple PointingPrePlans, you can use the same 'tmp_buf'
    for all of them, but 'buf' must be different.

    The PointingPrePlan does not keep a reference to the 'xpointing' array.

    Inherits from C++ base class (via pybind11):
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

    # Static member. Currently 1024 (defined in gpu_mm.hpp).
    preplan_size = gpu_mm_pybind11.PointingPrePlan._get_preplan_size()
    
    def __init__(self, xpointing_gpu, nypix_global, nxpix_global, buf=None, tmp_buf=None, periodic_xcoord=True, debug=False):
        if buf is None:
            buf = cp.empty(self.preplan_size, dtype=cp.uint32)
        if tmp_buf is None:
            tmp_buf = cp.empty(self.preplan_size, dtype=cp.uint32)

        gpu_mm_pybind11.PointingPrePlan.__init__(self, xpointing_gpu, nypix_global, nxpix_global, buf, tmp_buf, periodic_xcoord, debug)


class PointingPlan(gpu_mm_pybind11.PointingPlan):
    """
    The GPU pointing 'plan' is an argument to map2tod() or tod2map().
    (See "plans and preplans" in the gpu_mm docstring.)

    Constructor arguments:
     - preplan             instance of class PointingPrePlan
     - xpointing_gpu       shape (3, nsamp) array, see "xpointing" in gpu_mm docstring
     - buf                 1-d uint8 array with length >= preplan.plan_nbytes
     - tmp_buf             1-d uint8 array with length >= preplan.plan_constructor_tmp_nbytes

    The 'buf' and 'tmp_buf' arrays are allocated from cupy, and populated by the C++ constructor.
    If these arrays are None (the default), then they'll be allocated and freed on-the-fly.
    However, you may find it more efficient to use preallocated buffers, to avoid the overhead
    of this on-the-fly allocation.
    
    IMPORTANT: the PointingPlan keeps a reference to the 'buf' array and assumes that it
    has exclusive access, whereas the 'tmp_buf' array is only used by the constructor temporarily.
    Therefore, if you're constructing multiple PointingPlans, you can use the same 'tmp_buf'
    for all of them, but 'buf' must be different.

    The PointingPlan does not keep a reference to the 'xpointing' array.

    Inherits from C++ base class (via pybind11):
        self.nsamp                 int
        self.nypix_global          int
        self.nxpix_global          int
        self.get_plan_mt()
        self.__str__()
    """
    
    def __init__(self, preplan, xpointing_gpu, buf=None, tmp_buf=None, debug=False):
        if buf is None:
            buf = cp.empty(preplan.plan_nbytes, dtype=cp.uint8)
        if tmp_buf is None:
            tmp_buf = cp.empty(preplan.plan_constructor_tmp_nbytes, dtype=cp.uint8)

        gpu_mm_pybind11.PointingPlan.__init__(self, preplan, xpointing_gpu, buf, tmp_buf, debug)

            
####################################################################################################


def read_xpointing_npzfile(filename):
    """
    Reads a shape (3,nsamp) xpointing array, in an ad hoc .npz file format
    that I defined in November 2023. I plan to phase out this file format soon!
    """
    
    print(f'Reading xpointing file {filename}')
        
    f = np.load(filename)
    xp = f['xpointing']
    assert (xp.ndim == 3) and (xp.shape[0] == 3)   # ({x,y,alpha}, ndet, ntod)
    ndet = xp.shape[1]
    ntu = xp.shape[2]            # number of time samples 'nt', unpadded
    ntp = 32 * ((ntu+31) // 32)  # number of time samples 'nt', padded to multiple of 32

    # Nuisance issue: the Nov 2023 file format uses row ordering (x,y,alpha), whereas
    # we've now switched to ordering (y,x,alpha).
    
    xpointing_cpu = np.zeros((3,ndet,ntp), dtype=xp.dtype)
    xpointing_cpu[0,:,:ntu] = xp[1,:,:]    # FIXME (0,1) index swap here is horrible
    xpointing_cpu[1,:,:ntu] = xp[0,:,:]    # FIXME (0,1) index swap here is horrible
    xpointing_cpu[2,:,:ntu] = xp[2,:,:]
    xpointing_cpu[:,:,ntu:] = xpointing_cpu[:,:,ntu-1].reshape((3,-1,1))

    return xpointing_cpu


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
        nypix_global = np.random.randint(npix_max/8, npix_max//2 + 1)
        nxpix_global = np.random.randint(nypix_global + 2, npix_max + 1)
        scan_speed = np.random.uniform(0.1, 0.5)
        total_drift = np.random.uniform(0.1*(nxpix_global-nypix_global), (nxpix_global-nypix_global)-2)
                    
        return ToyPointing(ndet, nt, nypix_global, nxpix_global, scan_speed, total_drift, noisy=noisy)


class PointingPlanTester(gpu_mm_pybind11.PointingPlanTester):
    """
    PointingPlanTester(preplan, xpointing_gpu).

    This class is only used in unit tests, so I didn't write much of a docstring :)
    For more info, see the C++/CUDA code.

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
