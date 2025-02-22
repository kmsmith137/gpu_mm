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

   (This type of memory layout is used in "dynamic" maps, see next section.)

==========================
"DYNAMIC" PIXEL-SPACE MAPS
==========================

A DynamicMap is a LocalMap which starts empty (i.e. with zero cells), and adds
cells as needed, whenever tod2map() is called. After all TODs have been processed,
you call DynamicMap.finalize() to convert the DynamicMap to a LocalMap.

DynamicMaps always use a memory layout of the form

   float32 local_map[ncells][3][64][64];

where 'ncells' increases dynamically when tod2map() is called. The cell ordering
is arbitrary, and can be inspected after calling DynamicMap.finalize(). For details
on how this works (and for more info on DynamicMaps in general), see the DynamicMap
docstring.

(Context: recall that a maximum likelihood map-maker runs in two stages: a 
"dirty map" stage, followed by a CG stage. The first stage could use a DynamicMap 
to determine which map cells are needed on each MPI task. At the end of the first
stage, DynamicMap.finalize() can called, to "freeze in" the pixelization. The CG
stage would use only LocalMaps -- no DynamicMaps.)

=================
MPI PIXELIZATIONS
=================

The MpiPixelization class distributes map cells across MPI tasks. It defines three 
pixelizations on each MPI task. (These are instances of class LocalPixelization.)

  - self.toplevel_pixelization: assigns each map cell in the survey footprint to a
     unique MPI task. The toplevel pixelization is used to represent maps in the
     high-level CG solver.
        
  - self.working_pixelization: defines a many-to-one mapping between map cells and
     MPI tasks, such that each MPI task is assigned all cells which are "hit" by any
     of the task's TODs. The working pixelization is used in tod2map() and map2tod().
            
  - self.root_pixelization: assigns all map cells to the root task.

The MpiPixelization class also defines methods for converting maps between pixelizations
(broadcast(), reduce(), gather()), which are written with the CG solver in mind. For
more info, see the MpiPixelization docstring.


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


import os
import time
import ctypes
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

    if isinstance(local_map, DynamicMap):
        if not isinstance(plan, PointingPlan):
            raise RuntimeError("tod2map(): if output is a DynamicMap, then the 'plan' argument"
                               + f" must be a PointingPlan (got: {plan})'")
        
        if partial_pixelization != local_map.have_cell_mask:
            raise RuntimeError(f"tod2map(): {partial_pixelization=} was specified,"
                               + f" but DynamicMap.have_cell_mask={local_map.have_cell_mask}"
                               + " (expected these to be equal)")
        
        local_map.expand(plan)

        gpu_mm_pybind11.planned_tod2map(local_map._unstable_arr, tod, xpointing,
                                        local_map._unstable_pixelization, plan,
                                        partial_pixelization, debug)
        
        return
        
    if not isinstance(local_map, LocalMap):
        raise RuntimeError(f"tod2map(): Bad 'local_map' argument to tod2map(): expected LocalMap or DynamicMap, got: {local_map}")

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
      ystride        (int)
      polstride      (int)
      nycells        (int)
      nxcells        (int)
      npix           (int)
    """
    
    def __init__(self, nypix_global, nxpix_global, cell_offsets, ystride, polstride, periodic_xcoord = True):
        self.cell_offsets_cpu = cp.asnumpy(cell_offsets)   # numpy array
        self.cell_offsets_gpu = cp.asarray(cell_offsets)   # cupy array
        
        # Note: most of the members mentioned in the docstring are inherited from pybind11.
        # (Specifically: nypix_global, nxpix_global, periodic_xcoord, ystride, polstride, nycells, nxcells, npix).
        gpu_mm_pybind11.LocalPixelization.__init__(self, nypix_global, nxpix_global, self.cell_offsets_cpu, self.cell_offsets_gpu, ystride, polstride, periodic_xcoord)
    
    def is_simple_rectangle(self):
        """
        The simplest case of a local pixelization is a 3-d contiguous array of shape
        (3, 64*nycells, 64*nxcells). This function returns True if the LocalPixelization
        is of this simple type.
        """
        rect_offsets = self._make_rectangular_cell_offsets(self.nycells, self.nxcells)
        return np.array_equal(self.cell_offsets_cpu, rect_offsets)


    def __eq__(self, x):
        assert isinstance(x, LocalPixelization)
        
        if self is x:
            return True
        
        # nypix_global, nxpix_global, periodic_xcoord, ystride, polstride, nycells, nxcells, npix).
        if (self.nypix_global, self.nxpix_global) != (x.nypix_global, x.nxpix_global):
            return False
        if (self.periodic_xcoord, self.ystride, self.polstride) != (x.periodic_xcoord, x.ystride, x.polstride):
            return False
        if (self.nycells, self.nxcells, self.npix) != (x.nycells, x.nxcells, x.npix):
            return False

        return np.all(np.maximum(self.cell_offsets_cpu,-1) == np.maximum(x.cell_offsets_cpu,-1))
    

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
        assert isinstance(arr, np.ndarray) or isinstance(arr, cp.ndarray)

        # Currently, no check on array size! (might rethink this)
        # assert arr.size == (3*pixelization.npix,)

        # Currently, we only support float32, but adding support for float64 should be straightforward.
        assert arr.dtype == np.float32
        assert arr.flags['C_CONTIGUOUS']

        self.pixelization = pixelization
        self.nypix_global = pixelization.nypix_global
        self.nxpix_global = pixelization.nxpix_global
        self.periodic_xcoord = pixelization.periodic_xcoord
        self.on_gpu = isinstance(arr, cp.ndarray)
        self.npix = pixelization.npix  # note that array size is (3*npix), and number of cells is npix/64**2
        self.dtype = arr.dtype
        self.arr = arr


    def to_global(self):
        """Converts the LocalMap to a "global" map, and and returns the global map.

        The global map is represented as a numpy array of shape (3, nypix_global, nxpix_global).
        Regions of the global map which are not covered by the LocalMap are zeroed.
        
        Note: we currently ignore pixels in the LocalMap with (xcoord >= nxpix_global).
        Another option would be to "wrap" such pixels (if periodic_xcoord=True).
        If this feature would be useful, I can implement a 'wrap' optional argument.
        """

        src = cp.asnumpy(self.arr)
        dst = np.zeros((3, self.nypix_global, self.nxpix_global), dtype=self.dtype)

        gpu_mm_pybind11.local_map_to_global(self.pixelization, dst, src)
        return dst


####################################################################################################


class DynamicMap:
    def __init__(self, nypix_global, nxpix_global, dtype, cell_mask=None, periodic_xcoord=True, initial_ncells=1024):
        """
        A DynamicMap is a LocalMap which starts empty (i.e. with zero cells), and adds
        cells as needed, whenever tod2map() is called. After all TODs have been processed,
        you call DynamicMap.finalize() to convert the DynamicMap to a LocalMap.

        DynamicMaps always use a memory layout of the form
        
            float32 local_map[ncells][3][64][64];

        where 'ncells' increases dynamically when tod2map() is called. The cell ordering is
        arbitrary, and can be inspected after calling DynamicMap.finalize() -- see below.

        The meaning of the 'cell_mask' argument needs some explanation. This is intended for
        a situation where you're running the map maker on a "target" subset of the sky, and 
        want to ignore pixels outside the subset. In this case, 'cell_mask' should be a boolean 
        array of shape (nycells, nxcells), which is True in the "target" region. Then the
        DynamicMap() will only add cells which are in the target region.

        Here is some example code which explains typical usage of DynamicMap:

            dmap = DynamicMap(nypix_global, nxpix_global, dtype)

            for (tod, xpointing) in ...:
                # Here, 'tod' is a cupy array with shape (ndet,nt) or (nt,)
                # and 'xpointing' is a cupy array with shape (3,ndet,nt) or (3,nt).

                preplan = PointingPrePlan(xpointing, nypix_global, nxpix_global)
                plan = PointingPlan(preplan, xpointing)
                tod2map(dmap, tod, xpointing, plan)  # dynamically expands 'dmap'
        
            # Finalize (converts DynamicMap -> LocalMap).
            local_map = dmap.finalize()          # instance of class LocalMap 
            local_pix = local_map.pixelization   # instance of class LocalPixelization
            local_arr = local_map.arr            # cupy array, shape (ncells,3,64,64).
        
            # The memory layout is: float32 local_map[ncells][3][64][64].
            # This is encoded (including cell ordering) by the 2-d 'cell_offsets_cpu' array 
            # in the LocalPixelization. If the i-th cell has indices (iycell, ixcell),
            # then cell_offsets_cpu[iycell,ixcell] will be (3*64*64*i). Cells which are
            # not "touched" are represented by negative entries in the cell_offsets array.

        Some limitations of the current code (easy to change -- LMK if you need this):

           - DynamicMap supports tod2map() but not map2tod().
           - Can only inspect the cell ordering by calling finalize() at the end.
           - Cell ordering is nondetermininstic (!)
        """

        assert 0 < nxpix_global <= 64*1024
        assert 0 < nypix_global <= 64*1024
        assert dtype == cp.float32   # placeholder for eventually supporting float32 + float64
        assert initial_ncells > 0
        
        self.nypix_global = nypix_global
        self.nxpix_global = nxpix_global
        self.periodic_xcoord = periodic_xcoord
        self.have_cell_mask = (cell_mask is not None)
        self.ncells_allocated = initial_ncells
        self.reallocation_factor = 1.5
        self.kernel_nblocks = 1024
        self.dtype = dtype

        # Initialize 'cell_offsets' array (see LocalPixelization docstring).
        # Array elements are (-1) in "targeted" cells, or (-2) in "untargeted" cells.
        # (This convention is assumed by the gpu_mm_pybind11.expand_dynamic_map() cuda kernel.)
        
        if cell_mask is None:
            nycells = (nypix_global + 63) // 64
            nxcells = (nxpix_global + 63) // 64
            cell_offsets = np.full((nycells,nxcells), -1, dtype=int)
        else:
            cell_mask = np.asarray(cell_mask)
            assert cell_mask.ndim == 2
            assert cell_mask.dtype == bool
            cell_offsets = np.where(cell_mask, -1, -2)

        # Note: member names beginning with underscores are intended to indicate hidden danger :)
        #  - self._unstable_pixelization: A LocalPixelization where CPU/GPU state is inconsistent (!)
        #  - self._unstable_arr: This array is reallocated/resized between calls to tod2map().
        #  - self._padded_arr: This array is reallocated/resized between calls to tod2map().
        
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

        # The cuda kernel needs an auxiliary one-element array, to track the current number of cells.
        # (We allocate a 32-element cache line and truncate.)
        self.global_ncells = cp.zeros(32, dtype=cp.uint32)
        self.global_ncells = self.global_ncells[:1]
        
        self.ncells_curr = 0
        self.pixelization_is_finalized = False
        self.map_is_finalized = False
        

    def expand(self, plan):
        """Adds cells to the pixelization to cover a TOD (or more precisely, the PointingPlan for the TOD).
        The map will be zero-padded in the new cells (and reallocated if necessary).

        This function is intended to be called by tod2map(), when tod2map() is called with a DynamicMap
        as its first argument, but may be useful on its own.
        """

        if self.pixelization_is_finalized:
            raise RuntimeError("Can't currently call tod2map() [or DynamicMap.expand()]"
                               + " after MpiPixelization.__init__() [or DynamicMap.permute()].")
        
        assert isinstance(plan, PointingPlan)
        assert plan.nypix_global == self.nypix_global
        assert plan.nxpix_global == self.nxpix_global
        assert plan.periodic_xcoord == self.periodic_xcoord

        self.ncells_curr = gpu_mm_pybind11.expand_dynamic_map2(self.global_ncells, self._unstable_pixelization, plan)

        if self.ncells_curr > self.ncells_allocated:
            self.ncells_allocated = int(self.reallocation_factor * self.ncells_allocated)
            self.ncells_allocated = max(self.ncells_allocated, self.ncells_curr)
            self._padded_arr = cp.zeros(self.ncells_allocated * 3*64*64, dtype=self.dtype)
            self._padded_arr[:len(self._unstable_arr)] = self._unstable_arr[:]

        self._unstable_pixelization.npix = self.ncells_curr * (64*64)
        self._unstable_arr = self._padded_arr[:(self.ncells_curr * 3*64*64)]

    
    def permute(self, iycells, ixcells):
        """Permutes the pixelization's cell ordering.

        Intended as a helper for MpiPixelization.__init__(), to put the cells in a more MPI-friendly
        ordering, but may be useful on its own.

        The 'iycells' and 'ixcells' args are integer-valued arrays of length (self.ncells_curr),
        that give the (y,x) coordinates in the desired (permuted) cell ordering.
        
        Currently, permute() finalizes the pixelization -- this isn't logically necessary, but makes 
        sense if called through MpiPixelization.__init__().
        """

        if self.pixelization_is_finalized:
            raise RuntimeError("Can't currently call MpiPixelization.__init__() [or DynamicMap.permute()]"
                               + " after DynamicMap.finalize() [or MpiPixelization.__init__(), DynamicMap.permute()]")

        self._unstable_pixelization.copy_gpu_offsets_to_cpu()
        assert self._unstable_pixelization.npix == (self.ncells_curr * 64 * 64)   # gets updated by copy_gpu_offsets_to_cpu()
        new_offsets = self._unstable_pixelization.cell_offsets_cpu                # cell offsets will be overwritten in-place
        old_offsets = np.copy(new_offsets)                                        # keep temporary copy of old offsets
        
        # Lots of argument checking
        assert isinstance(iycells, np.ndarray) and isinstance(ixcells, np.ndarray)   # must be on CPU
        assert iycells.shape == ixcells.shape == (self.ncells_curr,)
        assert iycells.dtype == ixcells.dtype == int
        assert 0 <= np.min(iycells) <= np.max(iycells) < old_offsets.shape[0]
        assert 0 <= np.min(ixcells) <= np.max(ixcells) < old_offsets.shape[1]

        # Update the pixelization.
        new_offsets[iycells,ixcells] = np.arange(self.ncells_curr) * (3*64*64)

        if not np.all((old_offsets >= 0) == (new_offsets >= 0)):
            raise RuntimeError('DynamicMap.permute(): specified (iycells,ixcells) do not match footprint of current pixelization')

        # Sync pixelization to GPU and finalize.
        self.pixelization_is_finalized = True
        self._unstable_pixelization.copy_cpu_offsets_to_gpu()
        assert self._unstable_pixelization.npix == (self.ncells_curr * 64 * 64)   # gets updated by copy_cpu_offsets_to_gpu()

        # Remaining code permutes the map array.
        # FIXME code below does one call to cudaMemcpyAsync() per map cell!
        # At some point I'll write a cuda kernel (should be very straightforward)

        old_arr = self._unstable_arr
        self._padded_arr = self._unstable_arr = cp.empty(self.ncells_curr * 3*64*64, dtype=self.dtype)
        self.ncells_allocated = self.ncells_curr

        # 1-d array, maps (new cell index) -> (old cell index)*3*64*64
        src_offsets = old_offsets[iycells, ixcells]

        n = 3*64*64
        for i,j in enumerate(src_offsets):
            self._unstable_arr[i*n:(i*n+n)] = old_arr[j:(j+n)]
        

    def finalize(self):
        """Called at the end of the TOD loop. Returns a LocalMap."""

        if self.map_is_finalized:
            raise RuntimeError('Double call to DynamicMap.finalize() [not currently allowed]')

        if not self.pixelization_is_finalized:
            self._unstable_pixelization.copy_gpu_offsets_to_cpu()
            self.pixelization_is_finalized = True

        if self.ncells_allocated > self.ncells_curr:
            self._unstable_arr = cp.copy(self._unstable_arr)  # shrink overallocated map

        ret = LocalMap(self._unstable_pixelization, self._unstable_arr)
        
        assert self._unstable_pixelization.npix == (self.ncells_curr * 64 * 64)        
        self._padded_arr = self._unstable_arr = None    # don't hold references
        self.map_is_finalized = True
        
        return ret


    def randomly_permute(self):
        """Only called during testing."""
        
        self._unstable_pixelization.copy_gpu_offsets_to_cpu()
        iycells, ixcells = np.where(self._unstable_pixelization.cell_offsets_cpu >= 0)

        perm = np.random.permutation(self.ncells_curr)
        self.permute(iycells[perm], ixcells[perm])


####################################################################################################


class MpiPixelization:
    def __init__(self, dmap, noisy=True, npy_filename=None, comm=None):
        """Distributes map cells across MPI tasks.

        The main purpose of this class is to define three pixelizations on each MPI task.
        (These are instances of class LocalPixelization.)

          - self.toplevel_pixelization: assigns each map cell in the survey footprint to a
             unique MPI task. The toplevel pixelization is used to represent maps in the
             high-level CG solver.
        
          - self.working_pixelization: defines a many-to-one mapping between map cells and
             MPI tasks, such that each MPI task is assigned all cells which are "hit" by any
             of the task's TODs. The working pixelization is used in tod2map() and map2tod().
            
          - self.root_pixelization: puts all map cells on the root task. Currently only used
             in gather().

        And the following three methods, which are used to convert between pixelizations 
        in the CG solver:

          - broadcast(toplevel_map) -> (working_map).
            Sends each map cell to all MPI tasks which "hit" the cell.
            Called at the beginning of each CG iteration, before any map2tod() calls.

          - reduce(working_map) -> (toplevel_map).
            For each map cell, sum the cell contents over all MPI tasks which "hit" the cell.
            Called at the end of each CG iteration, to accumulate all tod2map() outputs.
        
          - gather(toplevel_map) -> (root_map).
            Called at the end of the CG solver, to gather the map into a single array,
            in order to write to disk. (You may also want to call LocalMap.to_global(),
            see the gather() docstring)

        Constructor arguments:

         - dmap: a DynamicMap on each MPI task. This tells the MpiPixelization constructor
            which map cells are "hit" by which MPI task(s).
        
         - noisy: if True, then some diagnostic info is printed (on MPI task 0).

         - npy_filename: if specified, then a .npy file will be saved (on MPI task 0)
            which can be used to reconstruct the MpiPixelization. This is intended in
            order to save examples, for experimenting with load-balancing algorithms.

         - comm: an MPI communicator (if unspecified, then MPI.COMM_WORLD is used)
        
        Note 1: The MpiPixelization constructor permutes the cell ordering in 'dmap',
        in order to make the orering more MPI-friendly! I think this should always be
        okay, since the cell ordering is arbitrary anyway.

        Note 2: The MpiPixelization constructor finalizes the 'dmap' pixelization, but does
        not finalize the map itself. (This means that you can still call dmap.finalize(),
        but you can't further enlarge the pixelization with tod2map(dmap, ...).

        Note 3: I'm currently using a simple load-balancing algorithm that gives decent results
        in cases where I've tried it, but I might experiment with fancier algorithms later. 
        For more info, see the CellAssignment docstring.
        """

        if comm is None:
            import mpi4py.MPI
            comm = mpi4py.MPI.COMM_WORLD

        t0 = time.time()
        assert isinstance(dmap, DynamicMap)
        assert not dmap.pixelization_is_finalized
        
        # FIXME temporary kludge that will go away when I do some code cleanup.
        dmap._unstable_pixelization.copy_gpu_offsets_to_cpu()
        cell_offsets = dmap._unstable_pixelization.cell_offsets_cpu

        self.comm = comm
        self.ntasks = comm.Get_size()
        self.my_rank = comm.Get_rank()
        self.nypix_global = dmap.nypix_global
        self.nxpix_global = dmap.nxpix_global
        self.periodic_xcoord = dmap.periodic_xcoord
        self.nycells, self.nxcells = cell_offsets.shape
        self.mpi = mpi4py.MPI

        # Make sure these 5 quantities are in sync across all MPI tasks.
        check = [ self.nypix_global, self.nxpix_global, self.periodic_xcoord, self.nycells, self.nxcells ]
        if not self._is_equal_on_all_tasks(check):
            raise RuntimeError(f'MpiPixelization: expected (nypix_global, nxpix_global, periodic_xcoord, nycells, nxcells) to be equal on all tasks')
        
        # my_hits = 2-d boolean array of shape (nycells, nxcells), indicates which cells are "hit" by local task.
        my_hits = (cell_offsets >= 0)
        
        # all_hits = 3-d boolean array of shape (ntasks, nycells, nxcells), indicates which cells are "hit" by each task.
        all_hits = np.zeros((self.ntasks,self.nycells,self.nxcells), dtype=bool)
        comm.Allgather((my_hits, self.nycells*self.nxcells), (all_hits, self.nycells*self.nxcells))

        if (npy_filename is not None) and (self.my_rank == 0):
            print(f'Writing {npy_filename}')
            np.save(npy_filename, all_hits)
            
        # Toplevel assignment of cells to tasks is factored into separate CellAssignment class, see below.
        self.cell_assignment = CellAssignment(all_hits)
        self.tl_ncells = self.cell_assignment.toplevel_ncells[self.my_rank]
        self.aux_ncells = self.cell_assignment.aux_ncells[self.my_rank]
        self.wbuf_ncells = self.cell_assignment.working_ncells[self.my_rank]

        # Paranoid code check: make sure that the cell assignment is the same on all MPI tasks.
        assert self._is_equal_on_all_tasks(self.cell_assignment.cell_owners)
        assert dmap._unstable_pixelization.npix == self.wbuf_ncells * 64*64

        # 1-d arrays of length (tl_ncells), containing (y,x) cell indices owned by local ("my") task.
        tl_y, tl_x = np.where(self.cell_assignment.cell_owners == self.my_rank)
        assert tl_y.shape == tl_x.shape == (self.tl_ncells,)

        # The "auxiliary" buffer is an intermediate buffer in MPI_Alltoallv().
        # It is logically divided into one sub-buffer for each MPI task.
        # The r-th sub-buffer is indexed by cells owned by current task, which are also "hit" by task r.
        # In broadcast() and reduce() [see below], the aux buffer is the Alltoallv() send/recv buffer, respectively.

        # 2-d boolean array of shape (ntasks, toplevel_ncells).
        aux_hits = all_hits[:, tl_y, tl_x]
        assert np.sum(aux_hits) == self.aux_ncells

        # The aux_index_map maps [0:aux_ncells) -> [0:toplevel_ncells).
        self.aux_counts = np.empty(self.ntasks, dtype=int)
        self.aux_displs = np.empty(self.ntasks, dtype=int)
        self.aux_index_map = np.empty(self.aux_ncells, dtype=int)

        pos = 0
        for r in range(self.ntasks):
            ix = np.where(aux_hits[r,:])[0]    # 1-d array of indices in [0:tl_ncells).
            n = len(ix)
            self.aux_counts[r] = n * (3*64*64)
            self.aux_displs[r] = pos * (3*64*64)
            self.aux_index_map[pos:(pos+n)] = ix
            pos += n
        
        assert pos == self.aux_ncells

        # The "working" buffer (wbuf) is a LocalMap, as viewed by MPI_Alltoallv().
        # We'll permute the LocalMap tiling to a more MPI-friendly layout (see call to dbuf.permute() below).
        # With this permutation, the wbuf consists of one sub-buffer for each MPI task.
        # The r-th sub-buffer is indexed by cells owned by task r, which are also touched by the local task.
        # In broadcast() and reduce() [see below], the work buffer is the Alltoallv() recv/send buffer, respectively.

        # The (wbuf_y, wbuf_x) arrays map [0:working_ncells) -> [0:nycells) or [0:nxcells).
        self.wbuf_counts = np.empty(self.ntasks, dtype=int)
        self.wbuf_displs = np.empty(self.ntasks, dtype=int)
        self.wbuf_y = np.empty(self.wbuf_ncells, dtype=int)
        self.wbuf_x = np.empty(self.wbuf_ncells, dtype=int)

        pos = 0
        for r in range(self.ntasks):
            # mask = 2-d boolean array of shape (nycells, nxcells), indicates which cells are in r-th sub-buffer.
            mask = np.logical_and(my_hits, self.cell_assignment.cell_owners == r) 
            y, x = np.where(mask)   # xy-indices of cells in r-th sub-buffer.
            n = len(y)
            self.wbuf_counts[r] = n * (3*64*64)
            self.wbuf_displs[r] = pos * (3*64*64)
            self.wbuf_y[pos:(pos+n)] = y
            self.wbuf_x[pos:(pos+n)] = x
            pos += n

        assert pos == self.wbuf_ncells

        # The "root" buffer consists of all cells that were "hit" by any task.
        # The root pixelization is used gather() [see below].

        self.root_ncells = np.sum(self.cell_assignment.toplevel_ncells)
        self.root_counts = self.cell_assignment.toplevel_ncells * (3*64*64)
        self.root_displs = np.empty(self.ntasks, dtype=int)
        self.root_y = np.empty(self.root_ncells, dtype=int)
        self.root_x = np.empty(self.root_ncells, dtype=int)

        pos = 0
        for r in range(self.ntasks):
            n = self.cell_assignment.toplevel_ncells[r]
            y, x = np.where(self.cell_assignment.cell_owners == r)
            assert y.shape == x.shape == (n,)
            self.root_displs[r] = pos * (3*64*64)
            self.root_y[pos:(pos+n)] = y
            self.root_x[pos:(pos+n)] = x
            pos += n

        assert pos == self.root_ncells
        
        # Construct self.toplevel_pixelization (instance of class LocalPixelization).
        
        tl_cell_offsets = np.full((self.nycells,self.nxcells), -1)
        tl_cell_offsets[tl_y,tl_x] = np.arange(self.tl_ncells) * (3*64*64)

        self.toplevel_pixelization = LocalPixelization(
            nypix_global = dmap.nypix_global,
            nxpix_global = dmap.nxpix_global,
            cell_offsets = tl_cell_offsets,
            ystride = 64,
            polstride = 64*64,
            periodic_xcoord = dmap.periodic_xcoord
        )

        # Construct self.working_pixelization (instance of class LocalPixelization).
        # Note call to DynamicPixelization.permute() here!

        dmap.permute(self.wbuf_y, self.wbuf_x)
        self.working_pixelization = dmap._unstable_pixelization

        # Construct self.root_pixelization (instance of class LocalPixelization)

        root_cell_offsets = np.full((self.nycells,self.nxcells), -1)
        root_cell_offsets[self.root_y, self.root_x] = np.arange(self.root_ncells) * (3*64*64)

        self.root_pixelization = LocalPixelization(
            nypix_global = dmap.nypix_global,
            nxpix_global = dmap.nxpix_global,
            cell_offsets = root_cell_offsets,
            ystride = 64,
            polstride = 64*64,
            periodic_xcoord = dmap.periodic_xcoord
        )

        if noisy and (self.my_rank == 0):
            print(f'MpiPixelization: creation time {(time.time()-t0):.03f} seconds, load balancing stats:'
                  + f' lb_toplevel={(self.cell_assignment.lb_toplevel):.03f},'
                  + f' lb_working={(self.cell_assignment.lb_working):.03f},'
                  + f' lb_aux={(self.cell_assignment.lb_aux):.03f}')
        
    
    def broadcast(self, toplevel_map):
        """broadcast(toplevel_map) -> (working_map).

        Sends each map cell to all MPI tasks which "hit" the cell.
        Called at the beginning of each CG iteration, before any map2tod() calls.
        """

        assert isinstance(toplevel_map, LocalMap)
        assert toplevel_map.pixelization == self.toplevel_pixelization

        src_buf = cp.asnumpy(toplevel_map.arr)  # copies GPU -> CPU if necessary
        src_buf = np.reshape(src_buf, self.tl_ncells * 3*64*64)
        aux_buf = np.empty(self.aux_ncells * 3*64*64, dtype = src_buf.dtype)
        dst_buf = np.empty(self.wbuf_ncells * 3*64*64, dtype = src_buf.dtype)

        if self.aux_ncells > 0:
            gpu_mm_pybind11.cell_broadcast(aux_buf, src_buf, self.aux_index_map)
        
        self.comm.Alltoallv((aux_buf, (self.aux_counts,self.aux_displs)),
                            (dst_buf, (self.wbuf_counts,self.wbuf_displs)))
        
        return LocalMap(self.working_pixelization, dst_buf.reshape(-1))
        

    def reduce(self, working_map):
        """reduce(working_map) -> (toplevel_map).

        For each map cell, sum the cell contents over all MPI tasks which "hit" the cell.
        Called at the end of each CG iteration, to accumulate all tod2map() outputs.
        """
        
        assert isinstance(working_map, LocalMap)
        assert working_map.pixelization == self.working_pixelization

        src_buf = cp.asnumpy(working_map.arr)  # copies GPU -> CPU if necessary
        src_buf = np.reshape(src_buf, self.wbuf_ncells * 3*64*64)
        aux_buf = np.empty(self.aux_ncells * 3*64*64, dtype = src_buf.dtype)
        dst_buf = np.empty(self.tl_ncells * 3*64*64, dtype = src_buf.dtype)   # empty() is okay here
        
        self.comm.Alltoallv((src_buf, (self.wbuf_counts, self.wbuf_displs)),
                            (aux_buf, (self.aux_counts, self.aux_displs)))

        if self.aux_ncells > 0:
            gpu_mm_pybind11.cell_reduce(dst_buf, aux_buf, self.aux_index_map)

        return LocalMap(self.toplevel_pixelization, dst_buf.reshape(-1))


    def gather(self, toplevel_map, root=0):
        """gather(toplevel_map) -> (root_map).

        Called at the end of the CG solver, to gather the map into a single array,
        in order to write to disk. 
        
        Returns a LocalMap (in the root_pixelization) on the root MPI task, and
        returns None on non-root tasks. This LocalMap covers the part of the sky
        which is "hit" by any TOD, and has its own pixelization to keep track of
        which cells are included. 

        If you want to further convert to a "global" map, which is just a 
        shape (3, nypix_global, nxpix_global) numpy array, then you can call
        LocalMap.to_global() on the output of gather().
        """

        assert isinstance(toplevel_map, LocalMap)
        assert toplevel_map.pixelization == self.toplevel_pixelization
        assert 0 <= root < self.ntasks
        
        src_buf = cp.asnumpy(toplevel_map.arr)  # copies GPU -> CPU if necessary
        src_buf = np.reshape(src_buf, (self.tl_ncells,3*64*64))
        dst_arg = dst_map = None

        if self.my_rank == root:
            dst_buf = np.empty((self.root_ncells,3*64*64), dtype = src_buf.dtype)
            dst_arg = (dst_buf, (self.root_counts, self.root_displs))
            dst_map = LocalMap(self.root_pixelization, dst_buf.reshape(-1))

        self.comm.Gatherv((src_buf, src_buf.size), dst_arg, root=root)
        return dst_map
        
    
    def _allreduce(self, arr):
        arr = np.asarray(arr)
        assert arr.data.c_contiguous
        ret = np.empty_like(arr)
        self.comm.Allreduce((arr,arr.size), (ret,ret.size))
        return ret

    def _allgather(self, arr):
        arr = np.asarray(arr)
        assert arr.data.c_contiguous
        ret = np.empty((self.ntasks,) + arr.shape, dtype=arr.dtype)
        self.comm.Allgather((arr,arr.size), (ret,arr.size))
        return ret


    def _is_equal_on_all_tasks(self, arr):
        arr = np.asarray(arr)
        big_arr = self._allgather(arr)
        return np.all(arr == big_arr)

    
class CellAssignment:
    def __init__(self, all_hits):
        """Helper class for MpiPixelization: computes the assignment of cells to tasks.

        Constructor argument 'all_hits' is a 3-d boolean array of shape (ntasks, nycells, nxcells) 
        indicating which cells are "hit" by each task.

        The constructor initializes the following members:

           self.cell_owners      shape (nycells,nxcells) array containing MPI rank of each cell's owner (or -1)
           self.toplevel_ncells  length (ntasks) array, contains number of cells owned by each task
           self.working_ncells   length (ntasks) array, contains number of cells "hit" by each task
           self.aux_ncells       length (ntasks) array, contains number of cells in MPI "aux" buffer
           self.lb_toplevel      load-balancing summary statistic: max(toplevel_ncells) / mean(toplevel_ncells)
           self.lb_working       load-balancing summary statistic: max(working_ncells) / mean(working_ncells)
           self.lb_aux           load-balancing summary statistic: max(toplevel_ncells) / mean(toplevel_ncells)

        Note that this function doesn't call any MPI functions -- it just computes the cell assignment
        redundantly on each MPI task. The assignment is guaranteed to be the same on all tasks (this gets
        checked by an assert in MpiPixelization.__init__()).

        I'm currently using the following simple cell assignment algorithm:

          - Define the "weight" of a cell to be the number of tasks which hit the cell.

          - Sort cells by weight.

          - For each cell, consider only tasks which hit the cell, and choose the task which
            currently has the smallest value of the cost function:

               Cost(task) = sum_{cells assigned to task} (5 + weight(cell))

        This algorithm seems to work well in toy examples -- let's see what happens in SO!
        """

        assert all_hits.ndim == 3
        assert all_hits.dtype == bool

        w_2d = np.sum(all_hits, axis=0)  # 2-d weights array, shape (nycells, nxcells)        
        y_1d, x_1d = np.where(w_2d > 0)  # 1-d ycell, xcell arrays
        w_1d = w_2d[y_1d, x_1d]          # 1-d weights array

        self.ntasks = all_hits.shape[0]
        self.cell_owners = np.full_like(w_2d, -1, dtype=int)
        curr_costs = np.zeros(self.ntasks, dtype=int)

        # Sort cells by weight.
        for w,y,x in sorted(zip(w_1d, y_1d, x_1d)):
            # The 10**10 term ensures that we only assign the cell to a task which hits it.
            owner = np.argmin(curr_costs - (10**10 * all_hits[:,y,x]))
            self.cell_owners[y,x] = owner
            curr_costs[owner] += (5+w)

        # The rest of this function just computes load-balancing summary stats.

        owners_1d = self.cell_owners[y_1d, x_1d]

        self.toplevel_ncells = np.bincount(owners_1d, minlength=self.ntasks)
        self.working_ncells = np.sum(all_hits.reshape((self.ntasks,-1)), axis=1)
        self.aux_ncells = np.bincount(owners_1d, w_1d, minlength=self.ntasks)
        self.aux_ncells = np.array(self.aux_ncells, dtype=int)  # float -> int

        self.lb_toplevel = np.max(self.toplevel_ncells) / np.mean(self.toplevel_ncells)
        self.lb_working = np.max(self.working_ncells) / np.mean(self.working_ncells)
        self.lb_aux = np.max(self.aux_ncells) / np.mean(self.aux_ncells)
        

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

_libgpu_mm = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "lib", "libgpu_mm.so"))

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
