#ifndef _GPU_MM_PLAN_ITERATOR_HPP
#define _GPU_MM_PLAN_ITERATOR_HPP

#include "gpu_mm_internals.hpp"  // ALL_LANES
#include <ksgpu/Array.hpp>

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// Defined in src_lib/test_plan_iterator.cu
extern void test_plan_iterator(const ksgpu::Array<ulong> &plan_mt, uint nmt_per_block, int warps_per_threadblock);
extern ksgpu::Array<ulong> make_random_plan_mt(long ncells, long min_nmt_per_cell, long max_nmt_per_cell);


// Usage of plan_iterator<W> is best explained by the following pseudocode:
//
//    plan_iterator<W> iterator(plan_mt, nmt, nmt_per_block);
//
//    while (iterator.get_cell()) {
//        // Per-cell initialization (using value of iterator.icell)
//      
//        while (iterator.get_cl()) {
//            // Per-mt processing (using values of iterator.icl and iterator.icl_flagged)
//        }
//
//        __syncthreads(); shmem -> global;
//    }


// The "irregular" PlanIterator allows each threadblock to use an arbitrary [imt_start, imt_end).
// (This was temporarily useful during code development, but not sure if it has long-term usefulness.)
// You probably want 'struct plan_iterator' instead (see below).

template<int W, bool Debug>
struct plan_iterator_irregular
{
    // Initialized in constructor, constant after construction.
    const ulong *plan_mt;
    uint imt_end;          // not necessarily multiple of 32

    // Ring buffer state.
    ulong mt_rb;           // warp holds plan_mt[imt_rb:imt_rb+32).
    uint imt_rb;           // same on all threads in warp (not block), multiple of 32.

    // Per-cell state.
    uint icell;              // current cell
    bool next_cell_seen;     // have we seen start of next cell in ring buffer?
    uint next_cell_start;    // if !next_cell_seen, value is arbitrary

    // Per-mt state.
    uint imt_next;           // mt-index that will be used in _next_ call to get_cl().
    uint icl_flagged;        // cl-index from _previous_ call to get_cl(), including mflag/zflag
    uint icl;                // cl-index from _previous_ call to get_cl(), not including mflag/zflag

    // There are two helper methods:
    //   _load_mt_rb()            Called whenever 'imt_rb' changes. Updates 'mt_rb'.
    //   _init_next_cell_state()  Called whenever 'mt_rb' or 'icell' changes. Updates 'next_cell_*'.

    __device__ plan_iterator_irregular(const ulong *plan_mt_, uint imt_start, uint imt_end_)
    {
	if constexpr (Debug) {
	    assert(imt_start <= imt_end_);
	    assert(blockDim.x == 32);
	    assert(blockDim.y == W);
	}

	// Initialized in constructor, constant after construction.
	this->plan_mt = plan_mt_;
	this->imt_end = imt_end_;

	// Ring buffer state.
	this->imt_rb = imt_start & ~31U;
	this->_load_mt_rb();

	// Sentinel initializations, to ensure that get_cell() does the right thing.
	this->next_cell_start = imt_start;
	this->next_cell_seen = true;
	this->imt_next = imt_start;
    }

    
    __device__ bool get_cell()
    {
	// Assumes that caller has initialized { next_cell_*, imt_next }.
	// We either return false, or we update { icell, next_cell_*, imt_next } and return true.

	if (!next_cell_seen) {
	    if constexpr (Debug) {
		assert(imt_rb+32 >= imt_end);
		assert(imt_next >= imt_end);
	    }
	    return false;
	}

	int src_lane = next_cell_start & 31;
	uint src_icell = uint(mt_rb) & ((1U << 20) - 1);

	if constexpr (Debug) {
	    assert(imt_rb + src_lane == next_cell_start);
	    assert(imt_next >= next_cell_start);
	}

	this->icell = __shfl_sync(ALL_LANES, src_icell, src_lane);
	this->imt_next = next_cell_start + threadIdx.y;   // Reminder: threadIdx.y is the warpId
	this->_init_next_cell_state();                    // Initializes next_cell_*
	return true;
    }
    

    __device__ bool get_cl()
    {
	// Caller has initialized current-cell state, next-cell state, and 'imt_next'.
	// We either return false, or we update { imt_next, icl, icl_flagged } and return true.

	while (imt_next >= imt_rb+32) {
	    if (next_cell_seen || (imt_rb+32 >= imt_end))
		return false;   // return false without advancing ring buffer

	    // Advance ring buffer
	    imt_rb += 32;
	    _load_mt_rb();            // initializes 'mt_rb' from 'imt_rb'
	    _init_next_cell_state();  // initializes next-cell state from 'mt_rb'
	}

	uint iend = next_cell_seen ? next_cell_start : imt_end;
	
	if (imt_next >= iend)
	    return false;
	
	// If we get here, then we update 'imt_next', 'icl', 'icl_flagged, and return true.
	
	int src_lane = imt_next & 31;

	if constexpr (Debug) {
	    uint src_icell = uint(mt_rb) & ((1U << 20) - 1);
	    assert(imt_next == imt_rb + src_lane);
	    assert((src_lane != threadIdx.x) || (src_icell == icell));
	}

	uint icl_rb = uint(mt_rb >> 20);
	this->icl_flagged = __shfl_sync(ALL_LANES, icl_rb, src_lane);
	this->icl = icl_flagged & ((1U << 26) - 1);
	this->imt_next += W;
	return true;
    }


    __device__ void _load_mt_rb()
    {
	// Initializes 'mt_rb' from 'imt_rb'.

	if constexpr (Debug) {
	    assert(imt_rb < imt_end);
	    assert((imt_rb & 31) == 0);
	}
	
	// Block dims are (32,W), so threadIdx.x is the laneId.
	uint imt = min(imt_rb + threadIdx.x, imt_end - 1);	
	this->mt_rb = plan_mt[imt];
    }


    __device__ void _init_next_cell_state()
    {
	// Initializes 'next_cell_seen' and 'next_cell_start', from 'mt_rb' and 'icell'.
	
	uint icell_remote = uint(mt_rb) & ((1U << 20) - 1);
	uint flags = __ballot_sync(ALL_LANES, icell_remote > icell);

	this->next_cell_seen = (flags != 0);
	this->next_cell_start = imt_rb + __ffs(flags) - 1;
    }
};


template<int W, bool Debug>
struct plan_iterator : public plan_iterator_irregular<W,Debug>
{
    __device__ plan_iterator(const ulong *plan_mt_, uint nmt, uint nmt_per_block)
	: plan_iterator_irregular<W,Debug> (
	      plan_mt_,
	      blockIdx.x * nmt_per_block,                // imt_start
	      min(nmt, (blockIdx.x+1) * nmt_per_block))  // imt_end
    {
	if constexpr (Debug) {
	    assert(nmt_per_block > 0);
	    assert((gridDim.x-1) * nmt_per_block < nmt);
	    assert((gridDim.x) * nmt_per_block >= nmt);
	}
    }
};

    
}  // namespace gpu_mm

#endif  // _GPU_MM_PLAN_ITERATOR_HPP
