#ifndef _GPU_MM2_PLAN_ITERATOR2_HPP
#define _GPU_MM2_PLAN_ITERATOR2_HPP

#include "gpu_mm2_internals.hpp"  // ALL_LANES

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// Defined in src_lib/PointingPlanIterator.cu
extern void test_plan_iterator2(const gputils::Array<ulong> &plan_mt, uint nmt_per_block, int warps_per_threadblock);


// Usage of PlanIterator2<W> is best explained by the following pseudocode:
//
//    PlanIterator2<W> iterator(plan_mt, nmt, nmt_per_block);
//
//    while (iterator.get_cell()) {
//        // Per-cell initialization (using value of iterator.icell)
//      
//        while (iterator.get_cl()) {
//            // Per-mt processing (using values of iterator.icl and iterator.sid)
//        }
//
//        __syncthreads(); shmem -> global;
//    }


template<int W, bool Debug>
struct PlanIterator2
{
    // Initialized in constructor, constant after construction.
    const ulong *plan_mt;
    uint imt_end;          // not necessarily multiple of 32
    // int nbatch;

    // Ring buffer state.
    ulong mt_rb;           // warp holds plan_mt[imt_rb:imt_rb+32).
    uint imt_rb;           // same on all threads in warp (not block), multiple of 32.

    // Per-cell state.
    uint icell;              // current cell
    bool next_cell_seen;     // have we seen start of next cell in ring buffer?
    uint next_cell_start;    // if !next_cell_seen, value is arbitrary

    // Per-mt state.
    uint imt_next;           // mt-index that will be used in _next_ call to get_cl().
    uint icl;                // cl-index from _previous_ call to get_cl().
    uint sid;                // secondary ID (in {0,1,2}) from _previous_ call to get_cl().

    // There are two helper methods:
    //   _load_mt_rb()            Called whenever 'imt_rb' changes. Updates 'mt_rb'.
    //   _init_next_cell_state()  Called whenever 'mt_rb' or 'icell' changes. Updates 'next_cell_*'.

    __device__ PlanIterator2(const ulong *plan_mt_, int nmt, int nmt_per_block)
    {
	if constexpr (Debug) {
	    assert(nmt_per_block > 0);
	    assert((nmt_per_block & 31) == 0);	
	    assert((gridDim.x-1) * nmt_per_block < nmt);
	    assert((gridDim.x) * nmt_per_block >= nmt);
	    assert(blockDim.x == 32);
	    assert(blockDim.y == W);
	    // assert(nbatch_ > 0);
	    // assert(nbatch_ <= 32);
	    // assert((nbatch_ & (nbatch_-1)) == 0);
	}

	uint b = blockIdx.x;

	// Initialized in constructor, constant after construction.
	this->plan_mt = plan_mt_;
	this->imt_end = min(nmt, (b+1)*nmt_per_block);
	// this->nbatch = nbatch_;

	// Ring buffer state.
	this->imt_rb = b * nmt_per_block;
	this->_load_mt_rb();

	// Sentinel initializations, to ensure that get_cell() does the right thing.
	this->next_cell_start = imt_rb;
	this->next_cell_seen = true;
	this->imt_next = imt_rb;
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
	// We either return false, or we update { imt_next, icl, sid } and return true.

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
	
	// If we get here, then we update 'imt_next', 'icl', 'sid', and return true.
	
	int src_lane = imt_next & 31;

	if constexpr (Debug) {
	    uint src_icell = uint(mt_rb) & ((1U << 20) - 1);
	    assert(imt_next == imt_rb + src_lane);
	    assert((src_lane != threadIdx.x) || (src_icell == icell));
	}
	
	uint mt0 = uint(mt_rb >> 20);
	mt0 = __shfl_sync(ALL_LANES, mt0, src_lane);
	
	this->icl = mt0 & ((1U << 26) - 1);
	this->sid = mt0 >> 26;
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


}  // namespace gpu_mm2

#endif  // _GPU_MM2_PLAN_ITERATOR2_HPP
