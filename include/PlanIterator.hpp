#ifndef _GPU_MM2_PLAN_ITERATOR_HPP
#define _GPU_MM2_PLAN_ITERATOR_HPP

#include "gpu_mm2_internals.hpp"  // ALL_LANES

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// Usage of plan_iterator_strict<W> is best explained by the following pseudocode
//
//    plan_iterator_strict<W> iterator;
//    
//    if (!iterator.init())
//	  return;
//
//    do {
//        // As this point, either init() or next_cell() has returned 'true'
//	  // Do per-cell initialization (using value of iterator.icell_curr)
//	
//	  while (iterator.next_mt()) {
//	      // At this point, next_mt() has returned 'true'
//	      // Do per-mt processing (using values of iterator.icell_curr, iterator.icl_curr)
//	  }
//	
//	  // At this point, next_mt() has returned 'false'
//	  // Do per-cell finalization (using value of iterator.icell_curr)
//	
//    } while (iterator.next_cell());


template<int W, bool Debug>
struct plan_iterator_strict
{
    const ulong *plan_mt;
    
    // Same on all threads in block, not necessarily multiple of 32, constant after construction
    uint imt_soft_end;
    uint imt_hard_end;
    
    // These members are managed by load_rb().
    ulong mt_rb;           // warp holds plan_mt[imt_rb:imt_rb+32).
    uint imt_rb;           // same on all threads in warp (not block), multiple of 32.
    uint icell_rb;         // derived from mt_rb

    // State related to outer loop over cells
    uint icell_curr;       // every warp is guaranteed to loop over the same icell_curr values
    uint imt_cell_start;   // starting index of current cell
    uint imt_cell_end;     // if not known yet, equal to 'imt_hard_end' 

    // State related to inner loop over plan_mt
    uint imt_next;         // for use in next call to next_mt(), can be >= imt_hard_end
    uint icl_curr;         // set in next_mt(), cleared in init() and next_cell()


    // Updates { mt_rb, icell_rb }.
    // Caller must first update (or initialize) imt_rb.
    __device__ void _load_rb()
    {
	if constexpr (Debug) {
	    assert(imt_rb < imt_hard_end);
	    assert((imt_rb & 31) == 0);
	}
	
	// Block dims are (32,W), so threadIdx.x is the laneId.
	uint imt = min(imt_rb + threadIdx.x, imt_hard_end - 1);
	
	this->mt_rb = plan_mt[imt];
	this->icell_rb = uint(mt_rb) & ((1U << 20) - 1);
    }


    // Updates imt_cell_end.
    // Caller must first initialize { imt_cell_start, icell_curr } plus ring buffer data.
    // Called in start_cell(), to initialize imt_cell_end at the beginning of a cell.
    // Also called in advance_rb(), when new ring buffer data is seen.

    __device__ void _update_cell_end()
    {
	if constexpr (Debug)
	    assert(imt_cell_end == imt_hard_end);

	// Block dims are (32,W), so threadIdx.x is the laneId.
	int imt_lane = imt_rb + threadIdx.x;
	bool flag = (imt_lane > imt_cell_start) && (imt_lane < imt_hard_end) && (icell_rb != icell_curr);
	uint mask = __ballot_sync(ALL_LANES, flag);

	imt_cell_end = mask ? (imt_rb + __ffs(mask) - 1) : imt_hard_end;
    }

    
    // Updates { icell_curr, imt_cell_end, imt_next }.
    // Caller must first initialize imt_cell_start.
    __device__ void _start_cell()
    {
	if constexpr (Debug) {
	    assert(imt_cell_start >= imt_rb);
	    assert(imt_cell_start < imt_rb+32);
	    assert(imt_cell_start < imt_soft_end);
	}
	
	icell_curr = __shfl_sync(ALL_LANES, icell_rb, imt_cell_start & 31);

	// Initialize imt_cell_end
	imt_cell_end = imt_hard_end;   // to avoid failing an assert in _update_cell_end()
	_update_cell_end();            // updates imt_cell_end
	
	// Block dims are (32,W), so threadIdx.y is the warpId.
	imt_next = imt_cell_start + threadIdx.y;
	icl_curr = 0;  // arbitrary
    }

    
    // Updates { imt_rb, mt_rb, icell_rb, imt_cell_end }.
    // Caller must first initialize { imt_rb, imt_cell_start, imt_cell_end, icell_curr }.
    
    __device__ void _advance_rb()
    {
	if constexpr (Debug) {
	    assert(imt_cell_end >= imt_rb + 32);
	}
	
	imt_rb += 32;
	_load_rb();
	_update_cell_end();
    }
    
    
    __device__ bool init(const ulong *plan_mt_, uint nmt, uint nmt_per_block)
    {
	if constexpr (Debug) {
	    assert(nmt_per_block > 0);
	    assert((nmt_per_block & 31) == 0);	
	    assert((gridDim.x-1) * nmt_per_block < nmt);
	    assert((gridDim.x) * nmt_per_block >= nmt);
	    assert(blockDim.x == 32);
	    assert(blockDim.y == W);
	}
	       
	// Initialize:
	//   this->plan_mt
	//   this->imt_soft_end
	//   this->imt_hard_end
	//   this->mt_rb
	//   this->imt_rb
	//   this->icell_rb
	
	uint b = blockIdx.x;

	this->plan_mt = plan_mt_;
	this->imt_rb = b*nmt_per_block;
	this->imt_soft_end = min(nmt, (b+1)*nmt_per_block);
	this->imt_hard_end = nmt;

	_load_rb();  // initializes mt_rb, icell_rb

	// Initialize:
	//   this->imt_cell_start
	
	this->imt_cell_start = imt_rb;  // correct value if b==0

	if (b > 0) {
	    ulong mt_prev = plan_mt[imt_rb-1];
	    uint icell_prev = uint(mt_prev) & ((1U << 20) - 1);

	    for (;;) {
		// Block dims are (32,W), so threadIdx.x is the laneId.
		int imt_lane = imt_rb + threadIdx.x;
		bool flag = (imt_lane < imt_soft_end) && (icell_rb != icell_prev);
		uint mask = __ballot_sync(ALL_LANES, flag);
		
		if (mask) {
		    this->imt_cell_start = imt_rb + __ffs(mask) - 1;
		    break;
		}

		if (imt_soft_end <= imt_rb + 32)
		    return false;

		// Don't call _advance_rb(), since { imt_cell_start, icell_curr } aren't initialized yet.
		imt_rb += 32;
		_load_rb();
	    }
	}

	// Based on imt_cell_start, initialize:
	//   icell_curr
	//   imt_cell_end
	//   imt_next

	_start_cell();
	return true;
    }


    __device__ bool next_mt()
    {
	if ((imt_next >= imt_cell_end) && (imt_cell_end <= imt_rb+32))
	    return false;

	if (imt_next >= imt_rb + 32) {
	    _advance_rb();

	    // Redo this test, since _advance_rb() may have modified the value of imt_cell_end.
	    if (imt_next >= imt_cell_end)
		return false;
	}

	if constexpr (Debug) {
	    assert(imt_next >= imt_rb);
	    assert(imt_next < imt_rb + 32);
	    assert(imt_next <= imt_cell_end);
	    assert(imt_cell_end <= imt_hard_end);
	}

	ulong mt_remote = __shfl_sync(ALL_LANES, mt_rb, imt_next & 31);
	uint icell_remote = uint(mt_remote) & ((1U << 20) - 1);
	uint icl_remote = uint(mt_remote >> 20) & ((1U << 26) - 1);

	if constexpr (Debug)
	    assert(icell_remote == icell_curr);
	
	icl_curr = icl_remote;
	imt_next += W;
	return true;
    }


    __device__ bool next_cell()
    {
	if constexpr (Debug) {
	    assert(imt_rb <= imt_cell_end);
	    assert(imt_cell_end <= imt_hard_end);
	    assert(imt_cell_end <= imt_next);  // condition for next_mt() to return false
	}
	
	if (imt_cell_end > imt_rb + 32) {
	    // Need to advance the ring buffer to "see" the next cell, but the ring buffer
	    // was not advanced in prior calls to next_mt().
	    //
	    // This only happens in this corner case: imt_cell_end == imt_hard_end < imt_next.
	    // In this case, we just advance the ring buffer.

	    _advance_rb();       // updates imt_rb, mt_rb, icell_rb
	    _update_cell_end();  // updates imt_cell_end

	    // Re-check assertions above
	    if constexpr (Debug) {
		assert(imt_rb <= imt_cell_end);
		assert(imt_cell_end <= imt_hard_end);
		assert(imt_cell_end <= imt_next);
		assert(imt_cell_end <= imt_rb + 32);
	    }
	    
	    // Fall through...
	}

	// At this point, the following conditions are satisfied:
	//
	//   imt_rb <= imt_cell_end <= (imt_rb + 32)
	//   imt_cell_end <= imt_next
	//
	// That is, the current cell ends in the current ring buffer.
	// We start a new cell iff (imt_cell_end < imt_soft_end).

	if (imt_cell_end >= imt_soft_end)
	    return false;
	
	imt_cell_start = imt_cell_end;
	_start_cell();  // updates icell_curr, imt_cell_end, imt_next
	return true;
    }
};


}  // namespace gpu_mm

#endif  // _GPU_MM2_PLAN_ITERATOR_HPP
