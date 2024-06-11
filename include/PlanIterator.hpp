
usage()
{
    PlanIterator<W> iterator;
    
    if (!iterator.init())
	return;

    do {
	// Point A: either init() or next_cell() has returned 'true'
	// Do per-cell initialization (iterator.icell_curr)
	
	while (iterator.next_mt()) {
	    // Point B: next_mt() has returned 'true'
	    // Do per-mt processing (iterator.icell_curr, iterator.icl_curr)
	}
	
	// Point C: next_mt() has returned 'false'
	// Do per-cell finalization (iterator.icell_curr)
	
    } while (iterator.next_cell());
}


template<int W>
struct PlanIterator
{
    // Same on all threads, not necessarily multiple of 32, constant after construction
    // FIXME imt_last or imt_end?
    uint imt_last;
    
    // These members are managed by 
    ulong mt_rb;     // warp holds plan_mt[imt_rb:imt_rb+32).
    uint imt_rb;     // same on all threads, multiple of 32.
    uint icell_rb;   // derived from mt_rb

    // State related to outer loop over cells
    uint icell_curr;          // same on all threads in block (not just warp)
    uint imt_of_curr_icell;   // same on all threads in block (not just warp)
    bool flag_next_cell;      // if true, then we know which cell is next
    uint icell_next;          // meaningful if flag_next_icell=true
    uint imt_of_next_icell;   // meaningful if flag_next_icell=true

    // State related to inner loop over plan_mt
    uint imt_next;

    
    // Updates 'mt_rb' and 'icell_rb', after caller updates (or initializes) 'imt_rb'.
    __device__ void load_rb()
    {
	// Block dims are (32,W), so threadIdx.x is the laneId.
	int laneId = threadIdx.x;
	uint imt = min(imt_rb+laneId, imt_last);
	
	this->mt_rb = plan_mt[imt];
	this->icell_rb = uint(mt_rb) & ((1U << 20) - 1);
    }


    __device__ void advance_rb(uint imt)
    {
	if (imt >= imt_rb+32) {
	    this->imt_rb += 32;
	    this->load_rb();
	}
		
	// FIXME can remove these assert after testing
	assert(imt >= imt_rb);
	assert(imt < imt_rb+32);
    }

    
    __device__ bool init(ulong *plan_mt, uint nmt, uint nmt_per_block)
    {
	uint b = blockIdx.x;

	// Initialize:
	//   this->imt_last
	//   this->mt_rb
	//   this->imt_rb
	//   this->icell_rb
	
	this->imt_rb = b*nmt_per_block;
	this->imt_last = max(nmt, (b+1)*nmt_per_block) - 1;
	this->load_rb();  // initializes mt_rb, icell_rb

	// FIXME can remove this assert after testing
	assert(imt_rb <= imt_last);

	// Code after this point initializes:
	//   this->icell_curr
	//   this->imt_of_curr_icell
	//   this->flag_next_cell
	//   this->icell_next
	//   this->imt_of_next_icell
	
	if (b == 0) {
	    this->icell_curr = __shfl_sync(ALL_LANES, icell_rb, 0);
	    this->imt_of_curr_icell = 0;

	    // FIXME this code will certainly go into a helper method
	    mask = __ballot_sync(ALL_LANES, icell_curr != icell_rb);
	    
	    if (mask == 0)
		this->flag_next_cell = false;
	    else {
		int lane = __ffs();
		this->flag_next_cell = true;
		this->imt_of_next_icell = imt_rb + lane;
		this->icell_next = __shfl_sync(ALL_LANES, icell_rb, lane);
	    }

	    return true;
	}

	ulong mt_prev = plan_mt[xx];  // same on all threads
	uint icell_prev = uint(mt_prev) & ((1U << 20) - 1);

	for (;;) {
	    for (uint imt = imt_rb; imt <= imt_last; imt++) {
		this->advance_rb(imt);
		this->icell_curr = __shfl_sync(ALL_LANES, icell_rb, imt & 31);
		
		if (icell_curr != icell_prev) {
		    this->imt_next = imt + warpId;
		    return true;
		}
	    }

	    return false;
	}
    }


    __device__ bool next_mt()
    {
	if (imt_next > imt_last)
	    return false;

	this->advance_rb(imt_next);
	
	uint icell_next = __shfl_sync(ALL_LANES, icell_rb, imt_next & 31);

	if (icell_next != icell_)
	    return false;

	this->icl_curr = __shfl_sync();
	this->imt_next += W;
	return true;
    }
};

