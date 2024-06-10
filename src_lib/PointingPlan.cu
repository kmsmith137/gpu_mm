#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/string_utils.hpp>
#include <gputils/constexpr_functions.hpp>   // constexpr_is_log2()

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// absorb_mt(): Helper function for plan_kerne().
// The number of arguments is awkwardly large!

__device__ __forceinline__ void
absorb_mt(ulong *plan_mt, int *shmem,        // pointers
	  ulong &mt_local, int &nmt_local,   // per-warp ring buffer
	  uint icell, uint amask, int na,    // map cells to absorb
	  uint s, int sid0, int na_prev,     // additional data needed to construct mt_new
	  int nmt_max, int &err)             // error testing and reporting
{
    if (na == 0)
	return;
    
    // Block dims are (W,32), so threadIdx.x is the laneId.
    int laneId = threadIdx.x;
    
    // Logical laneId (relative to current value of nmt_local, wrapped around)
    int llid = (laneId + 32 - nmt_local) & 31;   // FIXME do I need "+ 32"?
    
    // Permute 'icell' so that llid=N contains the N-th active icell
    uint src_lane = __fns(amask, 0, llid+1);
    icell = __shfl_sync(ALL_LANES, icell, src_lane & 31);  // FIXME do I need "& 31"?

    // Secondary cache line index (note that sid0 is 0-based, whereas sid is 1-based)
    uint a = na_prev + llid;
    uint sid = (a > 0) ? (sid0+a) : 0;

    // Promote (uint20 icell) to (uint64 mt_new).
    // Reminder: mt_new bit layout is
    //   Low 10 bits = Global xcell index
    //   Next 10 bits = Global ycell index
    //   Next 26 bits = Primary TOD cache line index
    //   High 18 bits = Secondary TOD cache line index, 1-based (relative to start of block)
    
    ulong mt_new = icell | (ulong(s >> 5) << 20) | (ulong(sid) << 46);
    int nmt_new = nmt_local + na;

    // Extend ring buffer.
    // If nmt_local is >32, then it "wraps around" from mt_local to mt_new.
    
    mt_local = (laneId < nmt_local) ? mt_local : mt_new;
    nmt_local += mt_new;

    if (nmt_local < 32)
	return;

    // If we get here, we've accumulated 32 values of 'mt_local'.
    // These values can now be written to global memory.
    
    int nout = 0;
    if (laneId == 0)
	nout = atomicAdd(shmem, 32);
    nout = __shfl_sync(ALL_LANES, nout, 0);  // broadcast from lane 0 to all lanes
    nout += laneId;

    if (nout < nmt_max)
	plan_mt[nout] = mt_local;

    nmt_local -= 32;
    mt_local = mt_new;
    err = (nout < nmt_max) ? err : (err | 4);
}


template<typename T, int W>
__global__ void plan_kernel(ulong *plan_mt, const T *xpointing, uint *nmt_cumsum, uint nsamp, uint nsamp_per_block, int nypix, int nxpix, int *errp)
{
    // Assumed for convienience in shared memory logic
    static_assert(W <= 30);
		      
    // Block dims are (W,32)
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    
    // Shared memory layout:
    //   int    nmt_counter       running total of 'plan_mt' values written by this block
    //   int    sid_counter       running total of secondary cache lines for this block
    //   int    nmt_local[W]      
    
    __shared__ int shmem[32];  // only need (W+2) elts, but convenient to pad to 32
    
    // Zero shared memory
    if (warpId == 0)
	shmem[laneId] = 0;
    
    // Range of TOD samples to be processed by this threadblock.
    int b = blockIdx.x;
    uint s0 = b * nsamp_per_block;
    uint s1 = min(nsamp, (b+1) * nsamp_per_block);
    
    // Range of nmt values to be written by this threadblock.
    uint mt_out0 = b ? nmt_cumsm[b-1] : 0;
    uint mt_out1 = nmt_cumsum[b];
    int nmt_max = mt_out1 - mt_out0;
    
    // Shift output pointer 'plan_mt'.
    // FIXME some day, consider implementing cache-aligned IO as optimization
    plan_mt += mt_out0;
    
    // (mt_local, nmt_local) act as a per-warp ring buffer.
    // The value of nmt_local is the same on all threads in the warp.
    ulong mt_local = 0;
    int nmt_local = 0;
    int err = 0;
     
    for (uint s = s0 + 32*warpId + laneId; s < s1; s += 32*W) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];

	// For now, I'm including the range checks, even though they should be
	// redundant with the preplan. (FIXME: does it affect running time?)
	
	range_check_ypix(ypix, nypix, err);  // defined in gpu_mm2_internals.hpp
	range_check_xpix(xpix, nxpix, err);  // defined in gpu_mm2_internals.hpp
	normalize_xpix(xpix, nxpix);         // defined in gpu_mm2_internals.hpp
	 
	int iypix0, iypix1, ixpix0, ixpix1;
	quantize_ypix(iypix0, iypix1, ypix, nypix);  // defined in gpu_mm2_internals.hpp
	quantize_xpix(ixpix0, ixpix1, xpix, nxpix);  // defined in gpu_mm2_internals.hpp
	
	int iycell_e, iycell_o, ixcell_e, ixcell_o;
	set_up_cell_pair(iycell_e, iycell_o, iypix0, iypix1);  // defined in gpu_mm2_internals.hpp
	set_up_cell_pair(ixcell_e, ixcell_o, ixpix0, ixpix1);  // defined in gpu_mm2_internals.hpp
	
	uint icell0, icell1, icell2, icell3;
	uint amask0, amask1, amask2, amask3;
	int na0, na1, na2, na3;
	 
	analyze_cell_pair(iycell_e, ixcell_e, icell0, amask0, na0);
	analyze_cell_pair(iycell_e, ixcell_o, icell1, amask1, na1);
	analyze_cell_pair(iycell_o, ixcell_e, icell2, amask2, na2);
	analyze_cell_pair(iycell_o, ixcell_o, icell3, amask3, na3);
	
	int sid0 = 0;
	if (laneId == 0)
	    sid0 = atomicAdd(shmem+1, na0+na1+na2+na3-1);
	sid0 = __shfl_sync(ALL_LANES, sid0, 0);  // broadcast from lane 0 to all lanes
	// Note that sid0 is a zero-based index
	
	absorb_mt(plan_mt, shmem,        // pointers
		  mt_local, nmt_local,   // per-warp ring buffer
		  icell0, amask0, na0,   // map cells to absorb
		  s, sid0, 0,            // additional data needed to construct mt_new
		  nmt_max, err);         // error testing and reporting
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  icell1, amask1, na1,
		  s, sid0, na0,
		  nmt_max, err);
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  icell2, amask2, na2,
		  s, sid0, na0+na1,
		  nmt_max, err);
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  icell3, amask3, na3,
		  s, sid0, na0+na1+na2,
		  nmt_max, err);
    }
    
    if (laneId == 0)
	shmem[warpId+2] = nmt_local;

    __syncthreads();

    // FIXME logic here could be optimized -- align IO on cache lines,
    // use fewer warp shuffles to reduce.
    
    int shmem_remote = shmem[laneId];
    
    int nout = __shfl_sync(ALL_LANES, shmem_remote, 0);    // nmt_counter
    for (int w = 0; w < warpId; w++)
	nout += __shfl_sync(ALL_LANES, shmem_remote, w+2);  // value of 'nmt_local' on warp w
    nout += laneId;

    if ((laneId < nmt_local) && (nout < nmt_max))
	plan_mt[nout] = mt_local;

    bool fail = (warpId == (W-1)) && (laneId == nmt_local) && (nout != nmt_max);
    err = fail ? (err | 4) : err;
	
    errp[b] = err;
}


// -------------------------------------------------------------------------------------------------


template<typename T>
PointingPlan::PointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu,
			   const Array<unsigned char> &buf, const Array<unsigned char> &tmp_buf)
{

}


// -------------------------------------------------------------------------------------------------


#define INSTANTIATE(T) \
    template PointingPlan::PointingPlan(const PointingPrePlan &pp, \
	const gputils::Array<T> &xpointing_gpu, \
	const gputils::Array<unsigned char> &buf, \
	const gputils::Array<unsigned char> &tmp_buf)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
