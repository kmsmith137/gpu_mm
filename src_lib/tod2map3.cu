#include "../include/gpu_mm2.hpp"
#include "../include/PlanIterator2.hpp"

#include <cassert>
#include <gputils/cuda_utils.hpp>

using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T>
__device__ void add_tqu(T *sp, int iy, int ix, int iy0_cell, int ix0_cell, T t, T q, T u, T w)
{
    bool y_in_cell = ((iy & ~63) == iy0_cell);
    bool x_in_cell = ((ix & ~63) == ix0_cell);
    int s = ((iy & 63) << 6) | (ix & 63);

    // Warp divergence here
    if (y_in_cell && x_in_cell) {
	atomicAdd(sp + s, w*t);
	atomicAdd(sp + s + 64*64, w*q);
	atomicAdd(sp + s + 2*64*64, w*u);
    }
}


template<typename T, bool Debug>
__global__ void __launch_bounds__(12*32, 1)
    tod2map3_kernel(T *map, const T *tod, const T *xpointing, const ulong *plan_mt,
		    uint nsamp, int nypix, int nxpix, uint nmt, uint nmt_per_block)
{
    static constexpr uint W = 12;   // warps_per_threadblock
    static constexpr T one = 1;
    static constexpr T two = 2;

    T dummy_computation = 0;

    // Shared memory layout
    // 55.5 KB in single precision, 111 KB in double precision.
    //
    //   __shared__ T map_cell[3*64*64];   // {t,q,u}
    //   __shared__ T exchange[5*12*32];   // {y,x,t,q,u}
    //   __shared__ int counts[32];
    
    T *shmem_exchange = dtype<T>::get_shmem();
    int *shmem_counts = dtype<int>::get_shmem() + (5*W*32) * (sizeof(T)/4);
    // int *shmem_counts = dtype<int>::get_shmem() + (3*64*64 + 5*W*32) * (sizeof(T)/4);

    if constexpr (Debug) {
	assert(blockDim.x == 32);
	assert(blockDim.y == W);
    }
    
    // Threadblock has shape (32,W), so threadIdx.x is the laneId, and threadIdx.y is the warpId.
    const uint laneId = threadIdx.x;
    const uint warpId = threadIdx.y;
    
    if (warpId == 0)
	shmem_counts[laneId] = 0;
    
    PlanIterator2<W,Debug> iterator(plan_mt, nmt, nmt_per_block);

    // Outer loop over map cells

    while (iterator.get_cell()) {
	// uint icell = iterator.icell;
	// uint iy0_cell = (icell >> 10) << 6;
	// uint ix0_cell = (icell & ((1<<10) - 1)) << 6;

	// Zero shared memmory map cell
	// for (int s = 32*warpId + laneId; s < 3*64*64; s += 32*W)
	// shmem[s] = 0;

	// Inner loop over TOD cache lines

	for (;;) {
	    bool have_cl = iterator.get_cl();

	    if (have_cl) {
		uint icl = iterator.icl;
		uint s = (icl << 5) + laneId;  // FIXME 32-bit overflow

		T ypix = xpointing[s];
		T xpix = xpointing[s + nsamp];
		T alpha = xpointing[s + 2*nsamp];
		T t = tod[s];

		// FIXME add 'status' argument, and calls to range_check_{xpix,ypix}().
		normalize_xpix(xpix, nxpix);   // defined in gpu_mm2_internals.hpp
		
		T cos_2a, sin_2a;
		dtype<T>::xsincos(two*alpha, &sin_2a, &cos_2a);

		uint ss = 32*warpId + laneId;
		shmem_exchange[ss] = ypix;
		shmem_exchange[ss + 32*W] = xpix;
		shmem_exchange[ss + 2*32*W] = t;
		shmem_exchange[ss + 3*32*W] = t * cos_2a;
		shmem_exchange[ss + 4*32*W] = t * sin_2a;
	    }
	    
	    if (laneId == 0)
		shmem_counts[warpId] = have_cl ? 32 : 0;
	    
	    __syncthreads();

	    // Which warps are still processing TOD cache lines?
	    uint wmask = __ballot_sync(ALL_LANES, shmem_counts[laneId] != 0);

	    if (!wmask)
		break;  // End of inner loop

	    for (int w = 0; w < W; w++) {
		if ((wmask & (1U << w)) == 0)
		    continue;

		uint ss = 32*w + laneId;
		T ypix = shmem_exchange[ss];
		T xpix = shmem_exchange[ss + 32*W];
		T val = shmem_exchange[ss + (warpId>>2)*32*W + 2*32*W];

		int iy0, iy1, ix0, ix1;
		quantize_ypix(iy0, iy1, ypix, nypix);  // defined in gpu_mm2_internals.hpp
		quantize_xpix(ix0, ix1, xpix, nxpix);  // defined in gpu_mm2_internals.hpp
		
		T dy = ypix - iy0;
		T dx = xpix - ix0;

		bool use_iy0 = (warpId ^ iy0) & 1;
		bool use_ix0 = (warpId ^ ix0) & 2;
		
		int iy = use_iy0 ? iy0 : iy1;
		int ix = use_ix0 ? ix0 : ix1;
		val *= use_iy0 ? (one-dy) : dy;
		val *= use_ix0 ? (one-dx) : dx;

		bool valid = ((iy >= 0) && (iy < 64) && (ix >= 0) && (ix < 64));

#if 1
		int ipix = 64*iy + ix;
		uint lbit = 1 << laneId;
		uint lbits = lbit | (lbit-1);
		
		//for (;;) {
		    int s = valid ? ipix : (-laneId);
		    uint match =  __match_any_sync(ALL_LANES, s);

		    //if (__reduce_and_sync(ALL_LANES, match == lbit))
		    //break;

		    uint mask_above = match & ~lbits;  // does not include laneId
		    uint mask_below = match & lbits;   // does include laneId;
		    
		    T src_val = laneId ? val : 0;
		    uint src_lane = __ffs(mask_above) - 1;
		    val += __shfl_sync(ALL_LANES, src_lane, src_val);

		    int num_below = __popc(mask_below);
		    valid = valid && (num_below & 1);
		    //}
#endif

		dummy_computation += (valid ? val : 0);
	    }
	    
	    __syncthreads();
	}
    }

    map[blockIdx.x*W*32 + warpId*32 + laneId] = dummy_computation;
}


template<typename T>
void launch_tod2map3(T *map, const T *tod, const T *xpointing, const ulong *plan_mt, 
		     long nsamp, long nypix, long nxpix, int nmt, int nmt_per_block, bool debug)
{
    static constexpr int W = 12;  // warps per threadblock

    check_nsamp(nsamp, "launch_tod2map3");
    check_nypix(nypix, "launch_tod2map3");
    check_nxpix(nxpix, "launch_tod2map3");
    
    xassert(nmt > 0);
    xassert(nmt_per_block > 0);
    xassert((nmt_per_block % 32) == 0);

    int nblocks = (nmt + nmt_per_block - 1) / nmt_per_block;
    // int shmem_nbytes = 3 * 64 * 64 * sizeof(T);
    int shmem_nbytes = (5 * W * 32 * sizeof(T)) + 128;
    
    if (debug) {
	tod2map3_kernel<T,true> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (map, tod, xpointing, plan_mt, nsamp, nypix, nxpix, nmt, nmt_per_block);
    }
    else {
	tod2map3_kernel<T,false> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (map, tod, xpointing, plan_mt, nsamp, nypix, nxpix, nmt, nmt_per_block);
    }

    CUDA_PEEK("tod2map3 kernel launch");
}


#define INSTANTIATE(T) \
    template void launch_tod2map3(T *map, const T *tod, const T *xpointing, const ulong *plan_mt, \
				  long nsamp, long nypix, long nxpix, int nmt, int nmt_per_block, bool debug);

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
