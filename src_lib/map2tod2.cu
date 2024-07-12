#include "../include/gpu_mm2.hpp"
#include "../include/PlanIterator2.hpp"

#include <cassert>
#include <gputils/cuda_utils.hpp>

using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// The "pre-map2tod" kernel partially zeroes the TOD.
// Launch with threadIdx = { 32*W } (not {32,W}).

template<typename T>
__global__ void pre_map2tod_kernel(T *tod, const ulong *plan_mt, uint nmt, uint nmt_per_block)
{
    uint imt0 = (blockIdx.x) * nmt_per_block + threadIdx.x;
    uint imt1 = (blockIdx.x + 1) * nmt_per_block;
    imt1 = (imt1 < nmt) ? imt1 : nmt;
    imt1 = (imt1 + 31U) & ~31U;
    
    for (uint imt = imt0; imt < imt1; imt += blockDim.x) {
	uint i = (imt < nmt) ? imt : (nmt-1);
	ulong mt = plan_mt[i];
	
	uint icl_flagged = uint(mt >> 20);
	uint icl = icl_flagged & ((1U << 26) - 1);
	bool zflag = (icl_flagged & (1U << 27)) != 0;
	uint mask = __ballot_sync(ALL_LANES, (imt < nmt) && zflag);

	for (uint lane = 0; lane < 32; lane++) {
	    if (mask & (1U << lane)) {
		uint zcl = __shfl_sync(ALL_LANES, icl, lane);
		uint s = (ulong(zcl) << 5) + (threadIdx.x & 31);
		tod[s] = 0;
	    }
	}
    }
}


template<typename T>
static void launch_pre_map2tod(T *tod, const ulong *plan_mt, int nmt)
{
    static constexpr int W = 4;  // warps per threadblock
    static constexpr int nmt_per_block = 1024;
    
    xassert(tod != nullptr);
    xassert(plan_mt != nullptr);
    xassert(nmt > 0);

    int nblocks = (nmt + nmt_per_block - 1) / nmt_per_block;

    pre_map2tod_kernel<T> <<< nblocks, 32*W >>>
	(tod, plan_mt, nmt, nmt_per_block);

    CUDA_PEEK("pre_map2tod kernel launch");
}


// -------------------------------------------------------------------------------------------------


template<typename T>
__device__ T eval_tqu(T *sp, int iy, int ix, int iy0_cell, int ix0_cell, T cos_2a, T sin_2a)
{
    bool y_in_cell = ((iy & ~63) == iy0_cell);
    bool x_in_cell = ((ix & ~63) == ix0_cell);
    int s = ((iy & 63) << 6) | (ix & 63);
    T ret = 0;
    
    // Warp divergence here
    // FIXME could remove it I think
    if (y_in_cell && x_in_cell) {
	ret = sp[s] + (cos_2a * sp[s+64*64]) + (sin_2a * sp[s+2*64*64]);
    }

    __syncwarp();
    return ret;
}


template<typename T, int W, bool Debug>
__global__ void __launch_bounds__(32*W, 1)
map2tod_kernel(T *tod, const T *map, const T *xpointing, const ulong *plan_mt,
	       uint nsamp, int nypix, int nxpix, uint nmt, uint nmt_per_block)
{
    static constexpr T one = 1;
    static constexpr T two = 2;
    
    // 48 KB in single precision, 96 KB in double precision.
    // __shared__ T shmem[3*64*64];
    T *shmem = dtype<T>::get_shmem();

    if constexpr (Debug) {
	assert(blockDim.x == 32);
	assert(blockDim.y == W);
    }
    
    // Threadblock has shape (32,W), so threadIdx.x is the laneId, and threadIdx.y is the warpId.
    const uint laneId = threadIdx.x;
    const uint warpId = threadIdx.y;
    
    plan_iterator<W,Debug> iterator(plan_mt, nmt, nmt_per_block);

    // Outer loop over map cells

    while (iterator.get_cell()) {
	uint icell = iterator.icell;
	uint iy0_cell = (icell >> 10) << 6;
	uint ix0_cell = (icell & ((1<<10) - 1)) << 6;
	
	// Shared -> global
	
	for (int y = warpId; y < 64; y += W) {
	    for (int x = laneId; x < 64; x += 32) {
		int ss = 64*y + x;                           // shared memory offset
		int sg = (iy0_cell+y)*nxpix + (ix0_cell+x);  // global memory offset

		shmem[ss] = map[sg];
		shmem[ss + 64*64] = map[sg + nypix*nxpix];
		shmem[ss + 2*64*64] = map[sg + 2*nypix*nxpix];
	    }
	}
	     
	__syncthreads();

	// Inner loop over TOD cache lines

	while (iterator.get_cl()) {
	    uint icl = iterator.icl;
	    uint s = (icl << 5) + laneId;  // FIXME 32-bit overflow

	    T ypix = xpointing[s];
	    T xpix = xpointing[s + nsamp];
	    T alpha = xpointing[s + 2*nsamp];

	    // FIXME add 'status' argument, and calls to range_check_{xpix,ypix}().
	    normalize_xpix(xpix, nxpix);   // defined in gpu_mm_internals.hpp

	    int iy0, iy1, ix0, ix1;
	    quantize_ypix(iy0, iy1, ypix, nypix);  // defined in gpu_mm_internals.hpp
	    quantize_xpix(ix0, ix1, xpix, nxpix);  // defined in gpu_mm_internals.hpp

	    T dy = ypix - iy0;
	    T dx = xpix - ix0;
	    
	    T cos_2a, sin_2a;
	    dtype<T>::xsincos(two*alpha, &sin_2a, &cos_2a);

	    T t = eval_tqu(shmem, iy0, ix0, iy0_cell, ix0_cell, cos_2a, sin_2a) * (one-dy) * (one-dx);
	    t += eval_tqu(shmem, iy0, ix1, iy0_cell, ix0_cell, cos_2a, sin_2a) * (one-dy) * (dx);
	    t += eval_tqu(shmem, iy1, ix0, iy0_cell, ix0_cell, cos_2a, sin_2a) * (dy) * (one-dx);
	    t += eval_tqu(shmem, iy1, ix1, iy0_cell, ix0_cell, cos_2a, sin_2a) * (dy) * (dx);

	    bool mflag = iterator.icl_flagged & (1U << 26);
	    
	    if (mflag)
		atomicAdd(tod+s, t);
	    else
		tod[s] = t;
	}

	__syncthreads();
    }
}


template<typename T>
void launch_map2tod(gputils::Array<T> &tod,
		    const gputils::Array<T> &map,
		    const gputils::Array<T> &xpointing,
		    const ulong *plan_mt, long nmt,
		    long nmt_per_block, bool debug)
{
    static constexpr int W = 16;  // warps per threadblock

    long nsamp, nypix, nxpix;
    check_tod_and_init_nsamp(tod, nsamp, "launch_map2tod", true);        // on_gpu=true
    check_map_and_init_npix(map, nypix, nxpix, "launch_map2tod", true);  // on_gpu=true
    check_xpointing(xpointing, nsamp, "launch_map2tod", true);           // on_gpu=true

    xassert(nmt > 0);
    xassert(plan_mt != nullptr);
    xassert(nmt_per_block > 0);
    xassert((nmt_per_block % 32) == 0);  // Not necessary, but failure probably indicates a bug

    long nblocks = (nmt + nmt_per_block - 1) / nmt_per_block;
    long shmem_nbytes = 3 * 64 * 64 * sizeof(T);
    
    launch_pre_map2tod(tod.data, plan_mt, nmt);
    
    if (debug) {
	map2tod_kernel<T,W,true> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (tod.data, map.data, xpointing.data, plan_mt, nsamp, nypix, nxpix, nmt, nmt_per_block);
    }
    else {
	map2tod_kernel<T,W,false> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (tod.data, map.data, xpointing.data, plan_mt, nsamp, nypix, nxpix, nmt, nmt_per_block);
    }

    CUDA_PEEK("map2tod kernel launch");
}


#define INSTANTIATE(T) \
    template void launch_map2tod(gputils::Array<T> &tod, \
				 const gputils::Array<T> &map, \
				 const gputils::Array<T> &xpointing, \
				 const ulong *plan_mt, long nmt, \
				 long nmt_per_block, bool debug)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
