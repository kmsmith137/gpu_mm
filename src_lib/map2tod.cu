#include "../include/gpu_mm.hpp"
#include "../include/plan_iterator.hpp"

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


// Helper for map2tod_kernel().
template<typename T>
__device__ T eval_tqu(T *sp, int iy, int ix, T cos_2a, T sin_2a)
{
    bool in_cell = ((ix | iy) & ~63) == 0;
    int s = (iy << 6) | ix;
    
    T ret = in_cell ? (sp[s] + (cos_2a * sp[s+64*64]) + (sin_2a * sp[s+2*64*64])) : 0;
    __syncwarp();
    return ret;
}


template<typename T, int W, bool Debug>
__global__ void __launch_bounds__(32*W, 1)
map2tod_kernel(
    T *tod,
    const T *lmap,
    const T *xpointing,
    const long *cell_offsets,
    const ulong *plan_mt,
    uint *errflags,
    long nsamp,
    int nycells,
    int nxcells,
    long ystride,
    long polstride,
    uint nmt,
    uint nmt_per_block)
{
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
    uint err = 0;
    
    plan_iterator<W,Debug> iterator(plan_mt, nmt, nmt_per_block);

    // Outer loop over map cells

    while (iterator.get_cell()) {
	uint icell = iterator.icell;
	uint iycell = icell >> 10;
	uint ixcell = icell & ((1<<10) - 1);

	// FIXME remove
	// uint iy0_cell = iycell << 6;
	// uint ix0_cell = ixcell << 6;

	bool valid = (iycell < nycells) && (ixcell < nxcells);
	long offset = valid ? cell_offsets[iycell*nxcells + ixcell] : -1;
	err = (offset >= 0) ? err : errflag_pixel_outlier;
	
	// Shared -> global

	if (offset >= 0) {
	    for (int y = warpId; y < 64; y += W) {
		for (int x = laneId; x < 64; x += 32) {
		    int ss = 64*y + x;                 // shared memory offset
		    long sg = offset + y*ystride + x;  // global memory offset
		    
		    shmem[ss] = lmap[sg];
		    shmem[ss + 64*64] = lmap[sg + polstride];
		    shmem[ss + 2*64*64] = lmap[sg + 2*polstride];
		}
	    }
	    __syncthreads();
	}

	// Inner loop over TOD cache lines

	while (iterator.get_cl()) {
	    bool mflag = iterator.icl_flagged & (1U << 26);
	    uint icl = iterator.icl;
	    long s = (long(icl) << 5) + laneId;

	    if (offset < 0) {
		if (!mflag)
		    tod[s] = 0;
		continue;
	    }
	    
	    T ypix = xpointing[s];
	    T xpix = xpointing[s + nsamp];
	    T alpha = xpointing[s + 2*nsamp];
	    
	    T cos_2a, sin_2a;
	    dtype<T>::xsincos(2*alpha, &sin_2a, &cos_2a);
	    
	    int iy = quantize_pixel(ypix, 64*1024);
	    int ix = quantize_pixel(xpix, 64*1024);
	    
	    T dy = ypix - iy;
	    T dx = xpix - ix;

	    iy -= (iycell << 6);
	    ix -= (ixcell << 6);
	    
	    T t = (1-dy) * (1-dx) * eval_tqu(shmem, iy,   ix,   cos_2a, sin_2a);
	    t +=  (1-dy) *   (dx) * eval_tqu(shmem, iy,   ix+1, cos_2a, sin_2a);
	    t +=    (dy) * (1-dx) * eval_tqu(shmem, iy+1, ix,   cos_2a, sin_2a);
	    t +=    (dy) *   (dx) * eval_tqu(shmem, iy+1, ix+1, cos_2a, sin_2a);
	    
	    if (mflag)
		atomicAdd(tod+s, t);
	    else
		tod[s] = t;
	}

	__syncthreads();
    }

    // No need for __syncthreads() before calling write_errflags().
    
    write_errflags(errflags, (uint *)shmem, err);
}


template<typename T>
void launch_map2tod(
    gputils::Array<T> &tod,
    const gputils::Array<T> &local_map,
    const gputils::Array<T> &xpointing,
    const LocalPixelization &local_pixelization,
    const ulong *plan_mt, uint *errflags,
    long nmt, long nmt_per_block, long nblocks,
    bool allow_outlier_pixels, bool debug)
{
    static constexpr int W = 16;  // warps per threadblock
    static constexpr int shmem_nbytes = 3 * 64 * 64 * sizeof(T);

    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "launch_map2tod", true);            // on_gpu = true
    check_local_map(local_map, local_pixelization, "launch_map2tod", true);  // on_gpu = true
    check_xpointing(xpointing, nsamp, "launch_map2tod", true);               // on_gpu = true

    xassert(nmt > 0);
    xassert(plan_mt != nullptr);
    xassert(nmt_per_block > 0);
    xassert((nmt_per_block % 32) == 0);  // Not necessary, but failure probably indicates a bug
    xassert((nblocks) * nmt_per_block >= nmt);
    xassert((nblocks-1) * nmt_per_block < nmt);
    
    launch_pre_map2tod(tod.data, plan_mt, nmt);
    
    if (debug) {
	map2tod_kernel<T,W,true> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (tod.data,
	     local_map.data,
	     xpointing.data,
	     local_pixelization.cell_offsets_gpu.data,
	     plan_mt,
	     errflags,
	     nsamp,
	     local_pixelization.nycells,
	     local_pixelization.nxcells,
	     local_pixelization.ystride,
	     local_pixelization.polstride,
	     nmt,
	     nmt_per_block);
    }
    else {
	map2tod_kernel<T,W,false> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (tod.data,
	     local_map.data,
	     xpointing.data,
	     local_pixelization.cell_offsets_gpu.data,
	     plan_mt,
	     errflags,
	     nsamp,
	     local_pixelization.nycells,
	     local_pixelization.nxcells,
	     local_pixelization.ystride,
	     local_pixelization.polstride,
	     nmt,
	     nmt_per_block);
    }

    CUDA_PEEK("map2tod kernel launch");

    // FIXME check_gpu_errflags() causes measurable slowdown (0.5 ms/call).
    // This isn't large enough to be a high priority, but I'd like to revisit it at
    // some point, mostly for the sake of my own understanding. (I don't understand
    // why it would slow things down so much!)
    
    uint errflags_to_ignore = allow_outlier_pixels ? errflag_pixel_outlier : 0;
    check_gpu_errflags(errflags, nblocks, "map2tod", errflags_to_ignore);
}


#define INSTANTIATE(T) \
    template void launch_map2tod(gputils::Array<T> &tod, \
				 const gputils::Array<T> &local_map, \
				 const gputils::Array<T> &xpointing, \
				 const LocalPixelization &local_pixelization, \
				 const ulong *plan_mt, uint *errflags,	\
				 long nmt, long nmt_per_block, long nblocks, \
				 bool allow_outlier_pixels, bool debug)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
