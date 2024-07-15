#include "../include/gpu_mm.hpp"
#include "../include/plan_iterator.hpp"

#include <cassert>
#include <gputils/cuda_utils.hpp>

using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// Helper for tod2map_kernel()
template<typename T>
__device__ void add_tqu(T *sp, int iy, int ix, T t, T q, T u, T w)
{
    bool in_cell = ((ix | iy) & ~63) == 0;
    int s = (iy << 6) | ix;

    // Warp divergence here
    if (in_cell) {
	atomicAdd(sp + s, w*t);
	atomicAdd(sp + s + 64*64, w*q);
	atomicAdd(sp + s + 2*64*64, w*u);
    }

    __syncwarp();
}


template<typename T, int W, bool Debug>
__global__ void __launch_bounds__(32*W, 1)
tod2map_kernel(
    T *lmap,
    const T *tod,
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
	// uint iy0_cell = (icell >> 10) << 6;
	// uint ix0_cell = (icell & ((1<<10) - 1)) << 6;
	bool valid = (iycell < nycells) && (ixcell < nxcells);
	
	long offset = valid ? cell_offsets[iycell*nxcells + ixcell] : -1;
	err = (offset >= 0) ? err : errflag_pixel_outlier;

	if (offset >= 0) {
	    // Zero shared memmory
	    for (int s = 32*warpId + laneId; s < 3*64*64; s += 32*W)
		shmem[s] = 0;
	}
	
	__syncthreads();

	// Inner loop over TOD cache lines

	while (iterator.get_cl()) {
	    if (offset < 0)
		continue;
	    
	    uint icl = iterator.icl;
	    long s = (long(icl) << 5) + laneId;

	    T ypix = xpointing[s];
	    T xpix = xpointing[s + nsamp];
	    T alpha = xpointing[s + 2*nsamp];
	    T t = tod[s];
	    
	    T q, u;
	    dtype<T>::xsincos(2*alpha, &u, &q);
	    q *= t;
	    u *= t;
	    
	    int iy = quantize_pixel(ypix, 64*1024);
	    int ix = quantize_pixel(xpix, 64*1024);

	    T dy = ypix - iy;
	    T dx = xpix - ix;
	    
	    iy -= (iycell << 6);
	    ix -= (ixcell << 6);
	    
	    add_tqu(shmem, iy,   ix,   t, q, u, (1-dy) * (1-dx));
	    add_tqu(shmem, iy,   ix+1, t, q, u, (1-dy) * (dx));
	    add_tqu(shmem, iy+1, ix,   t, q, u, (dy) * (1-dx));
	    add_tqu(shmem, iy+1, ix+1, t, q, u, (dy) * (dx));
	}

	if (offset < 0)
	    continue;
	
	__syncthreads();
	
	// Shared -> global
	
	for (int y = warpId; y < 64; y += W) {
	    for (int x = laneId; x < 64; x += 32) {
		int ss = 64*y + x;                 // shared memory offset
		long sg = offset + y*ystride + x;  // global memory offset

		T t = shmem[ss];
		if (!__reduce_or_sync(ALL_LANES, t != 0))
		    continue;

		atomicAdd(lmap + sg, t);
		atomicAdd(lmap + sg + polstride, shmem[ss+64*64]);
		atomicAdd(lmap + sg + 2*polstride, shmem[ss+2*64*64]);
	    }
	}

	__syncthreads();
    }
}


template<typename T>
void launch_tod2map(gputils::Array<T> &local_map,
		    const gputils::Array<T> &tod,
		    const gputils::Array<T> &xpointing,
		    const LocalPixelization &local_pixelization,
		    const ulong *plan_mt, uint *errflags,
		    long nmt, long nmt_per_block, long nblocks,
		    bool allow_outlier_pixels, bool debug)
{
    static constexpr int W = 16;  // warps per threadblock
    static constexpr int shmem_nbytes = 3 * 64 * 64 * sizeof(T);

    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "launch_tod2map", true);            // on_gpu = true
    check_local_map(local_map, local_pixelization, "launch_tod2map", true);  // on_gpu = true
    check_xpointing(xpointing, nsamp, "launch_tod2map", true);               // on_gpu = true

    xassert(nmt > 0);
    xassert(plan_mt != nullptr);
    xassert(nmt_per_block > 0);
    xassert((nmt_per_block % 32) == 0);  // Not necessary, but failure probably indicates a bug
    xassert((nblocks) * nmt_per_block >= nmt);
    xassert((nblocks-1) * nmt_per_block < nmt);
    
    if (debug) {
	tod2map_kernel<T,W,true> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (local_map.data,
	     tod.data,
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
	tod2map_kernel<T,W,false> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (local_map.data,
	     tod.data,
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

    CUDA_PEEK("tod2map kernel launch");
    
    uint errflags_to_ignore = allow_outlier_pixels ? errflag_pixel_outlier : 0;
    check_gpu_errflags(errflags, nblocks, "tod2map", errflags_to_ignore);
}


#define INSTANTIATE(T) \
    template void launch_tod2map(gputils::Array<T> &local_map, \
			         const gputils::Array<T> &tod, \
			         const gputils::Array<T> &xpointing, \
			         const LocalPixelization &local_pixelization, \
			         const ulong *plan_mt, uint *errflags, \
			         long nmt, long nmt_per_block, long nblocks, \
			         bool allow_outlier_pixels, bool debug)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
