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


template<typename T, int W, bool Debug>
__global__ void __launch_bounds__(32*W, 1)
    tod2map2_kernel(T *map, const T *tod, const T *xpointing, const ulong *plan_mt,
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
    
    PlanIterator2<W,Debug> iterator(plan_mt, nmt, nmt_per_block);

    // Outer loop over map cells

    while (iterator.get_cell()) {
	uint icell = iterator.icell;
	uint iy0_cell = (icell >> 10) << 6;
	uint ix0_cell = (icell & ((1<<10) - 1)) << 6;

	// Zero shared memmory
	for (int s = 32*warpId + laneId; s < 3*64*64; s += 32*W)
	    shmem[s] = 0;
	     
	__syncthreads();

	// Inner loop over TOD cache lines

	while (iterator.get_cl()) {
	    uint icl = iterator.icl;
	    uint s = (icl << 5) + laneId;  // FIXME 32-bit overflow

	    T ypix = xpointing[s];
	    T xpix = xpointing[s + nsamp];
	    T alpha = xpointing[s + 2*nsamp];
	    T t = tod[s];

	    // FIXME add 'status' argument, and calls to range_check_{xpix,ypix}().
	    normalize_xpix(xpix, nxpix);   // defined in gpu_mm2_internals.hpp

	    int iy0, iy1, ix0, ix1;
	    quantize_ypix(iy0, iy1, ypix, nypix);  // defined in gpu_mm2_internals.hpp
	    quantize_xpix(ix0, ix1, xpix, nxpix);  // defined in gpu_mm2_internals.hpp

	    T dy = ypix - iy0;
	    T dx = xpix - ix0;
	    
	    T q, u;
	    dtype<T>::xsincos(two*alpha, &u, &q);
	    q *= t;
	    u *= t;
	    
	    add_tqu(shmem, iy0, ix0, iy0_cell, ix0_cell, t, q, u, (one-dy) * (one-dx));
	    add_tqu(shmem, iy0, ix1, iy0_cell, ix0_cell, t, q, u, (one-dy) * (dx));
	    add_tqu(shmem, iy1, ix0, iy0_cell, ix0_cell, t, q, u, (dy) * (one-dx));
	    add_tqu(shmem, iy1, ix1, iy0_cell, ix0_cell, t, q, u, (dy) * (dx));
	}

	__syncthreads();
	
	// Shared -> global
	
	for (int y = warpId; y < 64; y += W) {
	    for (int x = laneId; x < 64; x += 32) {
		int ss = 64*y + x;                           // shared memory offset
		int sg = (iy0_cell+y)*nxpix + (ix0_cell+x);  // global memory offset

		T t = shmem[ss];
		if (!__reduce_or_sync(ALL_LANES, t != 0))
		    continue;

		atomicAdd(map + sg, t);
		atomicAdd(map + sg + nypix*nxpix, shmem[ss+64*64]);
		atomicAdd(map + sg + 2*nypix*nxpix, shmem[ss+2*64*64]);
	    }
	}

	__syncthreads();
    }
}


template<typename T>
void launch_tod2map2(T *map, const T *tod, const T *xpointing, const ulong *plan_mt, 
		     long nsamp, long nypix, long nxpix, int nmt, int nmt_per_block, bool debug)
{
    static constexpr int W = 16;  // warps per threadblock

    check_nsamp(nsamp, "launch_tod2map2");
    check_nypix(nypix, "launch_tod2map2");
    check_nxpix(nxpix, "launch_tod2map2");
    
    xassert(nmt > 0);
    xassert(nmt_per_block > 0);
    xassert((nmt_per_block % 32) == 0);

    int nblocks = (nmt + nmt_per_block - 1) / nmt_per_block;
    int shmem_nbytes = 3 * 64 * 64 * sizeof(T);
    
    if (debug) {
	tod2map2_kernel<T,W,true> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (map, tod, xpointing, plan_mt, nsamp, nypix, nxpix, nmt, nmt_per_block);
    }
    else {
	tod2map2_kernel<T,W,false> <<< nblocks, {32,W}, shmem_nbytes >>>
	    (map, tod, xpointing, plan_mt, nsamp, nypix, nxpix, nmt, nmt_per_block);
    }

    CUDA_PEEK("tod2map2 kernel launch");
}


#define INSTANTIATE(T) \
    template void launch_tod2map2(T *map, const T *tod, const T *xpointing, const ulong *plan_mt, \
				  long nsamp, long nypix, long nxpix, int nmt, int nmt_per_block, bool debug);

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
