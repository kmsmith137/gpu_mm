#include "../include/gpu_mm.hpp"
#include "../include/plan_iterator.hpp"

#include <cassert>
#include <ksgpu/cuda_utils.hpp>

using namespace ksgpu;

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
    int nypix_global,
    int nxpix_global,
    int nycells,
    int nxcells,
    long ystride,
    long polstride,
    uint nmt,
    uint nmt_per_block,
    bool periodic_xcoord,
    bool partial_pixelization,
    long lmap_size)   // for debugging
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
    pixel_locator<T> px(nypix_global, nxpix_global, periodic_xcoord);

    // Outer loop over map cells

    while (iterator.get_cell()) {
	uint icell = iterator.icell;
	uint iycell = icell >> 10;
	uint ixcell = icell & ((1<<10) - 1);

	bool valid = (iycell < nycells) && (ixcell < nxcells);	
	long offset = valid ? cell_offsets[iycell*nxcells + ixcell] : -1;
	err = ((offset >= 0) || partial_pixelization) ? err : errflag_not_in_pixelization;

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

	    // Locate pixel in shared memory.
	    px.locate(ypix, xpix, iycell, ixcell, err);
	    
	    add_tqu(shmem, px.iy0, px.ix0, t, q, u, (1-px.dy) * (1-px.dx));
	    add_tqu(shmem, px.iy0, px.ix1, t, q, u, (1-px.dy) * (px.dx));
	    add_tqu(shmem, px.iy1, px.ix0, t, q, u, (px.dy) * (1-px.dx));
	    add_tqu(shmem, px.iy1, px.ix1, t, q, u, (px.dy) * (px.dx));
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

		// Check for out-of-range memory access
		if constexpr (Debug) {
		    assert(polstride > 0);
		    assert(sg >= 0);
		    assert(sg + 2*polstride < lmap_size);
		}

		atomicAdd(lmap + sg, t);
		atomicAdd(lmap + sg + polstride, shmem[ss+64*64]);
		atomicAdd(lmap + sg + 2*polstride, shmem[ss+2*64*64]);
	    }
	}

	__syncthreads();
    }

    // No need for __syncthreads() before write_errflags(), since main loop has __syncthreads() at bottom.
    // Reminder: write_errflags() assumes thread layout is {32,W,1}, and block layout is {B,1,1}.
    write_errflags(errflags, (uint *)shmem, err);
}


template<typename T>
extern void launch_planned_tod2map(
    ksgpu::Array<T> &local_map,                 // total size (3 * local_pixelization.npix)
    const ksgpu::Array<T> &tod,                 // shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<T> &xpointing,           // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization, 
    const PointingPlan &plan,
    bool partial_pixelization,
    bool debug)
{
    static constexpr int W = 16;  // warps per threadblock
    static constexpr int shmem_nbytes = 3 * 64 * 64 * sizeof(T);

    check_tod(tod, plan.nsamp, "launch_tod2map", true);                      // on_gpu = true
    check_local_map(local_map, local_pixelization, "launch_tod2map", true);  // on_gpu = true
    check_xpointing(xpointing, plan.nsamp, "launch_tod2map", true);          // on_gpu = true

    // Verify consistency of (nypix, nxpix, periodic_xcoord) between plan and lppix    
    xassert_eq(local_pixelization.nypix_global, plan.nypix_global);
    xassert_eq(local_pixelization.nxpix_global, plan.nxpix_global);
    xassert_eq(local_pixelization.periodic_xcoord, plan.periodic_xcoord);
    
    if (debug) {
	tod2map_kernel<T,W,true> <<< plan.pp.pointing_nblocks, {32,W}, shmem_nbytes >>>
	    (local_map.data,                            // T *lmap
	     tod.data,                                  // const T *tod
	     xpointing.data,                            // const T *xpointing
	     local_pixelization.cell_offsets_gpu.data,  // const long *cell_offsets
	     plan.plan_mt,                              // const ulong *plan_mt
	     plan.err_gpu,                              // uint *errflags
	     plan.nsamp,                                // long nsamp
	     plan.nypix_global,                         // int nypix_global
	     plan.nxpix_global,                         // int nxpix_global
	     local_pixelization.nycells,                // int nycells
	     local_pixelization.nxcells,                // int nxcells
	     local_pixelization.ystride,                // long ystride
	     local_pixelization.polstride,              // long polstride
	     plan.pp.plan_nmt,                          // uint nmt
	     plan.pp.nmt_per_threadblock,               // uint nmt_per_block,
	     plan.periodic_xcoord,                      // bool periodic_xcoord
	     partial_pixelization,                      // bool partial_pixelization
	     local_map.size);                           // long lmap_size
    }
    else {
	tod2map_kernel<T,W,false> <<< plan.pp.pointing_nblocks, {32,W}, shmem_nbytes >>>
	    (local_map.data,                            // T *lmap
	     tod.data,                                  // const T *tod
	     xpointing.data,                            // const T *xpointing
	     local_pixelization.cell_offsets_gpu.data,  // const long *cell_offsets
	     plan.plan_mt,                              // const ulong *plan_mt
	     plan.err_gpu,                              // uint *errflags
	     plan.nsamp,                                // long nsamp
	     plan.nypix_global,                         // int nypix_global
	     plan.nxpix_global,                         // int nxpix_global
	     local_pixelization.nycells,                // int nycells
	     local_pixelization.nxcells,                // int nxcells
	     local_pixelization.ystride,                // long ystride
	     local_pixelization.polstride,              // long polstride
	     plan.pp.plan_nmt,                          // uint nmt
	     plan.pp.nmt_per_threadblock,               // uint nmt_per_block
	     plan.periodic_xcoord,                      // bool periodic_xcoord
	     partial_pixelization,                      // bool partial_pixelization
	     local_map.size);                           // long lmap_size
    }

    CUDA_PEEK("tod2map kernel launch");
    
    uint errflags_to_ignore = partial_pixelization ? errflag_not_in_pixelization : 0;
    check_gpu_errflags(plan.err_gpu, plan.pp.pointing_nblocks, "tod2map", errflags_to_ignore);
}


#define INSTANTIATE(T) \
    template void launch_planned_tod2map( \
	ksgpu::Array<T> &local_map, \
	const ksgpu::Array<T> &tod, \
	const ksgpu::Array<T> &xpointing, \
	const LocalPixelization &local_pixelization, \
	const PointingPlan &plan, \
	bool partial_pixelization, \
	bool debug)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
