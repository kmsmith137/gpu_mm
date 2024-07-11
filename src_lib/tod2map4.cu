#include <cassert>
#include <gputils/cuda_utils.hpp>

#include "../include/gpu_mm2.hpp"
#include "../include/PlanIterator2.hpp"

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// Helper function called by tod2map_kernel()
__device__ void update_shmem(float *shmem, int idec, int ira, int cell_idec, int cell_ira, float cos_2a, float sin_2a, float t)
{
    bool dec_in_cell = ((idec & ~63) == cell_idec);
    bool ra_in_cell = ((ira & ~63) == cell_ira);
    int s = ((idec & 63) << 6) | (ira & 63);

    // Warp divergence here
    if (dec_in_cell && ra_in_cell) {
	atomicAdd(shmem + s, t);
	atomicAdd(shmem + s + 64*64, t * cos_2a);
	atomicAdd(shmem + s + 2*64*64, t * sin_2a);
    }

    // FIXME is this a good idea?
    // __syncwarp();
}

template<int W, bool Debug>
__global__ void __launch_bounds__(32*W, 1)
tod2map4_kernel(
    float *map,                              // Shape (3, nypix, nxpix)   where axis 0 = {I,Q,U}
    const float *tod,                        // Shape (nsamp,)
    const float *xpointing,                  // Shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const ulong *plan_mt,                    // See long comment above. Shape (plan_ncltod,)
    long nsamp,                              // Number of TOD samples (= detectors * times)
    int nypix,                               // Length of map declination axis
    int nxpix,                               // Length of map RA axis
    uint nmt,
    uint nmt_per_block)
{
    __shared__ float shmem[3*64*64];
    
    // Threadblock has shape (32,W), so threadIdx.x is the laneId, and threadIdx.y is the warpId.
    const uint laneId = threadIdx.x;
    const uint warpId = threadIdx.y;

    PlanIterator2<W,Debug> iterator(plan_mt, nmt, nmt_per_block);

    while (iterator.get_cell()) {
	uint icell = iterator.icell;
	int iy0_cell = (icell >> 10) << 6;
	int ix0_cell = (icell & ((1<<10) - 1)) << 6;
	
	// Zero shared memmory
	for (int s = 32*warpId + laneId; s < 3*64*64; s += 32*W)
	    shmem[s] = 0;
    
	__syncthreads();

	while (iterator.get_cl()) {
	    // Value of 'cltod' is the same on each thread.
	    int cltod = iterator.icl;
	    
	    long s = (long(cltod) << 5) + laneId;
	    float x = tod[s];
	    float px_dec = xpointing[s];
	    float px_ra = xpointing[s + nsamp];
	    float alpha = xpointing[s + 2*nsamp];
	    
	    float cos_2a = cosf(2.0f * alpha);
	    float sin_2a = sinf(2.0f * alpha);
	    
	    int idec = int(px_dec);
	    int ira = int(px_ra);
	    float ddec = px_dec - float(idec);
	    float dra = px_ra - float(ira);
	    
	    if (Debug) {
		assert(idec >= 0);
		assert(idec < nypix-1);
		assert(ira >= 0);
		assert(ira < nxpix-1);
	    }
	    
	    update_shmem(shmem, idec,   ira,   iy0_cell, ix0_cell, cos_2a, sin_2a, x * (1.0f-ddec) * (1.0f-dra));
	    update_shmem(shmem, idec,   ira+1, iy0_cell, ix0_cell, cos_2a, sin_2a, x * (1.0f-ddec) * (dra));
	    update_shmem(shmem, idec+1, ira,   iy0_cell, ix0_cell, cos_2a, sin_2a, x * (ddec) * (1.0f-dra));
	    update_shmem(shmem, idec+1, ira+1, iy0_cell, ix0_cell, cos_2a, sin_2a, x * (ddec) * (dra));	    
	}
    
	__syncthreads();
	
	// Shared -> global
	
	for (int y = warpId; y < 64; y += W) {
	    for (int x = laneId; x < 64; x += 32) {
		int ss = 64*y + x;                            // shared memory offset
		int sg = (iy0_cell+y)*nxpix + (ix0_cell+x);  // global memory offset
		
		float t = shmem[ss];
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


void launch_tod2map4(
    gputils::Array<float> &map,                  // Shape (3, nypix, nxpix)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &tod,            // Shape (nsamp,)
    const gputils::Array<float> &xpointing,      // Shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const gputils::Array<ulong> &plan_mt,        // Shape (plan_nmt,)
    int nmt_per_block)
{
    static constexpr int W = 16;
    static constexpr bool Debug = false;
    
    long nsamp, nypix, nxpix;
    
    check_tod_and_init_nsamp(tod, nsamp, "launch_tod2map4", true);        // on_gpu=true
    check_map_and_init_npix(map, nypix, nxpix, "launch_tod2map4", true);  // on_gpu=true
    check_xpointing(xpointing, nsamp, "launch_tod2map4", true);           // on_gpu
    
    xassert(plan_mt.ndim == 1);
    xassert(plan_mt.is_fully_contiguous());

    int nmt = plan_mt.size;
    int nblocks = (nmt + nmt_per_block - 1) / nmt_per_block;
    // cout << "XXX nmt_per_block=" << nmt_per_block << ", nblocks=" << nblocks << endl;
    
    tod2map4_kernel<W,Debug> <<< nblocks, {32,W} >>>
	(map.data, tod.data, xpointing.data, plan_mt.data, nsamp, nypix, nxpix, nmt, nmt_per_block);
    
    CUDA_PEEK("tod2map4_kernel");
}


}  // namespace gpu_mm
