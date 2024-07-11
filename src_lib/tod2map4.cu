#include <cassert>
#include <gputils/cuda_utils.hpp>
#include "../include/gpu_mm2.hpp"

using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif

static constexpr uint ALL_LANES = 0xffffffffU;


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


__global__ void tod2map4_kernel(
    float *map,                              // Shape (3, nypix, nxpix)   where axis 0 = {I,Q,U}
    const float *tod,                        // Shape (nsamp,)
    const float *xpointing,                  // Shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const ulong *plan_mt,                    // See long comment above. Shape (plan_ncltod,)
    const int *plan_quadruples,              // See long comment above. Shape (plan_nquadruples, 4)
    long nsamp,                              // Number of TOD samples (= detectors * times)
    int nypix,                                // Length of map declination axis
    int nxpix)                                 // Length of map RA axis
{
    __shared__ float shmem[3*64*64];

    const int laneId = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    
    // Read quadruple for this block.
    // (After this, we don't need the 'plan_quadruples' pointer any more.)
    
    plan_quadruples += 4 * blockIdx.x;
    int cell_idec = plan_quadruples[0];  // divisible by 64
    int cell_ira = plan_quadruples[1];   // divisible by 64
    int icl_start = plan_quadruples[2];
    int icl_end = plan_quadruples[3];

    // Shift map pointer to per-thread (not per-block) base location
    const int idec_base = cell_idec + (threadIdx.x >> 6);
    const int ira_base = cell_ira + (threadIdx.x & 63);
    map += long(idec_base) * long(nxpix) + ira_base;
        
    // Read global memory -> shared.
    // Assumes blockIdx.x is a multiple of 64.

    const long npix = long(nypix) * long(nxpix);    
    const int spix = (blockDim.x >> 6) * nxpix;  // Global memory "stride" in loop below
    	
    do {
	const float *m = map;
	for (int s = threadIdx.x; s < 64*64; s += blockDim.x) {
	    shmem[s] = m[0];
	    shmem[s + 64*64] = m[npix];
	    shmem[s + 2*64*64] = m[2*npix];
	    m += spix;
	}
    } while (0);
    
    __syncthreads();

    int cltod_rb = 0;
    int icl_rb = -1;
    
    for (int icl = icl_start + warpId; icl < icl_end; icl += nwarps) {
	int icl_base = icl & ~31;
	
	if (icl_rb != icl_base) {
	    icl_rb = icl_base;
	    ulong mt = plan_mt[icl_rb + laneId];
	    cltod_rb = int(mt >> 20);

	    // bool valid = ((icl_rb + laneId >= icl_start) && (icl_rb + laneId < icl_end));
	    // uint iy0_cell = ((mt >> 10) & ((1<<10) - 1)) << 6;
	    // uint ix0_cell = (mt & ((1<<10) - 1)) << 6;
	    // assert(!valid || (iy0_cell == cell_idec));
	    // assert(!valid || (ix0_cell == cell_ira));
	}
	
	// Value of 'cltod' is the same on each thread.
	int cltod = __shfl_sync(ALL_LANES, cltod_rb, icl & 31);

	// By convention, negative cltods are allowed, but ignored.
	if (cltod < 0)
	    continue;

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
	
	// assert(idec >= 0);
	// assert(idec < nypix-1);
	// assert(ira >= 0);
	// assert(ira < nxpix-1);	    
	
	update_shmem(shmem, idec,   ira,   cell_idec, cell_ira, cos_2a, sin_2a, x * (1.0f-ddec) * (1.0f-dra));
	update_shmem(shmem, idec,   ira+1, cell_idec, cell_ira, cos_2a, sin_2a, x * (1.0f-ddec) * (dra));
	update_shmem(shmem, idec+1, ira,   cell_idec, cell_ira, cos_2a, sin_2a, x * (ddec) * (1.0f-dra));
	update_shmem(shmem, idec+1, ira+1, cell_idec, cell_ira, cos_2a, sin_2a, x * (ddec) * (dra));	    
    }

    
    __syncthreads();

    // Write shared memory -> global
    // Assumes blockIdx.x is a multiple of 64.
    
    do {
	float *m = map;
	for (int s = threadIdx.x; s < 64*64; s += blockDim.x) {
	    m[0] = shmem[s];
	    m[npix] = shmem[s + 64*64];
	    m[2*npix] = shmem[s + 2*64*64];
	    m += spix;
	}
    } while (0);
}


void launch_tod2map4(
    gputils::Array<float> &map,                  // Shape (3, nypix, nxpix)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &tod,            // Shape (nsamp,)
    const gputils::Array<float> &xpointing,      // Shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const gputils::Array<ulong> &plan_mt,        // Shape (plan_ncltod,)
    const gputils::Array<int> &plan_quadruples)  // Shape (plan_nquadruples, 4)
{
    long nsamp, nypix, nxpix;
    
    check_tod_and_init_nsamp(tod, nsamp, "launch_tod2map4", true);        // on_gpu=true
    check_map_and_init_npix(map, nypix, nxpix, "launch_tod2map4", true);  // on_gpu=true
    check_xpointing(xpointing, nsamp, "launch_tod2map4", true);           // on_gpu
    
    xassert(plan_mt.ndim == 1);
    xassert(plan_mt.is_fully_contiguous());

    xassert(plan_quadruples.ndim == 2);
    xassert(plan_quadruples.shape[1] == 4);
    xassert(plan_quadruples.is_fully_contiguous());
    
    int nblocks = plan_quadruples.shape[0];
    
    tod2map4_kernel<<< nblocks, 512 >>>
	(map.data, tod.data, xpointing.data, plan_mt.data, plan_quadruples.data, nsamp, nypix, nxpix);
    
    CUDA_PEEK("tod2map4_kernel");
}


}  // namespace gpu_mm
