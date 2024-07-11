#include "../include/gpu_mm.hpp"

#include <cassert>
#include <gputils/cuda_utils.hpp>

using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// For __shfl_sync()
static constexpr unsigned int ALL_LANES = 0xffffffff;


static void _check_tod2map_args(float *map, const float *tod, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    assert(tod != nullptr);
    assert(map != nullptr);
    assert(xpointing != nullptr);
    
    assert(ndet > 0);
    assert(nt > 0);
    assert(ndec > 0);
    assert(nra > 0);

    assert((nt % 32) == 0);
    assert((ndec % 64) == 0);
    assert((nra % 64) == 0);
}


static void _check_tod2map_args(Array<float> &map, const Array<float> &tod, const Array<float> &xpointing)
{
    assert(tod.ndim == 2);
    assert(tod.is_fully_contiguous());
    
    assert(map.ndim == 3);
    assert(map.shape[0] == 3);
    assert(map.is_fully_contiguous());
    
    assert(xpointing.ndim == 3);
    assert(xpointing.shape[0] == 3);
    assert(xpointing.shape[1] == tod.shape[0]);
    assert(xpointing.shape[2] == tod.shape[1]);
    assert(xpointing.is_fully_contiguous());
}

static void _check_tod2map_plan(const int *plan_cltod_list, const int *plan_quadruples, int plan_ncltod, int plan_nquadruples)
{
    assert(plan_cltod_list != nullptr);
    assert(plan_quadruples != nullptr);
    assert(plan_nquadruples > 0);
    assert(plan_ncltod > 0);
}


static void _check_tod2map_plan(const Array<int> &plan_cltod_list, const Array<int> &plan_quadruples)
{
    assert(plan_cltod_list.ndim == 1);
    assert(plan_cltod_list.is_fully_contiguous());

    assert(plan_quadruples.ndim == 2);
    assert(plan_quadruples.shape[1] == 4);
    assert(plan_quadruples.is_fully_contiguous());
}


// -------------------------------------------------------------------------------------------------
//
// reference_tod2map(), take 1: without a plan.



// Helper function called by reference_tod2map()
inline void update_map(float *map, long ipix, long npix, float cos_2a, float sin_2a, float t)
{
    assert((ipix >= 0) && (ipix < npix));
    
    map[ipix] += t;
    map[ipix+npix] += t * cos_2a;
    map[ipix+2*npix] += t * sin_2a;
}


void reference_tod2map(float *map, const float *tod, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    _check_tod2map_args(map, tod, xpointing, ndet, nt, ndec, nra);

    // A "sample" is a (detector, time) pair.
    long ns = long(ndet) * long(nt);
    long npix = long(ndec) * long(nra);

    // No memset(out, ...) here, since we want to accumulate (not overwrite) output.
    
    for (long s = 0; s < ns; s++) {
	float x = tod[s];
	float px_dec = xpointing[s];
	float px_ra = xpointing[s + ns];
	float alpha = xpointing[s + 2*ns];
	
	float cos_2a = cosf(2*alpha);
	float sin_2a = sinf(2*alpha);

	int idec = int(px_dec);
	int ira = int(px_ra);
	float ddec = px_dec - float(idec);
	float dra = px_ra - float(ira);
	
	assert(idec >= 0);
	assert(idec < ndec-1);
	assert(ira >= 0);
	assert(ira < nra-1);
	
	long ipix = long(idec) * long(nra) + ira;

	update_map(map, ipix,       npix, cos_2a, sin_2a, x * (1.0-ddec) * (1.0-dra));
	update_map(map, ipix+1,     npix, cos_2a, sin_2a, x * (1.0-ddec) * (dra));
	update_map(map, ipix+nra,   npix, cos_2a, sin_2a, x * (ddec) * (1.0-dra));
	update_map(map, ipix+nra+1, npix, cos_2a, sin_2a, x * (ddec) * (dra));
    }
}


void reference_tod2map(Array<float> &map, const Array<float> &tod, const Array<float> &xpointing)
{
    assert(map.on_host());
    assert(tod.on_host());
    assert(xpointing.on_host());
    
    _check_tod2map_args(map, tod, xpointing);
    
    reference_tod2map(map.data, tod.data, xpointing.data, tod.shape[0], tod.shape[1], map.shape[1], map.shape[2]);
}


// -------------------------------------------------------------------------------------------------
//
// reference_tod2map(), take 2: with a plan.



// Helper function called by reference_tod2map()
inline void update_map(float *map, int idec, int ira, int cell_idec, int cell_ira, int ndec, int nra, float cos_2a, float sin_2a, float t)
{
    bool dec_in_cell = (idec >= cell_idec) && (idec < cell_idec+64);
    bool ra_in_cell = (ira >= cell_ira) && (ira < cell_ira+64);

    if (!dec_in_cell || !ra_in_cell)
	return;

    long npix = long(ndec) * long(nra);
    long ipix = long(idec) * long(nra) + ira;
    
    map[ipix] += t;
    map[ipix+npix] += t * cos_2a;
    map[ipix+2*npix] += t * sin_2a;
}


void reference_tod2map(
    float *map,                            // shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const float *tod,                      // shape (ndet, nt)
    const float *xpointing,                // shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const int *plan_cltod_list,            // shape (plan_ncltod,)
    const int *plan_quadruples,            // shape (plan_nquadruples, 4)
    int plan_ncltod,                       // defines length of plan_cltod_list[] array
    int plan_nquadruples,                  // defines length of plan_quadruples[] array
    int ndet,                              // number of detectors
    int nt,                                // number of time samples per detector
    int ndec,                              // nength of map declination axis
    int nra)                               // nength of map RA axis
{
    _check_tod2map_args(map, tod, xpointing, ndet, nt, ndec, nra);
    _check_tod2map_plan(plan_cltod_list, plan_quadruples, plan_ncltod, plan_nquadruples);

    // A "sample" is a (detector, time) pair.
    long ns = long(ndet) * long(nt);

    // No memset(out, ...) here, since we want to accumulate (not overwrite) output.
    
    for (int q = 0; q < plan_nquadruples; q++) {
	int cell_idec = plan_quadruples[4*q];
	int cell_ira = plan_quadruples[4*q+1];
	int icl_istart = plan_quadruples[4*q+2];
	int icl_iend = plan_quadruples[4*q+3];
	
	assert(cell_idec >= 0);
	assert((cell_idec % 64) == 0);
	assert((cell_idec + 64) <= ndec);
	
	assert(cell_ira >= 0);
	assert((cell_ira % 64) == 0);
	assert((cell_ira + 64) <= nra);

	// Note that we don't allow (icl_istart == icl_iend),
	// since this probably indicates a bug in plan creation.
	
	assert(icl_istart >= 0);
	assert(icl_istart < icl_iend);
	assert(icl_iend <= plan_ncltod);
	
	for (int icl = icl_istart; icl < icl_iend; icl++) {
	    int cltod = plan_cltod_list[icl];

	    // By convention, negative cltods are allowed, but ignored.
	    if (cltod < 0)
		continue;

	    long s0 = 32 * long(cltod);
	    assert(s0+32 <= ns);

	    for (long s = s0; s < s0+32; s++) {
		float x = tod[s];
		float px_dec = xpointing[s];
		float px_ra = xpointing[s + ns];
		float alpha = xpointing[s + 2*ns];
		
		float cos_2a = cosf(2*alpha);
		float sin_2a = sinf(2*alpha);
		
		int idec = int(px_dec);
		int ira = int(px_ra);
		float ddec = px_dec - float(idec);
		float dra = px_ra - float(ira);
	
		assert(idec >= 0);
		assert(idec < ndec-1);
		assert(ira >= 0);
		assert(ira < nra-1);

		update_map(map, idec,   ira,   cell_idec, cell_ira, ndec, nra, cos_2a, sin_2a, x * (1.0-ddec) * (1.0-dra));
		update_map(map, idec,   ira+1, cell_idec, cell_ira, ndec, nra, cos_2a, sin_2a, x * (1.0-ddec) * (dra));
		update_map(map, idec+1, ira,   cell_idec, cell_ira, ndec, nra, cos_2a, sin_2a, x * (ddec) * (1.0-dra));
		update_map(map, idec+1, ira+1, cell_idec, cell_ira, ndec, nra, cos_2a, sin_2a, x * (ddec) * (dra));
	    }
	}
    }
}



void reference_tod2map(Array<float> &map, const Array<float> &tod, const Array<float> &xpointing,
		       const Array<int> &plan_cltod_list, const Array<int> &plan_quadruples)
{
    assert(map.on_host());
    assert(tod.on_host());
    assert(xpointing.on_host());
    assert(plan_cltod_list.on_host());
    assert(plan_quadruples.on_host());
    
    _check_tod2map_args(map, tod, xpointing);
    _check_tod2map_plan(plan_cltod_list, plan_quadruples);

    reference_tod2map(map.data, tod.data, xpointing.data,
		      plan_cltod_list.data, plan_quadruples.data,
		      plan_cltod_list.shape[0], plan_quadruples.shape[0],
		      tod.shape[0], tod.shape[1],
		      map.shape[1], map.shape[2]);
}


// -------------------------------------------------------------------------------------------------
//
// GPU tod2map


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


__global__ void tod2map_kernel(
    float *map,                              // Shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const float *tod,                        // Shape (ndet, nt)
    const float *xpointing,                  // Shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const int *plan_cltod_list,              // See long comment above. Shape (plan_ncltod,)
    const int *plan_quadruples,              // See long comment above. Shape (plan_nquadruples, 4)
    long nsamp,                              // Number of TOD samples (= detectors * times)
    int ndec,                                // Length of map declination axis
    int nra)                                 // Length of map RA axis
{
    __shared__ float shmem[3*64*64];
    
    // Read quadruple for this block.
    // (After this, we don't need the 'plan_quadruples' pointer any more.)
    
    plan_quadruples += 4 * blockIdx.x;
    int cell_idec = plan_quadruples[0];  // divisible by 64
    int cell_ira = plan_quadruples[1];   // divisible by 64
    int icl_start = plan_quadruples[2];
    int icl_end = plan_quadruples[3];

    // Shift values of (plan_cltod_list, icl_start, icl_end), so that 0 <= icl_start < 32.
    // The values of (icl_start, icl_end) are the same on all threads.
    int icl_sbase = icl_start & ~31;
    plan_cltod_list += icl_sbase;
    icl_start -= icl_sbase;
    icl_end -= icl_sbase;

    // Shift map pointer to per-thread (not per-block) base location
    const int idec_base = cell_idec + (threadIdx.x >> 6);
    const int ira_base = cell_ira + (threadIdx.x & 63);
    map += long(idec_base) * long(nra) + ira_base;
        
    // Read global memory -> shared.
    // Assumes blockIdx.x is a multiple of 64.

    const long npix = long(ndec) * long(nra);    
    const int spix = (blockDim.x >> 6) * nra;  // Global memory "stride" in loop below
    	
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

    // Outer loop over batches of 32 TOD cache lines.
    // The value of 'icl_warp' is the same on each thread.
    
    const int laneId = threadIdx.x & 31;
    
    for (int icl_warp = (threadIdx.x & ~31); icl_warp < icl_end; icl_warp += blockDim.x) {
	// Value of 'cltod_outer' is different on each thread.
	int cltod_outer = plan_cltod_list[icl_warp + laneId];

	// Values of (icl0, icl1) are the same on each thread.
	int icl0 = max(icl_warp, icl_start);
	int icl1 = min(icl_warp+32, icl_end);
	
	// Inner loop over TOD cache lines ('cltod')
	// The value of 'icl' is the same on each thread.
	
	for (int icl = icl0; icl < icl1; icl++) {
	    // Value of 'cltod' is the same on each thread.
	    int cltod = __shfl_sync(ALL_LANES, cltod_outer, icl & 31);

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
	    // assert(idec < ndec-1);
	    // assert(ira >= 0);
	    // assert(ira < nra-1);	    

	    update_shmem(shmem, idec,   ira,   cell_idec, cell_ira, cos_2a, sin_2a, x * (1.0f-ddec) * (1.0f-dra));
	    update_shmem(shmem, idec,   ira+1, cell_idec, cell_ira, cos_2a, sin_2a, x * (1.0f-ddec) * (dra));
	    update_shmem(shmem, idec+1, ira,   cell_idec, cell_ira, cos_2a, sin_2a, x * (ddec) * (1.0f-dra));
	    update_shmem(shmem, idec+1, ira+1, cell_idec, cell_ira, cos_2a, sin_2a, x * (ddec) * (dra));	    
	}
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


void launch_tod2map(
    gputils::Array<float> &map,                  // Shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &tod,            // Shape (ndet, nt)
    const gputils::Array<float> &xpointing,      // Shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const gputils::Array<int> &plan_cltod_list,  // Shape (plan_ncltod,)
    const gputils::Array<int> &plan_quadruples)  // Shape (plan_nquadruples, 4)
{
    _check_tod2map_args(map, tod, xpointing);
    _check_tod2map_plan(plan_cltod_list, plan_quadruples);
    
    assert(map.on_gpu());
    assert(tod.on_gpu());
    assert(xpointing.on_gpu());
    assert(plan_cltod_list.on_gpu());
    assert(plan_quadruples.on_gpu());
    
    int nblocks = plan_quadruples.shape[0];
    long nsamp = tod.shape[0] * tod.shape[1];
    int ndec = map.shape[1];
    int nra = map.shape[2];
    
    tod2map_kernel<<< nblocks, 512 >>>
	(map.data, tod.data, xpointing.data, plan_cltod_list.data, plan_quadruples.data, nsamp, ndec, nra);
    
    CUDA_PEEK("tod2map_kernel");
}


}  // namespace gpu_mm
