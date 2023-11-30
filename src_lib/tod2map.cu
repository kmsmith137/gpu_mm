#include "../include/gpu_mm.hpp"
#include <cassert>

using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


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

    // This version of tod2map() overwites the existing map.
    long npix = long(ndec) * long(nra);
    memset(map, 0, 3 * npix * sizeof(float));

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

    // This version of tod2map() overwites the existing map.
    long npix = long(ndec) * long(nra);
    memset(map, 0, 3 * npix * sizeof(float));
    
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
	    // (This is convenient in the GPU kernel.)
	    
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


}  // namespace gpu_mm
