#include "../include/gpu_mm.hpp"
#include <cassert>

using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// reference_tod2map(): slow single-threaded CPU tod2map, for testing.


// Helper function called by reference_tod2map()
inline void update_map(float *map, int ndec, int nra,         // map in global memory
		       int cell_idec, int cell_ira,           // base coords of current cell
		       int idec, int ira,                     // coords of data to be added
		       float cos_2a, float sin_2a, float t)   // data to be added
{
    bool dec_in_cell = (idec >= cell_idec) && (idec < cell_idec+64);
    bool ra_in_cell = (ira >= cell_ira) && (idec < cell_ira+64);

    if (!dec_in_cell || !ra_in_cell)
	return;

    long npix = long(ndec) * long(nra);
    
    map[idec*nra + ira] += t;
    map[idec*nra + ira + npix] += t * cos_2a;
    map[idec*nra + ira + 2*npix] += t * sin_2a;
}


void reference_tod2map(
    float *map,                            // shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const float *tod,                      // shape (ndet, nt)
    const float *xpointing,                // shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const unsigned int *plan_cltod_list,   // shape (plan_ncltod,)
    const unsigned int *plan_quadruples,   // shape (plan_nquadruples, 4)
    int plan_ncltod,                       // defines length of plan_cltod_list[] array
    int plan_nquadruples,                  // defines length of plan_quadruples[] array
    int ndet,                              // number of detectors
    int nt,                                // number of time samples per detector
    int ndec,                              // nength of map declination axis
    int nra)                               // nength of map RA axis
{
    assert(map != nullptr);
    assert(tod != nullptr);
    assert(xpointing != nullptr);
    assert(plan_cltod_list != nullptr);
    assert(plan_quadruples != nullptr);

    assert(plan_nquadruples > 0);
    assert(ndet > 0);
    assert(nt > 0);
    assert(ndec > 0);
    assert(nra > 0);

    // These are limitations of "version 0" and may go away in the future.
    assert((ndec % 64) == 0);
    assert((nra % 64) == 0);
    assert((nt % 32) == 0);

    // A "sample" is a (detector, time) pair.
    long nsamp = long(ndet) * long(nt);

    // This version of tod2map() overwites the existing map.
    long npix = long(ndec) * long(nra);
    memset(map, 0, 3 * npix * sizeof(float));
    
    for (int q = 0; q < plan_nquadruples; q++) {
	int cell_idec = plan_quadruples[4*q];
	int cell_ira = plan_quadruples[4*q+1];
	int cltod_list_istart = plan_quadruples[4*q+2];
	int cltod_list_iend = plan_quadruples[4*q+3];
	
	assert(cell_idec >= 0);
	assert((cell_idec % 64) == 0);
	assert((cell_idec + 64) <= ndec);
	
	assert(cell_ira >= 0);
	assert((cell_ira % 64) == 0);
	assert((cell_ira + 64) <= ndec);

	// Note that we don't allow (cltod_list_istart == cltod_list_iend),
	// since this probably indicates a bug in plan creation.
	
	assert(cltod_list_istart >= 0);
	assert(cltod_list_istart < cltod_list_iend);
	assert(cltod_list_iend <= plan_ncltod);
	
	for (int c = cltod_list_istart; c < cltod_list_iend; c++) {
	    int cltod = plan_cltod_list[c];

	    // By convention, negative cltods are allowed, but ignored.
	    // (This is convenient in the GPU kernel.)
	    
	    if (cltod < 0)
		continue;

	    long samp0 = 32 * long(cltod);
	    assert(samp0 <= nsamp);

	    for (long s = samp0; s < samp0+32; s++) {
		float x = tod[s];
		float px_dec = xpointing[s];
		float px_ra = xpointing[s + nsamp];
		float alpha = xpointing[s + 2*nsamp];
		
		float cos_2a = cosf(2*alpha);
		float sin_2a = sinf(2*alpha);

		int idec = int(px_dec);
		int ira = int(px_ra);
		float ddec = px_dec - float(idec);
		float dra = px_ra - float(ira);

		update_map(map, ndec, nra, cell_idec, cell_ira, idec,   ira,   cos_2a, sin_2a, x * (1.0-ddec) * (1.0-dra));
		update_map(map, ndec, nra, cell_idec, cell_ira, idec,   ira+1, cos_2a, sin_2a, x * (1.0-ddec) * (dra));
		update_map(map, ndec, nra, cell_idec, cell_ira, idec+1, ira,   cos_2a, sin_2a, x * (ddec) * (1.0-dra));
		update_map(map, ndec, nra, cell_idec, cell_ira, idec+1, ira+1, cos_2a, sin_2a, x * (ddec) * (dra));
	    }
	}
    }
}


}  // namespace gpu_mm
