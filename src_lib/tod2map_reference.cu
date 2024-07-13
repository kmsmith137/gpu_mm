#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"  // ALL_LANES
#include <gputils/cuda_utils.hpp>

using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// Helper function called by reference_tod2map()
inline void update_map(float *map, long ipix, long npix, float cos_2a, float sin_2a, float t)
{
    xassert((ipix >= 0) && (ipix < npix));
    
    map[ipix] += t;
    map[ipix+npix] += t * cos_2a;
    map[ipix+2*npix] += t * sin_2a;
}


void reference_tod2map(Array<float> &map, const Array<float> &tod, const Array<float> &xpointing)
{
    long nsamp, ndec, nra;
    check_tod_and_init_nsamp(tod, nsamp, "reference_tod2map", false);     // on_gpu=false
    check_map_and_init_npix(map, ndec, nra, "reference_tod2map", false);  // on_gpu=false
    check_xpointing(xpointing, nsamp, "reference_tod2map", false);        // on_gpu=false

    long npix = long(ndec) * long(nra);

    // No memset(out, ...) here, since we want to accumulate (not overwrite) output.
    
    for (long s = 0; s < nsamp; s++) {
	float x = tod.data[s];
	float px_dec = xpointing.data[s];
	float px_ra = xpointing.data[s + nsamp];
	float alpha = xpointing.data[s + 2*nsamp];
	
	float cos_2a = cosf(2*alpha);
	float sin_2a = sinf(2*alpha);

	int idec = int(px_dec);
	int ira = int(px_ra);
	float ddec = px_dec - float(idec);
	float dra = px_ra - float(ira);
	
	xassert(idec >= 0);
	xassert(idec < ndec-1);
	xassert(ira >= 0);
	xassert(ira < nra-1);
	
	long ipix = long(idec) * long(nra) + ira;

	update_map(map.data, ipix,       npix, cos_2a, sin_2a, x * (1.0-ddec) * (1.0-dra));
	update_map(map.data, ipix+1,     npix, cos_2a, sin_2a, x * (1.0-ddec) * (dra));
	update_map(map.data, ipix+nra,   npix, cos_2a, sin_2a, x * (ddec) * (1.0-dra));
	update_map(map.data, ipix+nra+1, npix, cos_2a, sin_2a, x * (ddec) * (dra));
    }
}


}  // namespace gpu_mm
