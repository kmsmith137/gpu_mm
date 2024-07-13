#include "../include/gpu_mm.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


void reference_map2tod(Array<float> &tod, const Array<float> &map, const Array<float> &xpointing)
{
    long nsamp, ndec, nra;
    check_tod_and_init_nsamp(tod, nsamp, "reference_map2tod", false);     // on_gpu=false
    check_map_and_init_npix(map, ndec, nra, "reference_map2tod", false);  // on_gpu=false
    check_xpointing(xpointing, nsamp, "reference_map2tod", false);        // on_gpu=false

    long npix = long(ndec) * long(nra);

    for (long s = 0; s < nsamp; s++) {
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
	
	long ipix = idec*nra + ira;
	float out = 0.0;
	
	float w = (1.0-ddec) * (1.0-dra);
	out += w * map.data[ipix];
	out += w * cos_2a * map.data[ipix+npix];
	out += w * sin_2a * map.data[ipix+2*npix];

	w = (1.0-ddec) * (dra);
	out += w * map.data[ipix + 1];
	out += w * cos_2a * map.data[ipix+npix + 1];
	out += w * sin_2a * map.data[ipix+2*npix + 1];
	
	w = ddec * (1.0 - dra);
	out += w * map.data[ipix + nra];
	out += w * cos_2a * map.data[ipix+npix + nra];
	out += w * sin_2a * map.data[ipix+2*npix + nra];
	
	w = ddec * dra;
	out += w * map.data[ipix + nra+1];
	out += w * cos_2a * map.data[ipix+npix + nra+1];
	out += w * sin_2a * map.data[ipix+2*npix + nra+1];

	tod.data[s] = out;
    }
}


}  // namespace gpu_mm
