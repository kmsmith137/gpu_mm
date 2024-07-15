#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"  // ALL_LANES
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// Helper function called by reference_tod2map()
template<typename T>
inline void update_map(const LocalPixelization &lpix, T *lmap, int iy, int ix, T cos_2a, T sin_2a, T t, bool allow_outlier_pixels)
{
    int iycell = iy >> 6;
    int ixcell = ix >> 6;
    
    if ((iy < 0) || (iycell >= lpix.nycells) || (ix < 0) || (ixcell >= lpix.nxcells)) {
	if (_unlikely(!allow_outlier_pixels))
	    throw runtime_error("reference_tod2map: pixel is out of range, and allow_outlier_pixels=false");
	return;
    }

    long offset = lpix.cell_offsets_cpu.data[iycell*lpix.nxcells + ixcell];

    if (offset < 0) {
	if (_unlikely(!allow_outlier_pixels))
	    throw runtime_error("reference_tod2map: pixel is out of range, and allow_outlier_pixels=false");
	return;
    }

    offset += (ix - (ixcell << 6));
    offset += (iy - (iycell << 6)) * lpix.ystride;

    lmap[offset] += t;
    lmap[offset + lpix.polstride] += t * cos_2a;
    lmap[offset + 2*lpix.polstride] += t * sin_2a;
}


template<typename T>
void reference_tod2map(
    Array<T> &local_map,
    const Array<T> &tod,
    const Array<T> &xpointing,
    const LocalPixelization &local_pixelization,
    bool allow_outlier_pixels)
{
    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "reference_tod2map", false);            // on_gpu = false
    check_local_map(local_map, local_pixelization, "reference_tod2map", false);  // on_gpu = false
    check_xpointing(xpointing, nsamp, "reference_tod2map", false);               // on_gpu = false

    int nypix = local_pixelization.nycells << 6;
    int nxpix = local_pixelization.nxcells << 6;
    
    for (long s = 0; s < nsamp; s++) {
	T ypix = xpointing.data[s];
	T xpix = xpointing.data[s + nsamp];
	T alpha = xpointing.data[s + 2*nsamp];
	T t = tod.data[s];

 	T cos_2a, sin_2a;
	dtype<T>::xsincos(2*alpha, &sin_2a, &cos_2a);
	
	// quantize_pixel() is defined in gpu_mm_internals.hpp
	int iy = quantize_pixel(ypix, nypix);   // satisfies -2 <= iy <= nypix
	int ix = quantize_pixel(xpix, nxpix);   // satisfies -2 <= ix <= nxpix
	
	T dy = ypix - T(iy);
	T dx = xpix - T(ix);

	update_map(local_pixelization, local_map.data, iy,   ix,   cos_2a, sin_2a, t * (1-dy) * (1-dx), allow_outlier_pixels);
	update_map(local_pixelization, local_map.data, iy,   ix+1, cos_2a, sin_2a, t * (1-dy) * (dx),   allow_outlier_pixels);
	update_map(local_pixelization, local_map.data, iy+1, ix,   cos_2a, sin_2a, t * (dy) * (1-dx),   allow_outlier_pixels);
	update_map(local_pixelization, local_map.data, iy+1, ix+1, cos_2a, sin_2a, t * (dy) * (dx),     allow_outlier_pixels);
    }
}


#define INSTANTIATE(T) \
    template void reference_tod2map( \
	Array<T> &local_map, \
	const Array<T> &tod, \
	const Array<T> &xpointing, \
	const LocalPixelization &local_pixelization, \
	bool allow_outlier_pixels)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
