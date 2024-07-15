#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"  // dtype<T>::xsincos()

#include <iostream>
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// Helper called by reference_map2tod()
template<typename T>
inline T eval_map(const LocalPixelization &lpix, const T *lmap, T cos_2a, T sin_2a, int iy, int ix, bool allow_outlier_pixels)
{
    int iycell = iy >> 6;
    int ixcell = ix >> 6;
    
    if ((iy < 0) || (iycell >= lpix.nycells) || (ix < 0) || (ixcell >= lpix.nxcells)) {
	if (_unlikely(!allow_outlier_pixels))
	    throw runtime_error("reference_map2tod: pixel is out of range, and allow_outlier_pixels=false");
	return 0;
    }

    long offset = lpix.cell_offsets_cpu.data[iycell*lpix.nxcells + ixcell];

    if (offset < 0) {
	if (_unlikely(!allow_outlier_pixels))
	    throw runtime_error("reference_map2tod: pixel is out of range, and allow_outlier_pixels=false");
	return 0;
    }

    offset += (ix - (ixcell << 6));
    offset += (iy - (iycell << 6)) * lpix.ystride;

    T ret = lmap[offset];
    ret += cos_2a * lmap[offset + lpix.polstride];
    ret += sin_2a * lmap[offset + 2*lpix.polstride];
    
    return ret;
}


template<typename T>
void reference_map2tod(
    Array<T> &tod,
    const Array<T> &local_map,
    const Array<T> &xpointing,
    const LocalPixelization &local_pixelization,
    bool allow_outlier_pixels)
{
    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "reference_map2tod", false);            // on_gpu = false
    check_local_map(local_map, local_pixelization, "reference_map2tod", false);  // on_gpu = false
    check_xpointing(xpointing, nsamp, "reference_map2tod", false);               // on_gpu = false

    int nypix = local_pixelization.nycells << 6;
    int nxpix = local_pixelization.nxcells << 6;
    
    for (long s = 0; s < nsamp; s++) {
	T ypix = xpointing.data[s];
	T xpix = xpointing.data[s + nsamp];
	T alpha = xpointing.data[s + 2*nsamp];

 	T cos_2a, sin_2a;
	dtype<T>::xsincos(2*alpha, &sin_2a, &cos_2a);
	
	// quantize_pixel() is defined in gpu_mm_internals.hpp
	int iy = quantize_pixel(ypix, nypix);   // satisfies -2 <= iy <= nypix
	int ix = quantize_pixel(xpix, nxpix);   // satisfies -2 <= ix <= nxpix
	
	T dy = ypix - T(iy);
	T dx = xpix - T(ix);

	T m00 = eval_map(local_pixelization, local_map.data, cos_2a, sin_2a, iy, ix, allow_outlier_pixels);
	T m01 = eval_map(local_pixelization, local_map.data, cos_2a, sin_2a, iy, ix+1, allow_outlier_pixels);
	T m10 = eval_map(local_pixelization, local_map.data, cos_2a, sin_2a, iy+1, ix, allow_outlier_pixels);
	T m11 = eval_map(local_pixelization, local_map.data, cos_2a, sin_2a, iy+1, ix+1, allow_outlier_pixels);

	tod.data[s] = (1-dy)*(1-dx)*m00 + (1-dy)*(dx)*m01 + (dy)*(1-dx)*m10 + (dy)*(dx)*m11;
    }
}


#define INSTANTIATE(T) \
    template void reference_map2tod( \
	Array<T> &tod, \
	const Array<T> &local_map, \
	const Array<T> &xpointing, \
	const LocalPixelization &local_pixelization, \
	bool allow_outlier_pixels)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
