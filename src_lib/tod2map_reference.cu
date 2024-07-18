#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"  // ALL_LANES
#include <ksgpu/cuda_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T>
void reference_tod2map(
    Array<T> &lmap,
    const Array<T> &tod,
    const Array<T> &xpointing,
    const LocalPixelization &lpix,
    bool partial_pixelization)
{
    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "reference_tod2map", false);     // on_gpu = false
    check_local_map(lmap, lpix, "reference_tod2map", false);              // on_gpu = false
    check_xpointing(xpointing, nsamp, "reference_tod2map", false);        // on_gpu = false
    
    // 'map_evaluator' and 'pixel_locator' are defined in gpu_mm_internals.hpp.
    map_accumulator<T,false> macc(lmap.data, lpix.cell_offsets_cpu.data, lpix.nycells, lpix.nxcells, lpix.ystride, lpix.polstride, partial_pixelization);
    pixel_locator<T> px(lpix.nypix_global, lpix.nxpix_global, lpix.periodic_xcoord);
    uint err = 0;
    
    for (long s = 0; s < nsamp; s++) {
	T ypix = xpointing.data[s];
	T xpix = xpointing.data[s + nsamp];
	T alpha = xpointing.data[s + 2*nsamp];
	T t = tod.data[s];

	T sin_2a, cos_2a;
	dtype<T>::xsincos(2*alpha, &sin_2a, &cos_2a);

	px.locate(ypix, xpix, err);
	macc.accum(px, t, t*cos_2a, t*sin_2a, err);
    }

    check_err(err, "reference_tod2map");
}


#define INSTANTIATE(T) \
    template void reference_tod2map( \
	Array<T> &local_map, \
	const Array<T> &tod, \
	const Array<T> &xpointing, \
	const LocalPixelization &lpix, \
	bool partial_pixelization)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
