#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"  // ALL_LANES
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

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
    bool periodic_xcoord,
    bool partial_pixelization)
{
    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "reference_tod2map", false);     // on_gpu = false
    check_local_map(lmap, lpix, "reference_tod2map", false);              // on_gpu = false
    check_xpointing(xpointing, nsamp, "reference_tod2map", false);        // on_gpu = false

    int nypix = lpix.nycells << 6;  // FIXME
    int nxpix = lpix.nxcells << 6;  // FIXME
    
    // 'map_evaluator' and 'pixel_locator' are defined in gpu_mm_internals.hpp.
    map_accumulator<T,false> macc(lmap.data, lpix.cell_offsets_cpu.data, lpix.nycells, lpix.nxcells, lpix.ystride, lpix.polstride, partial_pixelization);
    pixel_locator<T> px(nypix, nxpix, periodic_xcoord);
    uint err = 0;
    
    for (long s = 0; s < nsamp; s++) {
	T ypix = xpointing.data[s];
	T xpix = xpointing.data[s + nsamp];
	T alpha = xpointing.data[s + 2*nsamp];
	T t = tod.data[s];

 	T q, u;
	dtype<T>::xsincos(2*alpha, &u, &q);
	q *= t;
	u *= t;

	px.locate(ypix, xpix, err);

	macc.accum(px.iy0, px.ix0, t, q, u, (1-px.dy) * (1-px.dx), err);
	macc.accum(px.iy0, px.ix1, t, q, u, (1-px.dy) * (px.dx), err);
	macc.accum(px.iy1, px.ix0, t, q, u, (px.dy) * (1-px.dx), err);
	macc.accum(px.iy1, px.ix1, t, q, u, (px.dy) * (px.dx), err);
    }

    check_err(err, "reference_tod2map");
}


#define INSTANTIATE(T) \
    template void reference_tod2map( \
	Array<T> &local_map, \
	const Array<T> &tod, \
	const Array<T> &xpointing, \
	const LocalPixelization &lpix, \
	bool periodic_xcoord, \
	bool partial_pixelization)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
