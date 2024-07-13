#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif

// -------------------------------------------------------------------------------------------------


template<typename T>
__device__ T eval_map(const T *map, int npix, T cos_2a, T sin_2a)
{
    return map[0] + (cos_2a * map[npix]) + (sin_2a * map[2*npix]);
}


template<typename T>
__global__ void unplanned_map2tod_kernel(T *tod, const T *map, const T *xpointing, uint nsamp, int nypix, int nxpix, uint nsamp_per_block)
{
    static constexpr T one = 1;
    static constexpr T two = 2;
    
    int npix = nypix * nxpix;
    uint s0 = blockIdx.x * nsamp_per_block;
    uint s1 = min(nsamp, s0 + nsamp_per_block);
    
    for (uint s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];
	T alpha = xpointing[s + 2*nsamp];

	// FIXME remove this code
	if (1) {
	    uint err = 0;
	    range_check_ypix(ypix, nypix, err);
	    range_check_xpix(xpix, nxpix, err);
	    assert(err == 0);
	}
	
	int iy0, iy1, ix0, ix1;
	quantize_ypix(iy0, iy1, ypix, nypix);  // defined in gpu_mm_internals.hpp
	quantize_xpix(ix0, ix1, xpix, nxpix);  // defined in gpu_mm_internals.hpp

	T dy = ypix - iy0;
	T dx = xpix - ix0;

	T cos_2a, sin_2a;
	dtype<T>::xsincos(two*alpha, &sin_2a, &cos_2a);

	T t = eval_map(map + iy0*nxpix + ix0, npix, cos_2a, sin_2a) * (one-dy) * (one-dx);
	t += eval_map(map + iy0*nxpix + ix1, npix, cos_2a, sin_2a) * (one-dy) * (dx);
	t += eval_map(map + iy1*nxpix + ix0, npix, cos_2a, sin_2a) * (dy) * (one-dx);
	t += eval_map(map + iy1*nxpix + ix1, npix, cos_2a, sin_2a) * (dy) * (dx);

	tod[s] = t;
    }
}


template<typename T>
void launch_unplanned_map2tod(Array<T> &tod, const Array<T> &map, const Array<T> &xpointing)
{
    long nsamp, nypix, nxpix;
    
    check_map_and_init_npix(map, nypix, nxpix, "launch_unplanned_map2tod", true);  // on_gpu=true
    check_tod_and_init_nsamp(tod, nsamp, "launch_unplanned_map2tod", true);        // on_gpu=true
    check_xpointing(xpointing, nsamp, "launch_unplanned_map2tod", true);           // on_gpu=true

    int nthreads_per_block = 128;
    int nsamp_per_block = 1024;
    int nblocks = (nsamp + nsamp_per_block - 1) / nsamp_per_block;
    
    unplanned_map2tod_kernel <<< nblocks, nthreads_per_block >>>
	(tod.data, map.data, xpointing.data, nsamp, nypix, nxpix, nsamp_per_block);

    CUDA_PEEK("unplanned_map2tod kernel launch");
}


#define INSTANTIATE(T) \
    template void launch_unplanned_map2tod(Array<T> &tod, const Array<T> &map, const Array<T> &xpointing)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
