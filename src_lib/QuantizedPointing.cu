#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"

#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// ixpixp, iypixp: length 'nsamp' output arrayts
// nsamp, nsamp_per_block: must be divisible by 32
// errp: array of length 

template<typename T>
__global__ void quantize_kernel(int *iypix, int *ixpix, const T *xpointing, uint nsamp, uint nsamp_per_block, int nypix, int nxpix, int *err)
{
    int b = blockIdx.x;
    uint s0 = b * nsamp_per_block;
    uint s1 = min((b+1) * nsamp_per_block, nsamp);

    for (uint s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
	T ypix = xpointing[s];
	T xpix = xpointing[s+nsamp];

	// FIXME add 'status' argument, and calls to range_check_{xpix,ypix}().
	normalize_xpix(xpix, nxpix);   // defined in gpu_mm2_internals.hpp

	int iy0, iy1, ix0, ix1;
	quantize_ypix(iy0, iy1, ypix, nypix);  // defined in gpu_mm2_internals.hpp
	quantize_xpix(ix0, ix1, xpix, nxpix);  // defined in gpu_mm2_internals.hpp

	ixpix[s] = ix0;
	iypix[s] = iy0;
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T>
QuantizedPointing::QuantizedPointing(const Array<T> &xpointing_gpu, long nypix_, long nxpix_)
{
    this->nypix = nypix_;
    this->nxpix = nxpix_;

    // Call to check_xpointing() initializes this->nsamp.
    check_xpointing(xpointing_gpu, this->nsamp, "QuantizedPointing constructor");
    check_nypix(nypix, "QuantizedPointing constructor");
    check_nxpix(nxpix, "QuantizedPointing constructor");

    uint nsamp_per_block = 1024;
    uint nblocks = uint(nsamp + nsamp_per_block - 1) / nsamp_per_block;
    
    Array<int> iypix_gpu({nsamp}, af_gpu);
    Array<int> ixpix_gpu({nsamp}, af_gpu);
    Array<int> err_gpu({nblocks}, af_gpu);
    
    quantize_kernel <<< nblocks, 128 >>>
	(iypix_gpu.data, ixpix_gpu.data, xpointing_gpu.data, nsamp, nsamp_per_block, nypix, nxpix, err_gpu.data);

    CUDA_PEEK("quantize_kernel launch");

    this->iypix_cpu = iypix_gpu.to_host();
    this->ixpix_cpu = ixpix_gpu.to_host();
    Array<int> err_cpu = err_gpu.to_host();

    int err = 0;
    for (uint b = 0; b < nblocks; b++)
	err |= err_cpu.data[b];

    check_err(err, "QuantizedPointing constructor");
}


#define INSTANTIATE(T) \
    template QuantizedPointing::QuantizedPointing(const Array<T> &xpointing_gpu, long nypix, long nxpix)

INSTANTIATE(float);
INSTANTIATE(double);


} // namespace gpu_mm2
