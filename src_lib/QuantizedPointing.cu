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
// errp: array of length (B*W), where B=nblocks and W=warps_per_threadblock

template<typename T>
__global__ void quantize_kernel(int *iypix, int *ixpix, const T *xpointing, uint nsamp, uint nsamp_per_block, int nypix, int nxpix, uint *errp)
{
    int b = blockIdx.x;
    uint s0 = b * nsamp_per_block;
    uint s1 = min((b+1) * nsamp_per_block, nsamp);
    uint err = 0;

    for (uint s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
	T ypix = xpointing[s];
	T xpix = xpointing[s+nsamp];

	range_check_ypix(ypix, nypix, err);  // defined in gpu_mm2_internals.hpp
	range_check_xpix(xpix, nxpix, err);  // defined in gpu_mm2_internals.hpp
	normalize_xpix(xpix, nxpix);         // defined in gpu_mm2_internals.hpp

	int iy0, iy1, ix0, ix1;
	quantize_ypix(iy0, iy1, ypix, nypix);  // defined in gpu_mm2_internals.hpp
	quantize_xpix(ix0, ix1, xpix, nxpix);  // defined in gpu_mm2_internals.hpp

	ixpix[s] = ix0;
	iypix[s] = iy0;
    }
    
    err = __reduce_or_sync(ALL_LANES, err);

    int laneId = threadIdx.x & 31;
    int ierr = (b * blockDim.x + threadIdx.x) >> 5;
    
    if (laneId == 0)
	errp[ierr] = err;
}


// -------------------------------------------------------------------------------------------------


template<typename T>
QuantizedPointing::QuantizedPointing(const Array<T> &xpointing_gpu, long nypix_, long nxpix_)
{
    this->nsamp = 0;
    this->nypix = nypix_;
    this->nxpix = nxpix_;

    // Call to check_xpointing() initializes this->nsamp.
    check_xpointing(xpointing_gpu, this->nsamp, "QuantizedPointing constructor");
    check_nypix(nypix, "QuantizedPointing constructor");
    check_nxpix(nxpix, "QuantizedPointing constructor");

    uint nsamp_per_block = 1024;
    uint warps_per_threadblock = 4;
    uint nblocks = uint(nsamp + nsamp_per_block - 1) / nsamp_per_block;
    uint nerr = nblocks * warps_per_threadblock;
    
    Array<int> iypix_gpu({nsamp}, af_gpu);
    Array<int> ixpix_gpu({nsamp}, af_gpu);
    Array<uint> err_gpu({nerr}, af_gpu);
    
    quantize_kernel <<< nblocks, 32*warps_per_threadblock >>>
	(iypix_gpu.data, ixpix_gpu.data, xpointing_gpu.data, nsamp, nsamp_per_block, nypix, nxpix, err_gpu.data);

    CUDA_PEEK("quantize_kernel launch");

    this->iypix_cpu = iypix_gpu.to_host();
    this->ixpix_cpu = ixpix_gpu.to_host();
    Array<uint> err_cpu = err_gpu.to_host();

    uint err = 0;
    for (uint b = 0; b < nerr; b++)
	err |= err_cpu.data[b];

    check_err(err, "QuantizedPointing constructor");
}


string QuantizedPointing::str() const
{
    stringstream ss;
    ss << "QuantizedPointing(nsamp=" << nsamp << ", nypix=" << nypix << ", nxpix=" << nxpix << ")";
    return ss.str();
}


// -------------------------------------------------------------------------------------------------

    
struct nmt_cumsum_helper
{
    const int nypix;
    const int nxpix;

    int nmt = 0;
    int cells[128];

    nmt_cumsum_helper(int nypix_, int nxpix_) :
	nypix(nypix_), nxpix(nxpix_) { }

    void add(int iypix, int ixpix)
    {
	assert((iypix >= 0) && (iypix < nypix));
	assert((ixpix >= 0) && (ixpix < nxpix));

	int ytile = iypix >> 6;
	int xtile = ixpix >> 6;
	int icell = (ytile << 10) | xtile;

	for (int i = 0; i < nmt; i++)
	    if (cells[i] == icell)
		return;

	assert(nmt < 128);
	cells[nmt++] = icell;
    }
};


Array<uint> QuantizedPointing::compute_nmt_cumsum(int rk) const
{
    xassert(rk >= 0);
    long nblocks = (nsamp + (1<<rk) - 1) >> rk;   // ceil(nsamp/2^rk)
    Array<uint> ret({nblocks}, af_uhost);
    
    long s_curr = 0;
    long nmt_curr = 0;  // accumulated nmt[s] for s < s_curr
    nmt_cumsum_helper h(nypix, nxpix);

    for (long b = 0; b < nblocks; b++) {
	long s_end = min((b+1) << rk, nsamp);

	while (s_curr < s_end) {
	    h.nmt = 0;
	    for (long s = s_curr; s < s_curr+32; s++) {
		int iypix = iypix_cpu.data[s];
		int ixpix = ixpix_cpu.data[s];
		int ixpix1 = (ixpix < (nxpix-1)) ? (ixpix+1) : 0;
		
		h.add(iypix, ixpix);
		h.add(iypix, ixpix1);
		h.add(iypix+1, ixpix);
		h.add(iypix+1, ixpix1);
	    }

	    nmt_curr += h.nmt;
	    s_curr += 32;
	}

	xassert(nmt_curr < (1L << 32));
	xassert(s_curr == s_end);
	ret.data[b] = nmt_curr;
    }

    xassert(s_curr == nsamp);
    return ret;
}


// ------------------------------------------------------------------------------------------------


#define INSTANTIATE(T) \
    template QuantizedPointing::QuantizedPointing(const Array<T> &xpointing_gpu, long nypix, long nxpix)

INSTANTIATE(float);
INSTANTIATE(double);


} // namespace gpu_mm2
