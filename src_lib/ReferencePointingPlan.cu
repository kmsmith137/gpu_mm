#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <algorithm>  // std::sort()

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
ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu)
{
    this->nsamp = pp.nsamp;
    this->nypix = pp.nypix;
    this->nxpix = pp.nxpix;
    this->nblocks = pp.nblocks;
    this->rk = pp.rk;

    check_xpointing(xpointing_gpu, pp.nsamp, "ReferencePointingPlan constructor", true);   // on_gpu=true

    // ---------------------------------------------------------------------------------------------
    
    uint nsamp_per_block = 1024;
    uint warps_per_threadblock = 4;
    uint quantize_nblocks = uint(nsamp + nsamp_per_block - 1) / nsamp_per_block;
    uint nerr = quantize_nblocks * warps_per_threadblock;

    this->iypix_arr = Array<int> ({nsamp}, af_gpu);
    this->ixpix_arr = Array<int> ({nsamp}, af_gpu);
    Array<uint> errp = Array<uint> ({nerr}, af_gpu);
    
    quantize_kernel <<< quantize_nblocks, 32*warps_per_threadblock >>>
	(iypix_arr.data, ixpix_arr.data, xpointing_gpu.data, nsamp, nsamp_per_block, nypix, nxpix, errp.data);

    CUDA_PEEK("quantize_kernel launch");

    this->iypix_arr = iypix_arr.to_host();
    this->ixpix_arr = ixpix_arr.to_host();
    errp = errp.to_host();

    uint err = 0;
    for (uint b = 0; b < nerr; b++)
	err |= errp.data[b];

    check_err(err, "ReferencePointingPlan constructor (quantize_kernel)");

    // ---------------------------------------------------------------------------------------------

    this->nmt_cumsum = Array<uint> ({nblocks}, af_uhost);
    
    for (long b = 0; b < nblocks; b++) {
	ulong s0 = min((b) << rk, nsamp);
	ulong s1 = min((b+1) << rk, nsamp);
	    
	while (s0 < s1) {
	    this->_ntmp_cells = 0;
	    for (ulong s = s0; s < s0+32; s++) {
		int iypix = iypix_arr.data[s];
		int ixpix = ixpix_arr.data[s];
		int ixpix1 = (ixpix < (nxpix-1)) ? (ixpix+1) : 0;
		
		this->_add_tmp_cell(iypix, ixpix);
		this->_add_tmp_cell(iypix, ixpix1);
		this->_add_tmp_cell(iypix+1, ixpix);
		this->_add_tmp_cell(iypix+1, ixpix1);
	    }

	    std::sort(_tmp_cells, _tmp_cells + _ntmp_cells);
		
	    for (int i = 0; i < this->_ntmp_cells; i++) {
		ulong mt = ulong(_tmp_cells[i]) | (ulong(s0 >> 5) << 20);
		_mtvec.push_back(mt);
	    }
	    
	    s0 += 32;
	}
	
	nmt_cumsum.data[b] = _mtvec.size();
    }

    std::sort(_mtvec.begin(), _mtvec.end());

    long nmt = _mtvec.size();
    this->sorted_mt = Array<ulong> ({nmt}, af_uhost);
    memcpy(sorted_mt.data, &_mtvec[0], nmt * sizeof(ulong));
    this->_mtvec = vector<ulong> ();  // deallocate
}


void ReferencePointingPlan::_add_tmp_cell(int iypix, int ixpix)
{
    xassert((iypix >= 0) && (iypix < nypix));
    xassert((ixpix >= 0) && (ixpix < nxpix));
    
    int ycell = iypix >> 6;
    int xcell = ixpix >> 6;
    int icell = (ycell << 10) | xcell;
    
    for (int i = 0; i < _ntmp_cells; i++)
	if (_tmp_cells[i] == icell)
	    return;
    
    xassert(_ntmp_cells < 128);
    _tmp_cells[_ntmp_cells++] = icell;
}


string ReferencePointingPlan::str() const
{
    stringstream ss;
    ss << "ReferencePointingPlan(nsamp=" << nsamp << ", nypix=" << nypix << ", nxpix=" << nxpix << ", rk=" << rk << ")";
    return ss.str();
}



// ------------------------------------------------------------------------------------------------


#define INSTANTIATE(T) \
    template ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu)

INSTANTIATE(float);
INSTANTIATE(double);


} // namespace gpu_mm2
