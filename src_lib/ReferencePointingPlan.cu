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
ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu, const Array<unsigned char> &tmp)
{
    this->nsamp = pp.nsamp;
    this->nypix = pp.nypix;
    this->nxpix = pp.nxpix;
    this->nblocks = pp.nblocks;
    this->rk = pp.rk;

    check_xpointing(xpointing_gpu, pp.nsamp, "ReferencePointingPlan constructor", true);   // on_gpu=true

    // ---------------------------------------------------------------------------------------------

    // Note: 'warps_per_threadblock' and 'nsamp_per_block' are static members of ReferencePointingPlan.
    uint quantize_nblocks = uint(nsamp + nsamp_per_block - 1) / nsamp_per_block;
    uint nerr = quantize_nblocks * warps_per_threadblock;
    
    long min_nbytes = (2*nsamp + nerr) * sizeof(int);
    check_buffer(tmp, min_nbytes, "ReferencePointingPlan constructor", "tmp");

    int *iypix_gpu = (int *) tmp.data;
    int *ixpix_gpu = (int *) (iypix_gpu + nsamp);
    uint *err_gpu = (uint*) (ixpix_gpu + nsamp);
    
    this->iypix_arr = Array<int> ({nsamp}, af_rhost);
    this->ixpix_arr = Array<int> ({nsamp}, af_rhost);
    Array<uint> err_arr = Array<uint> ({nerr}, af_rhost);

    quantize_kernel <<< quantize_nblocks, 32*warps_per_threadblock >>>
	(iypix_gpu, ixpix_gpu, xpointing_gpu.data, nsamp, nsamp_per_block, nypix, nxpix, err_gpu);

    CUDA_PEEK("quantize_kernel launch");
    CUDA_CALL(cudaDeviceSynchronize());
    
    CUDA_CALL(cudaMemcpy(iypix_arr.data, iypix_gpu, nsamp * sizeof(int), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(ixpix_arr.data, ixpix_gpu, nsamp * sizeof(int), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(err_arr.data, err_gpu, nerr * sizeof(uint), cudaMemcpyDefault));

    uint err = 0;
    for (uint b = 0; b < nerr; b++)
	err |= err_arr.data[b];

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


template<typename T>
ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu) :
    // Delegate to externally-allocated version of constructor
    ReferencePointingPlan(pp, xpointing_gpu, Array<unsigned char> ({get_constructor_tmp_nbytes(pp)}, af_gpu))
{ }


// Static member function.
long ReferencePointingPlan::get_constructor_tmp_nbytes(const PointingPrePlan &pp)
{
    // Note: 'warps_per_threadblock' and 'nsamp_per_block' are static members of ReferencePointingPlan.
    uint quantize_nblocks = uint(pp.nsamp + nsamp_per_block - 1) / nsamp_per_block;
    uint nerr = quantize_nblocks * warps_per_threadblock;
    return (2*pp.nsamp + nerr) * sizeof(int);
}


// Helper for constructor.
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
    template ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu); \
    template ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu, const Array<unsigned char> &tmp)

INSTANTIATE(float);
INSTANTIATE(double);


} // namespace gpu_mm2