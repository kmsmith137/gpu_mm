#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <algorithm>  // std::sort()

using namespace std;
using namespace gputils;

namespace gpu_mm {
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

	range_check_ypix(ypix, nypix, err);  // defined in gpu_mm_internals.hpp
	range_check_xpix(xpix, nxpix, err);  // defined in gpu_mm_internals.hpp
	normalize_xpix(xpix, nxpix);         // defined in gpu_mm_internals.hpp

	int iy0, iy1, ix0, ix1;
	quantize_ypix(iy0, iy1, ypix, nypix);  // defined in gpu_mm_internals.hpp
	quantize_xpix(ix0, ix1, xpix, nxpix);  // defined in gpu_mm_internals.hpp

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
    this->plan_nmt = pp.plan_nmt;
    this->ncl_per_threadblock = pp.ncl_per_threadblock;
    this->planner_nblocks = pp.planner_nblocks;

    check_xpointing(xpointing_gpu, pp.nsamp, "ReferencePointingPlan constructor", true);   // on_gpu=true
    // check_buffer(tmp) happens later

    xassert(nsamp > (planner_nblocks-1) * ncl_per_threadblock * 32);
    xassert(nsamp <= (planner_nblocks) * ncl_per_threadblock * 32);
    
    // ------------------------------  quantize kernel  ------------------------------

    // Note: 'warps_per_threadblock' and 'nsamp_per_block' are static members of ReferencePointingPlan.
    uint quantize_nblocks = uint(nsamp + nsamp_per_block - 1) / nsamp_per_block;
    uint quantize_nerr = quantize_nblocks * warps_per_threadblock;
    
    long min_nbytes = (2*nsamp + quantize_nerr) * sizeof(int);
    check_buffer(tmp, min_nbytes, "ReferencePointingPlan constructor", "tmp");

    int *iypix_gpu = (int *) tmp.data;
    int *ixpix_gpu = (int *) (iypix_gpu + nsamp);
    uint *err_gpu = (uint*) (ixpix_gpu + nsamp);
    
    this->iypix_arr = Array<int> ({nsamp}, af_rhost);
    this->ixpix_arr = Array<int> ({nsamp}, af_rhost);
    Array<uint> err_arr = Array<uint> ({quantize_nerr}, af_rhost);

    quantize_kernel <<< quantize_nblocks, 32*warps_per_threadblock >>>
	(iypix_gpu, ixpix_gpu, xpointing_gpu.data, nsamp, nsamp_per_block, nypix, nxpix, err_gpu);

    CUDA_PEEK("quantize_kernel launch");
    CUDA_CALL(cudaDeviceSynchronize());
    
    CUDA_CALL(cudaMemcpy(iypix_arr.data, iypix_gpu, nsamp * sizeof(int), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(ixpix_arr.data, ixpix_gpu, nsamp * sizeof(int), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(err_arr.data, err_gpu, quantize_nerr * sizeof(uint), cudaMemcpyDefault));

    uint err = 0;
    for (uint b = 0; b < quantize_nerr; b++)
	err |= err_arr.data[b];

    check_err(err, "ReferencePointingPlan constructor (quantize_kernel)");

    // ------------------------------  nmt_cumsum, sorted_mt  ------------------------------

    this->nmt_cumsum = Array<uint> ({planner_nblocks}, af_uhost);
    this->sorted_mt = Array<ulong> ({plan_nmt}, af_uhost);
    long nmt_curr = 0;
    
    for (long b = 0; b < planner_nblocks; b++) {
	long s0 = b * ncl_per_threadblock * 32;
	long s1 = (b+1) * ncl_per_threadblock * 32;
	s1 = min(s1, nsamp);
	    
	while (s0 < s1) {
	    this->_ntmp_cells = 0;
	    
	    for (long s = s0; s < s0+32; s++) {
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
		xassert(nmt_curr < plan_nmt);
		sorted_mt.data[nmt_curr] = ulong(_tmp_cells[i]) | (ulong(s0 >> 5) << 20);
		nmt_curr++;
	    }
	    
	    s0 += 32;
	}
	
	nmt_cumsum.data[b] = nmt_curr;
    }

    xassert(nmt_curr == plan_nmt);
    std::sort(sorted_mt.data, sorted_mt.data + plan_nmt);
}


template<typename T>
ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu) :
    // Delegate to externally-allocated version of constructor
    ReferencePointingPlan(pp, xpointing_gpu, Array<unsigned char> ({get_constructor_tmp_nbytes(pp)}, af_gpu))
{ }


// Static member function.
long ReferencePointingPlan::get_constructor_tmp_nbytes(const PointingPrePlan &pp)
{
    // FIXME remove cut-and-paste with constructor.
    // Note: 'warps_per_threadblock' and 'nsamp_per_block' are static members of ReferencePointingPlan.
    uint quantize_nblocks = uint(pp.nsamp + nsamp_per_block - 1) / nsamp_per_block;
    uint quantize_nerr = quantize_nblocks * warps_per_threadblock;
    return (2*pp.nsamp + quantize_nerr) * sizeof(int);
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
    ss << "ReferencePointingPlan(nsamp=" << nsamp << ", nypix=" << nypix << ", nxpix=" << nxpix << ")";
    return ss.str();
}



// ------------------------------------------------------------------------------------------------


#define INSTANTIATE(T) \
    template ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu); \
    template ReferencePointingPlan::ReferencePointingPlan(const PointingPrePlan &pp, const Array<T> &xpointing_gpu, const Array<unsigned char> &tmp)

INSTANTIATE(float);
INSTANTIATE(double);


} // namespace gpu_mm
