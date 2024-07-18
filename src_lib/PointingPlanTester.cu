#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <ksgpu/cuda_utils.hpp>
#include <algorithm>  // std::sort()

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// ixpixp, iypixp: length 'nsamp' output arrays
// nsamp, nsamp_per_block: must be divisible by 32
// errp: array of length (B*W), where B=nblocks and W=warps_per_threadblock

template<typename T, int W>
__global__ void quantize_kernel(
    int *iypix,
    int *ixpix,
    const T *xpointing,
    uint *errflags,
    long nsamp,
    long nsamp_per_block,
    int nypix_global,
    int nxpix_global,
    bool periodic_xcoord)
{
    // For write_errflags().
    __shared__ uint shmem[W];

    const int warpId = threadIdx.y;
    const int laneId = threadIdx.x;
    
    // Range of TOD samples to be processed by this threadblock.
    int b = blockIdx.x;
    long s0 = b * nsamp_per_block;
    long s1 = min(nsamp, (b+1) * nsamp_per_block);

    // pixel_locator is defined in gpu_mm_internals.hpp
    pixel_locator<T> px(nypix_global, nxpix_global, periodic_xcoord);
    uint err = 0;

    for (long s = s0 + 32*warpId + laneId; s < s1; s += blockDim.x) {
	T ypix = xpointing[s];
	T xpix = xpointing[s+nsamp];

	px.locate(ypix, xpix, err);
	
	iypix[s] = px.iy0;
	ixpix[s] = px.ix0;
    }

    // No need for __syncthreads() before write_errflags(), since no one else uses shared memory.
    // Warning: write_errflags() assumes thread layout is {32,W,1}, and block layout is {B,1,1}!
    write_errflags(errflags, shmem, err);
}


// -------------------------------------------------------------------------------------------------


template<typename T>
PointingPlanTester::PointingPlanTester(const PointingPrePlan &pp, const Array<T> &xpointing_gpu, const Array<unsigned char> &tmp)
{
    static constexpr int W = 4;
    
    this->nsamp = pp.nsamp;
    this->nypix_global = pp.nypix_global;
    this->nxpix_global = pp.nxpix_global;
    this->periodic_xcoord = pp.periodic_xcoord;
    this->plan_nmt = pp.plan_nmt;
    this->ncl_per_threadblock = pp.ncl_per_threadblock;
    this->planner_nblocks = pp.planner_nblocks;

    check_xpointing(xpointing_gpu, pp.nsamp, "PointingPlanTester constructor", true);   // on_gpu=true
    // check_buffer(tmp) happens later

    xassert(nsamp > (planner_nblocks-1) * ncl_per_threadblock * 32);
    xassert(nsamp <= (planner_nblocks) * ncl_per_threadblock * 32);
    
    // ------------------------------  quantize kernel  ------------------------------

    // Note: 'nsamp_per_block' is a static member of PointingPlanTester.
    uint quantize_nblocks = uint(nsamp + nsamp_per_block - 1) / nsamp_per_block;
    
    long min_nbytes = (2*nsamp + quantize_nblocks) * sizeof(int);
    check_buffer(tmp, min_nbytes, "PointingPlanTester constructor", "tmp");

    int *iypix_gpu = (int *) tmp.data;
    int *ixpix_gpu = (int *) (iypix_gpu + nsamp);
    uint *errflags_gpu = (uint*) (ixpix_gpu + nsamp);

    quantize_kernel<T,W> <<< quantize_nblocks, {32,W} >>>
	(iypix_gpu,
	 ixpix_gpu,
	 xpointing_gpu.data,
	 errflags_gpu,
	 nsamp,
	 nsamp_per_block,
	 nypix_global,
	 nxpix_global,
	 periodic_xcoord);

    CUDA_PEEK("quantize_kernel launch");

    this->iypix_arr = Array<int> ({nsamp}, af_rhost);
    this->ixpix_arr = Array<int> ({nsamp}, af_rhost);
    check_gpu_errflags(errflags_gpu, quantize_nblocks, "PointingPlanTester constructor (quantize_kernel)");
    
    CUDA_CALL(cudaMemcpy(iypix_arr.data, iypix_gpu, nsamp * sizeof(int), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(ixpix_arr.data, ixpix_gpu, nsamp * sizeof(int), cudaMemcpyDefault));

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
		int ixpix1 = (ixpix < (nxpix_global-1)) ? (ixpix+1) : 0;
		
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
PointingPlanTester::PointingPlanTester(const PointingPrePlan &pp, const Array<T> &xpointing_gpu) :
    // Delegate to externally-allocated version of constructor
    PointingPlanTester(pp, xpointing_gpu, Array<unsigned char> ({get_constructor_tmp_nbytes(pp)}, af_gpu))
{ }


// Static member function.
long PointingPlanTester::get_constructor_tmp_nbytes(const PointingPrePlan &pp)
{
    // FIXME remove cut-and-paste with constructor.
    // Note: 'nsamp_per_block' is a static member of PointingPlanTester.
    uint quantize_nblocks = uint(pp.nsamp + nsamp_per_block - 1) / nsamp_per_block;
    return (2*pp.nsamp + quantize_nblocks) * sizeof(int);
}


// Helper for constructor.
void PointingPlanTester::_add_tmp_cell(int iypix, int ixpix)
{
    xassert((iypix >= 0) && (iypix < nypix_global));
    xassert((ixpix >= 0) && (ixpix < nxpix_global));
    
    int ycell = iypix >> 6;
    int xcell = ixpix >> 6;
    int icell = (ycell << 10) | xcell;
    
    for (int i = 0; i < _ntmp_cells; i++)
	if (_tmp_cells[i] == icell)
	    return;
    
    xassert(_ntmp_cells < 128);
    _tmp_cells[_ntmp_cells++] = icell;
}


string PointingPlanTester::str() const
{
    stringstream ss;
    ss << "PointingPlanTester(nsamp=" << nsamp << ", nypix_global=" << nypix_global << ", nxpix_global=" << nxpix_global << ")";
    return ss.str();
}



// ------------------------------------------------------------------------------------------------


#define INSTANTIATE(T) \
    template PointingPlanTester::PointingPlanTester(const PointingPrePlan &pp, const Array<T> &xpointing_gpu); \
    template PointingPlanTester::PointingPlanTester(const PointingPrePlan &pp, const Array<T> &xpointing_gpu, const Array<unsigned char> &tmp)

INSTANTIATE(float);
INSTANTIATE(double);


} // namespace gpu_mm
