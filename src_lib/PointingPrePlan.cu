#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/string_utils.hpp>
#include <cub/device/device_radix_sort.cuh>

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


static __device__ inline uint count_local_nmt(int iycell, int ixcell)
{
    uint icell = (iycell << 10) | ixcell;
    uint mmask = __match_any_sync(ALL_LANES, icell);  // all matching lanes
	
    // Block dims are (W,32), so threadIdx.x is the laneId.
    bool is_lowest = ((mmask >> threadIdx.x) == 1);
    bool valid = (iycell >= 0) && (ixcell >= 0);
	
    return (valid && is_lowest) ? 1 : 0;
}


// nmt_out, errflags: 1-d arrays of length nblocks.
// xpointing: shape (3,nsamp)
//
// nsamp: must be a multiple of 32
// nsamp_per_block: must be a multiple of 32


template<typename T, int W, bool Debug>
__global__ void preplan_kernel(
    uint *nmt_out,
    uint *errflags,
    const T *xpointing,
    long nsamp,
    long nsamp_per_block,
    int nypix_global,
    int nxpix_global,
    bool periodic_xcoord)
{
    // Shared memory layout:
    //   uint nmt_unreduced[W]     only used at end of kernel
    //   uint err_unreduced[W]     only used at end of kernel
    
    __shared__ uint shmem[2*W];

    const int warpId = threadIdx.y;
    const int laneId = threadIdx.x;
    
    // Range of TOD samples to be processed by this threadblock.
    int b = blockIdx.x;
    long s0 = b * nsamp_per_block;
    long s1 = min(nsamp, (b+1) * nsamp_per_block);

    // cell_enumerator is defined in gpu_mm_internals.hpp
    cell_enumerator<T,Debug> cells(nypix_global, nxpix_global, periodic_xcoord);
    uint nmt = 0;
    uint err = 0;

    for (long s = s0 + 32*warpId + laneId; s < s1; s += 32*W) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];

	cells.enumerate(ypix, xpix, err);
	
	nmt += count_local_nmt(cells.iy0, cells.ix0);
	nmt += count_local_nmt(cells.iy0, cells.ix1);
	nmt += count_local_nmt(cells.iy0, cells.ix2);
	nmt += count_local_nmt(cells.iy1, cells.ix0);
	nmt += count_local_nmt(cells.iy1, cells.ix1);
	nmt += count_local_nmt(cells.iy1, cells.ix2);
    }
    
    // Do frst-stage reduction (across threads in warp) and write to shared memory.
    
    nmt = __reduce_add_sync(ALL_LANES, nmt);
    err = __reduce_or_sync(ALL_LANES, err);
    
    if (laneId == 0) {
	shmem[warpId] = nmt;
	shmem[warpId+W] = err;
    }

    __syncthreads();

    if (warpId != 0)
	return;

    nmt = (laneId < W) ? shmem[laneId] : 0;
    err = (laneId < W) ? shmem[laneId+W] : 0;
    
    nmt = __reduce_add_sync(ALL_LANES, nmt);
    err = __reduce_or_sync(ALL_LANES, err);
    
    // Write to global memory.
    
    if (laneId == 0) {
	nmt_out[b] = nmt;
	errflags[b] = err;
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T>
PointingPrePlan::PointingPrePlan(
    const Array<T> &xpointing_gpu,
    long nypix_global_,
    long nxpix_global_,
    bool periodic_xcoord_,
    bool debug)
    // Delegate to version of constructor which uses externally allocated arrays.
    : PointingPrePlan(xpointing_gpu, nypix_global_, nxpix_global_,
		      Array<uint> ({preplan_size}, af_gpu),
		      Array<uint> ({preplan_size}, af_gpu),
		      periodic_xcoord_, debug)
{ }


template<typename T>
PointingPrePlan::PointingPrePlan(
    const Array<T> &xpointing_gpu,
    long nypix_global_,
    long nxpix_global_,
    const Array<uint> &nmt_gpu,
    const Array<uint> &err_gpu,
    bool periodic_xcoord_,
    bool debug)
{
    // Warps per threadblock in preplan_kernel
    constexpr int W = 4;
    
    this->nypix_global = nypix_global_;
    this->nxpix_global = nxpix_global_;
    this->periodic_xcoord = periodic_xcoord_;
    this->nmt_cumsum = nmt_gpu;
    
    check_xpointing_and_init_nsamp(xpointing_gpu, this->nsamp, "PointingPrePlan constructor", true);  // on_gpu=true
    check_nypix_global(nypix_global, "PointingPrePlan constructor");
    check_nxpix_global(nxpix_global, "PointingPrePlan constructor");

    xassert(nmt_gpu.is_fully_contiguous());
    xassert(nmt_gpu.size == preplan_size);
    xassert(nmt_gpu.on_gpu());

    xassert(err_gpu.is_fully_contiguous());
    xassert(err_gpu.size == preplan_size);
    xassert(err_gpu.on_gpu());
    
    // Launch params for planner/preplanner kernels.
    // Use at least 16 TOD cache lines per threadblock, and at most PointingPrePlan::preplan_size threadblocks.

    long ncl = (nsamp + 31) / 32;
    this->ncl_per_threadblock = max(16L, (ncl + preplan_size - 1) / preplan_size);
    this->ncl_per_threadblock = (ncl_per_threadblock + 3) & ~3;
    this->planner_nblocks = (ncl + ncl_per_threadblock - 1) / ncl_per_threadblock;
    
    xassert(planner_nblocks > 0);
    xassert(planner_nblocks <= preplan_size);

    if (debug) {
	preplan_kernel<T,W,true> <<< planner_nblocks, {32,W} >>>
	    (nmt_gpu.data,
	     err_gpu.data,
	     xpointing_gpu.data,
	     nsamp,
	     32*ncl_per_threadblock,
	     nypix_global,
	     nxpix_global,
	     periodic_xcoord);
    }
    else {
	preplan_kernel<T,W,false> <<< planner_nblocks, {32,W} >>>
	    (nmt_gpu.data,
	     err_gpu.data,
	     xpointing_gpu.data,
	     nsamp,
	     32*ncl_per_threadblock,
	     nypix_global,
	     nxpix_global,
	     periodic_xcoord);
    }

    CUDA_PEEK("preplan_kernel launch");

    Array<uint> a({ 2*preplan_size }, af_rhost | af_zero);
    uint *nmt_cpu = a.data;
    uint *err_cpu = a.data + preplan_size;
    
    CUDA_CALL(cudaMemcpyAsync(nmt_cpu, nmt_gpu.data, preplan_size * sizeof(uint), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(err_cpu, err_gpu.data, preplan_size * sizeof(uint), cudaMemcpyDefault));
    
    check_cpu_errflags(err_cpu, planner_nblocks, "PointingPrePlan constructor");
    long nmt = 0;
    
    for (long i = 0; i < planner_nblocks; i++) {
	nmt += nmt_cpu[i];
	nmt_cpu[i] = nmt;
    }

    for (long i = planner_nblocks; i < preplan_size; i++)
	nmt_cpu[i] = nmt;
    
    xassert(nmt > 0);
    xassert(nmt <= 1024L * 1024L * 1024L);  // FIXME arbitrary threshold

    CUDA_CALL(cudaMemcpyAsync(nmt_gpu.data, nmt_cpu, preplan_size * sizeof(uint), cudaMemcpyDefault));
    
    // Initialize this->cub_nbytes.
    
    CUDA_CALL(cub::DeviceRadixSort::SortKeys(
        nullptr,                  // void *d_temp_storage
	this->cub_nbytes,         // size_t &temp_storage_bytes
	(const ulong *) nullptr,  // const KeyT *d_keys_in
	(ulong *) nullptr,        // KeyT *d_keys_out
	nmt,                      // NumItemsT num_items
	0,                        // int begin_bit = 0
	20                        // int end_bit = sizeof(KeyT) * 8
	// cudaStream_t stream = 0
    ));

    assert(cub_nbytes > 0);
    
    // Launch params for pointing (map2tod/tod2map) operations.
    // (These must be initialized after 'nmt' is computed.)
    
    this->nmt_per_threadblock = sqrt(nmt);
    this->nmt_per_threadblock = (nmt_per_threadblock + 32) & ~31;
    this->pointing_nblocks = (nmt + nmt_per_threadblock - 1) / nmt_per_threadblock;

    // Initialize remaining public members.
    // Note: align128() is defined in gpu_mm_internals.hpp

    long max_nblocks = max(planner_nblocks, pointing_nblocks);
    long mt_nbytes = align128(nmt * sizeof(ulong));
    long err_nbytes = align128(max_nblocks * sizeof(uint));
    
    this->plan_nmt = nmt;
    this->plan_nbytes = mt_nbytes + err_nbytes;
    this->plan_constructor_tmp_nbytes = mt_nbytes + align128(cub_nbytes);
    this->overhead = 32*nmt/double(nsamp) - 1.0;
}


Array<uint> PointingPrePlan::get_nmt_cumsum() const
{
    Array<uint> ret({planner_nblocks}, af_rhost | af_zero);
    CUDA_CALL(cudaMemcpy(ret.data, nmt_cumsum.data, planner_nblocks * sizeof(uint), cudaMemcpyDefault));
    return ret;
}


string PointingPrePlan::str() const
{
    stringstream ss;
    
    ss << "PointingPrePlan("
       << "nsamp=" << nsamp
       << ", nypix_global=" << nypix_global
       << ", nxpix_global=" << nxpix_global
       << ", plan_nbytes=" << plan_nbytes << " (" << nbytes_to_str(plan_nbytes) << ")"
       << ", tmp_nbytes=" << plan_constructor_tmp_nbytes << " (" << nbytes_to_str(plan_constructor_tmp_nbytes) << ")"
       << ", overhead=" << overhead
       << ", ncl_per_threadblock=" << ncl_per_threadblock
       << ", planner_blocks=" << planner_nblocks
       << ", nmt_per_threadblock=" << nmt_per_threadblock
       << ", pointing_nblocks=" << pointing_nblocks
       << ")";

    return ss.str();
}


// -------------------------------------------------------------------------------------------------

#define INSTANTIATE(T) \
    template PointingPrePlan::PointingPrePlan(const Array<T> &xpointing_gpu, long nypix_global, long nxpix_global, bool periodic_xcoord, bool debug); \
    template PointingPrePlan::PointingPrePlan(const Array<T> &xpointing_gpu, long nypix_global, long nxpix_global, const Array<uint> &buf, const Array<uint> &tmp, bool periodic_xcoord, bool debug)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
