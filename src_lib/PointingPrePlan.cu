#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/string_utils.hpp>
#include <cub/device/device_radix_sort.cuh>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// out: 1-d array of length nblocks.
//  - bit 0: 1 if ypix out of bounds
//  - bit 1: 1 if xpix out of bounds
//  - next 30 bits: total number of (map cell, tod cache line) pairs in threadblock ("nmt")
//
// xpointing: shape (3,nsamp)
//
// nsamp: must be a multiple of 32
// nsamp_per_block: must be a multiple of 32

// FIXME check for 32-bit overflows


template<typename T>
__global__ void preplan_kernel(uint *outp, const T *xpointing, uint nsamp, uint nsamp_per_block, int nypix, int nxpix)
{
    __shared__ uint shmem[32];
    
    uint err = 0;
    uint nmt = 0;
    
    // Range of TOD samples to be processed by this threadblock.
    int b = blockIdx.x;
    uint s0 = b * nsamp_per_block;
    uint s1 = min(nsamp, (b+1) * nsamp_per_block);

    for (uint s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];

	range_check_ypix(ypix, nypix, err);  // defined in gpu_mm_internals.hpp
	range_check_xpix(xpix, nxpix, err);  // defined in gpu_mm_internals.hpp
	normalize_xpix(xpix, nxpix);         // defined in gpu_mm_internals.hpp
	
	int iypix0, iypix1, ixpix0, ixpix1;
	quantize_ypix(iypix0, iypix1, ypix, nypix);  // defined in gpu_mm_internals.hpp
	quantize_xpix(ixpix0, ixpix1, xpix, nxpix);  // defined in gpu_mm_internals.hpp

	int iycell_e, iycell_o, ixcell_e, ixcell_o;
	set_up_cell_pair(iycell_e, iycell_o, iypix0, iypix1);  // defined in gpu_mm_internals.hpp
	set_up_cell_pair(ixcell_e, ixcell_o, ixpix0, ixpix1);  // defined in gpu_mm_internals.hpp
	
	nmt += count_nmt(iycell_e, ixcell_e);  // defined in gpu_mm_internals.hpp
	nmt += count_nmt(iycell_e, ixcell_o);  // defined in gpu_mm_internals.hpp
	nmt += count_nmt(iycell_o, ixcell_e);  // defined in gpu_mm_internals.hpp
	nmt += count_nmt(iycell_o, ixcell_o);  // defined in gpu_mm_internals.hpp
    }
    
    // Reduce across threads in the warp.
    err = __reduce_or_sync(ALL_LANES, err);
    nmt = __reduce_add_sync(ALL_LANES, nmt);
    uint out = err | (nmt << 2);

    // Reduce across warps in the block.

    int nwarps = blockDim.x >> 5;
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    
    if (laneId == 0)
	shmem[warpId] = out;

    __syncthreads();

    if (warpId != 0)
	return;

    out = shmem[laneId];
    out = (laneId < nwarps) ? out : 0;
    err = out & 3;
    nmt = out >> 2;

    // FIXME use subset of lanes
    err = __reduce_or_sync(ALL_LANES, err);
    nmt = __reduce_add_sync(ALL_LANES, nmt);
    out = err | (nmt << 2);

    if (laneId == 0)
	outp[b] = out;
}


// -------------------------------------------------------------------------------------------------


template<typename T>
PointingPrePlan::PointingPrePlan(const Array<T> &xpointing_gpu, long nypix_, long nxpix_)
{
    this->nypix = nypix_;
    this->nxpix = nxpix_;
    
    check_xpointing_and_init_nsamp(xpointing_gpu, this->nsamp, "PointingPrePlan constructor", true);  // on_gpu=true
    check_nypix(nypix, "PointingPrePlan constructor");
    check_nxpix(nxpix, "PointingPrePlan constructor");
    
    // Launch params for planner/preplanner kernels.
    // Use at least 16 TOD cache lines per threadblock, and at most 1024 threadblocks.

    long ncl = (nsamp + 31) / 32;
    this->ncl_per_threadblock = max(16L, (ncl+1023)/1024);
    this->ncl_per_threadblock = (ncl_per_threadblock + 3) & ~3;
    this->planner_nblocks = (ncl + ncl_per_threadblock - 1) / ncl_per_threadblock;
    
    Array<uint> arr_gpu({planner_nblocks}, af_gpu);

    preplan_kernel <<< planner_nblocks, 128 >>>
	(arr_gpu.data, xpointing_gpu.data, nsamp, 32*ncl_per_threadblock, nypix, nxpix);

    CUDA_PEEK("preplan_kernel launch");

    Array<uint> arr_cpu = arr_gpu.to_host();
    uint *p = arr_cpu.data;
    uint err = 0;
    ulong nmt = 0;
    
    for (long i = 0; i < planner_nblocks; i++) {
	err |= (p[i] & 3);
	nmt += (p[i] >> 2);
	p[i] = nmt;
    }

    check_err(err, "PointingPrePlan constructor");
    
    if (nmt >= 0x80000000U)
	throw runtime_error("internal error: plan is unexpectedly large");  // FIXME arbitrary threshold -- what is best here?

    arr_gpu.fill(arr_cpu);
    this->nmt_cumsum = arr_gpu;
    this->plan_nmt = nmt;
    
    // Used when launching pointing (tod2map/map2tod) operations.
    this->nmt_per_threadblock = sqrt(plan_nmt);
    this->nmt_per_threadblock = (nmt_per_threadblock + 32) & ~31;
    this->pointing_nblocks = (plan_nmt + nmt_per_threadblock - 1) / nmt_per_threadblock;

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

    // Initialize public members containing byte counts: plan_nbytes, plan_constructor_tmp_nbytes.
    // Note: align128() is defined in gpu_mm_internals.hpp

    long mt_nbytes = align128(plan_nmt * sizeof(ulong));
    long err_nbytes = align128(planner_nblocks * sizeof(int));
    
    this->plan_nbytes = mt_nbytes + err_nbytes;
    this->plan_constructor_tmp_nbytes = mt_nbytes + align128(cub_nbytes);
    this->overhead = 32*plan_nmt/double(nsamp) - 1.0;
}


string PointingPrePlan::str() const
{
    stringstream ss;
    
    ss << "PointingPrePlan("
       << "nsamp=" << nsamp
       << ", nypix=" << nypix
       << ", nxpix=" << nxpix
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
    template PointingPrePlan::PointingPrePlan(const Array<T> &xpointing_gpu, long nypix, long nxpix)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
