#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/string_utils.hpp>
#include <cub/device/device_radix_sort.cuh>

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// out: 1-d array of length nblocks.
//  - bit 0: 1 if ypix out of bounds
//  - bit 1: 1 if xpix out of bounds
//  - next 30 bits: total number of (map cell, tod cache line) pairs in threadblock ("nmt")
//
// xpointing: shape (3, nsamp)
//
// nsamp: must be a multiple of 32
// nsamp_per_block: must be a multiple of 32


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

	range_check_ypix(ypix, nypix, err);  // defined in gpu_mm2_internals.hpp
	range_check_xpix(xpix, nxpix, err);  // defined in gpu_mm2_internals.hpp
	normalize_xpix(xpix, nxpix);         // defined in gpu_mm2_internals.hpp
	
	int iypix0, iypix1, ixpix0, ixpix1;
	quantize_ypix(iypix0, iypix1, ypix, nypix);  // defined in gpu_mm2_internals.hpp
	quantize_xpix(ixpix0, ixpix1, xpix, nxpix);  // defined in gpu_mm2_internals.hpp

	int iycell_e, iycell_o, ixcell_e, ixcell_o;
	set_up_cell_pair(iycell_e, iycell_o, iypix0, iypix1);  // defined in gpu_mm2_internals.hpp
	set_up_cell_pair(ixcell_e, ixcell_o, ixpix0, ixpix1);  // defined in gpu_mm2_internals.hpp
	
	nmt += count_nmt(iycell_e, ixcell_e);  // defined in gpu_mm2_internals.hpp
	nmt += count_nmt(iycell_e, ixcell_o);  // defined in gpu_mm2_internals.hpp
	nmt += count_nmt(iycell_o, ixcell_e);  // defined in gpu_mm2_internals.hpp
	nmt += count_nmt(iycell_o, ixcell_o);  // defined in gpu_mm2_internals.hpp
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

    // Initializes this->nsamp.
    check_xpointing(xpointing_gpu, this->nsamp, "PointingPrePlan constructor");
    check_nypix(nypix, "PointingPrePlan constructor");
    check_nxpix(nxpix, "PointingPrePlan constructor");
    
    // Aim for 1000-2000 threadblocks.
    this->rk = max(7, int(log2(nsamp)) - 10);      // rk = log2(samples per block)
    this->nblocks = (nsamp + (1<<rk) - 1) >> rk;   // ceil(nsamp/2^rk)

    assert(rk >= 5);    // Kernel assumes nsamp_per_block is a multiple of 32.
    assert(rk <= 25);   // Ensures 'uint32 nmt' can't overflow in kernel.
    
    Array<uint> arr_gpu({nblocks}, af_gpu);

    preplan_kernel <<< nblocks, 128 >>>
	(arr_gpu.data, xpointing_gpu.data, nsamp, (1 << rk), nypix, nxpix);

    CUDA_PEEK("preplan_kernel launch");

    Array<uint> arr_cpu = arr_gpu.to_host();
    uint *p = arr_cpu.data;
    uint err = 0;
    ulong nmt = 0;
    
    for (long i = 0; i < nblocks; i++) {
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
    // Note: align128() is defined in gpu_mm2_internals.hpp

    this->plan_nbytes = align128(plan_nmt * sizeof(ulong));
    this->plan_constructor_tmp_nbytes = plan_nbytes + align128(cub_nbytes);
    // Forthcoming: plan_map2tod_tmp_nbytes
}


// -------------------------------------------------------------------------------------------------


void PointingPrePlan::show(ostream &os) const
{
    long plan_ntt = (nsamp / 32);
    double ratio = double(plan_nmt) / double(plan_ntt);

    os << "PointingPrePlan:\n"
       << "   nsamp=" << nsamp << " (float64 TOD is " << nbytes_to_str(8*nsamp) << ")\n"
       << "   nypix=" << nypix << ", nxpix=" << nxpix << " (float64 map is " << nbytes_to_str(24*nypix*nxpix) << ")\n"
       << "   ntt=" << plan_ntt << ", nmt=" << plan_nmt << ", ratio=" << ratio << "\n"
       << "   rk=" << rk << ", nblocks=" << nblocks << "\n"
       << "     Plan size: " << nbytes_to_str(4*plan_ntt + 8*plan_nmt) << "\n"
       << "     Temporary memory needed for construction: " << nbytes_to_str(8*plan_nmt + cub_nbytes)
       << endl;
}


// -------------------------------------------------------------------------------------------------

#define INSTANTIATE(T) \
    template PointingPrePlan::PointingPrePlan(const Array<T> &xpointing_gpu, long nypix, long nxpix)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
