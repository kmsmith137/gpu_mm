#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"

#include <gputils/cuda_utils.hpp>
#include <gputils/string_utils.hpp>
#include <gputils/constexpr_functions.hpp>   // constexpr_is_log2()
#include <cub/device/device_radix_sort.cuh>

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// absorb_mt(): Helper function for plan_kerne().
// The number of arguments is awkwardly large!

__device__ __forceinline__ void
absorb_mt(ulong *plan_mt, int *shmem,        // pointers
	  ulong &mt_local, int &nmt_local,   // per-warp ring buffer
	  uint icell, uint amask, int na,    // map cells to absorb
	  uint s, int na_prev,               // additional data needed to construct mt_new
	  int nmt_max, uint &err)            // error testing and reporting
{
    if (na == 0)
	return;
    
    // Block dims are (W,32), so threadIdx.x is the laneId.
    int laneId = threadIdx.x;
    
    // Logical laneId (relative to current value of nmt_local, wrapped around)
    int llid = (laneId + 32 - nmt_local) & 31;
    
    // Permute 'icell' so that llid=N contains the N-th active icell
    uint src_lane = __fns(amask, 0, llid+1);
    icell = __shfl_sync(ALL_LANES, icell, src_lane & 31);  // FIXME do I need "& 31"?

    // Secondary cache line index 0 <= a < 2.
    uint a = na_prev + llid;
    a = (a < 2) ? a : 2;

    // Promote (uint20 icell) to (uint64 mt_new).
    // Reminder: mt_new bit layout is
    //   Low 10 bits = Global xcell index
    //   Next 10 bits = Global ycell index
    //   Next 26 bits = Primary TOD cache line index
    //   High 18 bits = Secondary TOD cache line index 0 <= a < 2
    
    ulong mt_new = icell | (ulong(s >> 5) << 20) | (ulong(a) << 46);

    // Extend ring buffer.
    // If nmt_local is >32, then it "wraps around" from mt_local to mt_new.
    
    mt_local = (laneId < nmt_local) ? mt_local : mt_new;
    nmt_local += na;

    if (nmt_local < 32)
	return;

    // If we get here, we've accumulated 32 values of 'mt_local'.
    // These values can now be written to global memory.

    // Output array index (in 'plan_mt' array)
    int nout = 0;
    if (laneId == 0)
	nout = atomicAdd(shmem, 32);
    nout = __shfl_sync(ALL_LANES, nout, 0);  // broadcast from lane 0 to all lanes
    nout += laneId;

    if (nout < nmt_max)
	plan_mt[nout] = mt_local;

    nmt_local -= 32;
    mt_local = mt_new;
    err = (nout < nmt_max) ? err : (err | 4);
}


template<typename T, int W>
__global__ void plan_kernel(ulong *plan_mt, const T *xpointing, uint *nmt_cumsum, uint nsamp, uint nsamp_per_block, int nypix, int nxpix, uint *errp)
{
    // Assumed for convienience in shared memory logic
    static_assert(W <= 30);

    // FIXME can be removed soon
    assert(blockDim.x == 32);
    assert(blockDim.y == W);
		      
    // Block dims are (W,32)
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    
    // Shared memory layout:
    //   int    nmt_counter       running total of 'plan_mt' values written by this block
    //   int    sid_counter       running total of secondary cache lines for this block
    //   int    nmt_local[W]      used once at end of kernel
    
    __shared__ int shmem[32];  // only need (W+2) elts, but convenient to pad to 32
    
    // Zero shared memory
    if (warpId == 0)
	shmem[laneId] = 0;
    
    // Range of TOD samples to be processed by this threadblock.
    int b = blockIdx.x;
    uint s0 = b * nsamp_per_block;
    uint s1 = min(nsamp, (b+1) * nsamp_per_block);
    
    // Range of nmt values to be written by this threadblock.
    uint mt_out0 = b ? nmt_cumsum[b-1] : 0;
    uint mt_out1 = nmt_cumsum[b];
    int nmt_max = mt_out1 - mt_out0;
    
    // Shift output pointer 'plan_mt'.
    // FIXME some day, consider implementing cache-aligned IO as optimization
    plan_mt += mt_out0;
    
    // (mt_local, nmt_local) act as a per-warp ring buffer.
    // The value of nmt_local is the same on all threads in the warp.
    ulong mt_local = 0;
    int nmt_local = 0;
    uint err = 0;

    for (uint s = s0 + 32*warpId + laneId; s < s1; s += 32*W) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];

	// For now, I'm including the range checks, even though they should be
	// redundant with the preplan. (FIXME: does it affect running time?)
	
	range_check_ypix(ypix, nypix, err);  // defined in gpu_mm2_internals.hpp
	range_check_xpix(xpix, nxpix, err);  // defined in gpu_mm2_internals.hpp
	normalize_xpix(xpix, nxpix);         // defined in gpu_mm2_internals.hpp
	 
	int iypix0, iypix1, ixpix0, ixpix1;
	quantize_ypix(iypix0, iypix1, ypix, nypix);  // defined in gpu_mm2_internals.hpp
	quantize_xpix(ixpix0, ixpix1, xpix, nxpix);  // defined in gpu_mm2_internals.hpp
	
	int iycell_e, iycell_o, ixcell_e, ixcell_o;
	set_up_cell_pair(iycell_e, iycell_o, iypix0, iypix1);  // defined in gpu_mm2_internals.hpp
	set_up_cell_pair(ixcell_e, ixcell_o, ixpix0, ixpix1);  // defined in gpu_mm2_internals.hpp
	
	uint icell0, icell1, icell2, icell3;
	uint amask0, amask1, amask2, amask3;
	int na0, na1, na2, na3;
	 
	analyze_cell_pair(iycell_e, ixcell_e, icell0, amask0, na0);
	analyze_cell_pair(iycell_e, ixcell_o, icell1, amask1, na1);
	analyze_cell_pair(iycell_o, ixcell_e, icell2, amask2, na2);
	analyze_cell_pair(iycell_o, ixcell_o, icell3, amask3, na3);

	absorb_mt(plan_mt, shmem,        // pointers
		  mt_local, nmt_local,   // per-warp ring buffer
		  icell0, amask0, na0,   // map cells to absorb
		  s, 0,                  // additional data needed to construct mt_new
		  nmt_max, err);         // error testing and reporting
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  icell1, amask1, na1,
		  s, na0,
		  nmt_max, err);
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  icell2, amask2, na2,
		  s, na0+na1,
		  nmt_max, err);
	
	absorb_mt(plan_mt, shmem,
		  mt_local, nmt_local,
		  icell3, amask3, na3,
		  s, na0+na1+na2,
		  nmt_max, err);
    }
    
    if (laneId == 0)
	shmem[warpId+2] = nmt_local;

    __syncthreads();

    // FIXME logic here could be optimized -- align IO on cache lines,
    // use fewer warp shuffles to reduce.
    
    int shmem_remote = shmem[laneId];
    
    int nout = __shfl_sync(ALL_LANES, shmem_remote, 0);    // nmt_counter
    for (int w = 0; w < warpId; w++)
	nout += __shfl_sync(ALL_LANES, shmem_remote, w+2);  // value of 'nmt_local' on warp w
    nout += laneId;

    if ((laneId < nmt_local) && (nout < nmt_max))
	plan_mt[nout] = mt_local;

    bool fail = (warpId == (W-1)) && (laneId == 0) && ((nout + nmt_local) != nmt_max);
    err = fail ? (err | 4) : err;

    errp[b] = err;
}


// -------------------------------------------------------------------------------------------------


template<typename T>
PointingPlan::PointingPlan(const PointingPrePlan &preplan, const Array<T> &xpointing_gpu,
			   const Array<unsigned char> &buf_, const Array<unsigned char> &tmp_buf) :
    nsamp(preplan.nsamp),
    nypix(preplan.nypix),
    nxpix(preplan.nxpix),
    pp(preplan),
    buf(buf_)
{
    check_buffer(buf, preplan.plan_nbytes, "PointingPlan constructor", "buf");
    check_buffer(tmp_buf, preplan.plan_constructor_tmp_nbytes, "PointingPlan constructor", "tmp_buf");
    check_xpointing(xpointing_gpu, preplan.nsamp, "PointingPlan constructor", true);   // on_gpu=true
    
    long mt_nbytes = align128(pp.plan_nmt * sizeof(ulong));
    size_t cub_nbytes = pp.cub_nbytes;
    
    this->plan_mt = (ulong *) (buf.data);
    this->err_gpu = (uint *) (buf.data + mt_nbytes);
    this->err_cpu = Array<uint> ({pp.nblocks}, af_rhost | af_zero);

    ulong *unsorted_mt = (ulong *) (tmp_buf.data);
    void *cub_tmp = (void *) (tmp_buf.data + mt_nbytes);

    // Number of warps in plan_kernel.
    constexpr int W = 4;

    plan_kernel<T,W> <<< pp.nblocks, {32,W} >>>
	(unsorted_mt,             // ulong *plan_mt,
	 xpointing_gpu.data,      // const T *xpointing,
	 pp.nmt_cumsum.data,      // uint *nmt_cumsum,
	 pp.nsamp,                // uint nsamp,
	 1 << pp.rk,              // uint nsamp_per_block,
	 pp.nypix,                // int nypix,
	 pp.nxpix,                // int nxpix,
	 this->err_gpu);          // uint *errp)

    CUDA_PEEK("plan_kernel launch");

    CUDA_CALL(cudaMemcpyAsync(err_cpu.data, err_gpu, pp.nblocks * sizeof(uint), cudaMemcpyDeviceToHost));

    CUDA_CALL(cub::DeviceRadixSort::SortKeys(
        cub_tmp,         // void *d_temp_storage
	cub_nbytes,      // size_t &temp_storage_bytes
	unsorted_mt,     // const KeyT *d_keys_in
	this->plan_mt,   // KeyT *d_keys_out
	pp.plan_nmt,     // NumItemsT num_items
	0,               // int begin_bit = 0
	20               // int end_bit = sizeof(KeyT) * 8
	// cudaStream_t stream = 0
    ));
    
    cudaDeviceSynchronize();

    uint err = 0;
    for (int b = 0; b < pp.nblocks; b++)
	err |= err_cpu.data[b];

    check_err(err, "PointingPlan constructor");
}


// This constructor allocates GPU memory (rather than using externally managed GPU memory)
template<typename T>
PointingPlan::PointingPlan(const PointingPrePlan &preplan, const Array<T> &xpointing_gpu) :
    PointingPlan(preplan, xpointing_gpu,
		 Array<unsigned char>({preplan.plan_nbytes}, af_gpu), 
		 Array<unsigned char>({preplan.plan_constructor_tmp_nbytes}, af_gpu))
{ }



template<typename T>
void PointingPlan::map2tod(Array<T> &tod, const Array<T> &map, const Array<T> &xpointing, bool debug) const
{
    check_map(map, nypix, nxpix, "PointingPlan::map2tod", true);          // on_gpu=true
    check_tod(tod, nsamp, "PointingPlan::map2tod", true);                 // on_gpu=true
    check_xpointing(xpointing, nsamp, "PointingPlan::map2tod", true);     // on_gpu=true

    // FIXME revisit?
    int nmt_per_block = 1 << pp.rk;
    
    launch_map2tod2(tod.data, map.data, xpointing.data, this->plan_mt,
		    this->nsamp, this->nypix, this->nxpix,
		    this->pp.plan_nmt, nmt_per_block, debug);
}


template<typename T>
void PointingPlan::tod2map(Array<T> &map, const Array<T> &tod, const Array<T> &xpointing, bool debug) const
{
    check_map(map, nypix, nxpix, "PointingPlan::tod2map", true);          // on_gpu=true
    check_tod(tod, nsamp, "PointingPlan::tod2map", true);                 // on_gpu=true
    check_xpointing(xpointing, nsamp, "PointingPlan::tod2map", true);     // on_gpu=true

    // FIXME revisit?
    int nmt_per_block = 1 << pp.rk;
    
    launch_tod2map2(map.data, tod.data, xpointing.data, this->plan_mt,
		    this->nsamp, this->nypix, this->nxpix,
		    this->pp.plan_nmt, nmt_per_block, debug);
}


// Only used in unit tests
Array<ulong> PointingPlan::get_plan_mt(bool gpu) const
{
    int aflags = gpu ? af_gpu : af_rhost;
    cudaMemcpyKind direction = gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    
    Array<ulong> ret({pp.plan_nmt}, aflags);
    CUDA_CALL(cudaMemcpy(ret.data, this->plan_mt, pp.plan_nmt * sizeof(ulong), direction));
    return ret;
}


string PointingPlan::str() const
{
    long plan_ntt = (nsamp / 32);
    double ratio = double(pp.plan_nmt) / double(plan_ntt);
    
    stringstream ss;

    // FIXME reduce cut-and-paste with PointingPrePlan::str()
    ss << "PointingPlan(nsamp=" << nsamp << ", nypix=" << nypix << ", nxpix=" << nxpix
       << ", rk=" << pp.rk << ", nblocks=" << pp.nblocks
       << ", ntt=" << plan_ntt << ", nmt=" << pp.plan_nmt << ", ratio=" << ratio
       << ", plan_nbytes=" << pp.plan_nbytes << " (" << nbytes_to_str(pp.plan_nbytes) << ")"
       << ")";

    return ss.str();
}



// -------------------------------------------------------------------------------------------------


#define INSTANTIATE(T) \
    template PointingPlan::PointingPlan( \
	const PointingPrePlan &pp, \
	const gputils::Array<T> &xpointing_gpu, \
	const gputils::Array<unsigned char> &buf, \
	const gputils::Array<unsigned char> &tmp_buf); \
    template PointingPlan::PointingPlan( \
	const PointingPrePlan &pp, \
	const gputils::Array<T> &xpointing_gpu); \
    template void PointingPlan::map2tod( \
	gputils::Array<T> &tod, \
	const gputils::Array<T> &map, \
	const gputils::Array<T> &xpointing, \
	bool debug) const; \
    template void PointingPlan::tod2map( \
	gputils::Array<T> &map, \
	const gputils::Array<T> &tod, \
	const gputils::Array<T> &xpointing, \
	bool debug) const


INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
