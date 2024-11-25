#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/string_utils.hpp>
#include <ksgpu/constexpr_functions.hpp>   // constexpr_is_log2()
#include <cub/device/device_radix_sort.cuh>

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// Helper for planning kernel.
struct cell_analysis
{
    uint icell;
    uint amask;
    int na;
    
    __device__ cell_analysis(int iycell, int ixcell)
    {
	icell = (iycell << 10) | ixcell;
	uint mmask = __match_any_sync(ALL_LANES, icell);  // all matching lanes
	
	// Block dims are (W,32), so threadIdx.x is the laneId.
	bool is_lowest = ((mmask >> threadIdx.x) == 1);
	bool valid = (iycell >= 0) && (ixcell >= 0);
	
	amask = __ballot_sync(ALL_LANES, valid && is_lowest);
	na = __popc(amask);	
    }
};


// Helper for planning kernel.
struct mt_ringbuf
{
    ulong *plan_mt;   // global memory
    int *shmem;       // shared memory counter
    int nmt_max;      // length of global memory array
    ulong mt_local;   // per-warp ring buffer
    int nmt_local;    // current size of per-warp ring buffer
    int na_cumul;

    __device__ mt_ringbuf(ulong *plan_mt_, int *shmem_, int nmt_max_)
    {
	plan_mt = plan_mt_;
	shmem = shmem_;
	nmt_max = nmt_max_;
	mt_local = 0;
	nmt_local = 0;
	na_cumul = 0;
    }
    
    __device__ void absorb(const cell_analysis &ca, uint s, bool mflag, uint &err)
    {
	if (ca.na == 0)
	    return;
    
	// Block dims are (W,32), so threadIdx.x is the laneId.
	int laneId = threadIdx.x;
    
	// Logical laneId (relative to current value of nmt_local, wrapped around)
	int llid = (laneId + 32 - nmt_local) & 31;
    
	// Permute 'icell' so that llid=N contains the N-th active icell
	uint src_lane = __fns(ca.amask, 0, llid+1);
	uint icell = __shfl_sync(ALL_LANES, ca.icell, src_lane & 31);  // FIXME do I need "& 31"?

	// Reminder: mt bit layout is
	//   Low 10 bits = Global xcell index
	//   Next 10 bits = Global ycell index
	//   Next 26 bits = Primary TOD cache line index
	//   Next bit = mflag (does cache line overlap multiple map cells?)
	//   Next bit = zflag (mflag && first appearance of cache line)
	
	bool zflag = mflag && (na_cumul == 0) && (llid == 0);

	// Construct mt_new from icell, s, mflag, zflag.
	uint mt20 = (s >> 5);
	mt20 |= (mflag ? (1U << 26) : 0);
	mt20 |= (zflag ? (1U << 27) : 0);
	
	ulong mt_new = icell | (ulong(mt20) << 20);

	// Extend ring buffer.
	// If nmt_local is >32, then it "wraps around" from mt_local to mt_new.
	
	mt_local = (laneId < nmt_local) ? mt_local : mt_new;
	nmt_local += ca.na;

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
	na_cumul += ca.na;
	err = (nout < nmt_max) ? err : (err | errflag_inconsistent_nmt);
    }
};


template<typename T, int W, bool Debug>
__global__ void plan_kernel(
    ulong *plan_mt,
    const T *xpointing,
    const uint *nmt_cumsum,
    uint *errflags,
    uint nsamp,
    uint nsamp_per_block,
    int nypix_global,
    int nxpix_global,
    bool periodic_xcoord)
{
    // Shared memory layout:
    //   int   nmt_counter          running total of mt-values written by this block (updated with AtomicAdd())
    //   uint  nmt_unreduced[W]     only used at end of kernel
    //   uint  err_unreduced[W]     only used at end of kernel
    
    __shared__ int shmem[2*W+1];
    
    // Block dims are (W,32)
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    
    // Just need to zero 'nmt_counter'.
    if ((warpId == 0) && (laneId == 0))
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
    
    // cell_enumerator is defined in gpu_mm_internals.hpp
    cell_enumerator<T,Debug> cells(nypix_global, nxpix_global, periodic_xcoord);
    mt_ringbuf rb(plan_mt, shmem, nmt_max);
    uint err = 0;

    for (uint s = s0 + 32*warpId + laneId; s < s1; s += 32*W) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];

	cells.enumerate(ypix, xpix, err);

	cell_analysis ca0(cells.iy0, cells.ix0);
	cell_analysis ca1(cells.iy0, cells.ix1);
	cell_analysis ca2(cells.iy0, cells.ix2);
	cell_analysis ca3(cells.iy1, cells.ix0);
	cell_analysis ca4(cells.iy1, cells.ix1);
	cell_analysis ca5(cells.iy1, cells.ix2);

	bool mflag = ((ca0.na + ca1.na + ca2.na + ca3.na + ca4.na + ca5.na) > 1);
	
	rb.absorb(ca0, s, mflag, err);
	rb.absorb(ca1, s, mflag, err);
	rb.absorb(ca2, s, mflag, err);
	rb.absorb(ca3, s, mflag, err);
	rb.absorb(ca4, s, mflag, err);
	rb.absorb(ca5, s, mflag, err);
	rb.na_cumul = 0;  // reset after every TOD cache line
    }

    // Done with main loop -- now we just need to write errflag and partial
    // mt_ringbufs to global memory. First, we write per-warp values of
    // (err, rb.nmt_local) to shared memory (after warp-reducing err).

    err = __reduce_or_sync(ALL_LANES, err);
    
    if (laneId == 0) {
	shmem[warpId+1] = rb.nmt_local;
	shmem[warpId+W+1] = err;
    }

    __syncthreads();

    // Read values of rb.nmt_local (from all warps) from shared memory.
    int nmt_remote = (laneId < W) ? shmem[laneId+1] : 0;
    int nmt_counter = shmem[0];

    // Index for writing to mt_ringbuf to global memory.
    int imt = (laneId < warpId) ? nmt_remote : 0;
    imt = __reduce_add_sync(ALL_LANES, imt);
    imt += nmt_counter;
    imt += laneId;

    // Write mt_ringbuf to global memory.
    if ((laneId < rb.nmt_local) && (imt < nmt_max))
	plan_mt[imt] = rb.mt_local;

    if (warpId != 0)
	return;

    // Just need to fully reduce 'err' and write to global memory.
    err = (laneId < W) ? shmem[laneId+W+1] : 0;
    err = __reduce_or_sync(ALL_LANES, err);

    // Check total number of mt-values written by kernel.
    int nmt_tot = nmt_counter + __reduce_add_sync(ALL_LANES, nmt_remote);
    err = (nmt_tot == nmt_max) ? err : (err | errflag_inconsistent_nmt);

    if (laneId == 0)
	errflags[b] = err;
}


// -------------------------------------------------------------------------------------------------


template<typename T>
PointingPlan::PointingPlan(const PointingPrePlan &preplan, const Array<T> &xpointing_gpu,
			   const Array<unsigned char> &buf_, const Array<unsigned char> &tmp_buf, bool debug)
    : nsamp(preplan.nsamp),
      nypix_global(preplan.nypix_global),
      nxpix_global(preplan.nxpix_global),
      periodic_xcoord(preplan.periodic_xcoord),
      pp(preplan),
      buf(buf_)
{
    check_buffer(buf, preplan.plan_nbytes, "PointingPlan constructor", "buf");
    check_buffer(tmp_buf, preplan.plan_constructor_tmp_nbytes, "PointingPlan constructor", "tmp_buf");
    check_xpointing(xpointing_gpu, preplan.nsamp, "PointingPlan constructor", true);   // on_gpu=true

    long max_nblocks = max(preplan.planner_nblocks, preplan.pointing_nblocks);
    long mt_nbytes = align128(preplan.plan_nmt * sizeof(ulong));
    long err_nbytes = align128(max_nblocks * sizeof(uint));
    size_t cub_nbytes = pp.cub_nbytes;
    
    xassert(preplan.plan_nbytes == mt_nbytes + err_nbytes);
    xassert(preplan.plan_constructor_tmp_nbytes == mt_nbytes + align128(cub_nbytes));

    this->plan_mt = (ulong *) (buf.data);
    this->err_gpu = (uint *) (buf.data + mt_nbytes);

    // Set errflags to zero. This isn't logically necessary, but can make debugging less confusing.
    cudaMemsetAsync(this->err_gpu, 0, err_nbytes);

    ulong *unsorted_mt = (ulong *) (tmp_buf.data);
    void *cub_tmp = (void *) (tmp_buf.data + mt_nbytes);

    // Number of warps in plan_kernel.
    constexpr int W = 4;

    // Launch plan_kernel.
    if (debug) {
	plan_kernel<T,W,true> <<< pp.planner_nblocks, {32,W} >>>
	    (unsorted_mt,                   // ulong *plan_mt,
	     xpointing_gpu.data,            // const T *xpointing,
	     pp.nmt_cumsum.data,            // const uint *nmt_cumsum,
	     this->err_gpu,                 // uint *errflags
	     pp.nsamp,                      // uint nsamp,
	     pp.ncl_per_threadblock << 5,   // uint nsamp_per_block (FIXME 32-bit overflow)
	     pp.nypix_global,               // int nypix_global
	     pp.nxpix_global,               // int nxpix_global
	     pp.periodic_xcoord);           // bool periodic_xcoord
    }
    else {
	plan_kernel<T,W,false> <<< pp.planner_nblocks, {32,W} >>>
	    (unsorted_mt,                   // ulong *plan_mt,
	     xpointing_gpu.data,            // const T *xpointing,
	     pp.nmt_cumsum.data,            // const uint *nmt_cumsum,
	     this->err_gpu,                 // uint *errflags
	     pp.nsamp,                      // uint nsamp,
	     pp.ncl_per_threadblock << 5,   // uint nsamp_per_block (FIXME 32-bit overflow)
	     pp.nypix_global,               // int nypix_global
	     pp.nxpix_global,               // int nxpix_global
	     pp.periodic_xcoord);           // bool periodic_xcoord
    }
    
    CUDA_PEEK("plan_kernel launch");

    // Launch NVIDIA radix sort.
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
    
    check_gpu_errflags(this->err_gpu, pp.planner_nblocks, "PointingPlan constructor");
}


// This constructor allocates GPU memory (rather than using externally managed GPU memory)
template<typename T>
PointingPlan::PointingPlan(const PointingPrePlan &preplan, const Array<T> &xpointing_gpu, bool debug_) :
    PointingPlan(preplan, xpointing_gpu,
		 Array<unsigned char>({preplan.plan_nbytes}, af_gpu), 
		 Array<unsigned char>({preplan.plan_constructor_tmp_nbytes}, af_gpu),
		 debug_)
{ }


// Only used in unit tests
Array<ulong> PointingPlan::get_plan_mt(bool gpu) const
{
    int aflags = gpu ? af_gpu : af_rhost;
    cudaMemcpyKind direction = gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    
    Array<ulong> ret({pp.plan_nmt}, aflags);
    CUDA_CALL(cudaMemcpy(ret.data, this->plan_mt, pp.plan_nmt * sizeof(ulong), direction));
    return ret;
}


// I needed this once for tracking down a bug.
void PointingPlan::_check_errflags(const string &where) const
{
    long max_nblocks = max(pp.planner_nblocks, pp.pointing_nblocks);
    check_gpu_errflags(this->err_gpu, max_nblocks, where.c_str());
}


string PointingPlan::str() const
{
    // FIXME reduce cut-and-paste with PointingPrePlan::str()
    stringstream ss;
    
    ss << "PointingPlan("
       << "nsamp=" << nsamp
       << ", nypix_global=" << nypix_global
       << ", nxpix_global=" << nxpix_global
       << ", plan_nbytes=" << pp.plan_nbytes << " (" << nbytes_to_str(pp.plan_nbytes) << ")"
       << ", tmp_nbytes=" << pp.plan_constructor_tmp_nbytes << " (" << nbytes_to_str(pp.plan_constructor_tmp_nbytes) << ")"
       << ", overhead=" << pp.overhead
       << ", nmt_per_threadblock=" << pp.nmt_per_threadblock
       << ", pointing_nblocks=" << pp.pointing_nblocks
       << ")";

    return ss.str();
}


// -------------------------------------------------------------------------------------------------


#define INSTANTIATE(T) \
    template PointingPlan::PointingPlan( \
	const PointingPrePlan &pp, \
	const ksgpu::Array<T> &xpointing_gpu, \
	const ksgpu::Array<unsigned char> &buf, \
	const ksgpu::Array<unsigned char> &tmp_buf, \
        bool debug);  \
    \
    template PointingPlan::PointingPlan( \
	const PointingPrePlan &pp, \
	const ksgpu::Array<T> &xpointing_gpu, \
	bool debug)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
