#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"  // ALL_LANES
#include "../include/plan_iterator.hpp"

#include <vector>
#include <iostream>
#include <algorithm> // std::sort
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>
#include <ksgpu/string_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// This file just contains helper functions for testing!


template<typename T>
__device__ void assert_equal_within_warp(T x)
{
    uint mask = __match_any_sync(ALL_LANES, x);
    assert(mask == ALL_LANES);
}


__device__ void assert_equal_within_block(uint *sp, uint x)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int W = blockDim.y;
    
    if (laneId == 0)
	sp[warpId] = x;

    __syncthreads();

    if (warpId == 0) {
	uint y = sp[min(laneId,W-1)];
	assert_equal_within_warp(y);
    }
    
    __syncthreads();	
}


template<int W>
__global__ void iterator_test_kernel(ulong *plan_mt, uint nmt, uint nmt_per_block, int *out_mt_counts)
{
    __shared__ uint shmem[W];
	
    assert(blockDim.x == 32);
    assert(blockDim.y == W);

    int laneId = threadIdx.x;
    // int warpId = threadIdx.y;
    
    plan_iterator<W,true> iterator(plan_mt, nmt, nmt_per_block);
    
    while (iterator.get_cell()) {
	uint icell = iterator.icell;
	assert(icell < (1U << 20));
	assert_equal_within_block(shmem, icell);

	for (;;) {
	    uint imt = iterator.imt_next;

	    if (!iterator.get_cl())
		break;

	    uint icl = iterator.icl;
	    assert(icl < (1U << 26));
	    assert(imt < nmt);
	    
	    assert_equal_within_warp(imt);
	    assert_equal_within_warp(icl);
	    assert(iterator.icell == icell);

	    ulong mt = plan_mt[imt];
	    uint mt_icell = uint(mt) & ((1U << 20) - 1);
	    uint mt_icl = uint(mt >> 20) & ((1U << 26) - 1);
	    
	    assert(icell == mt_icell);
	    assert(icl == mt_icl);

	    if (laneId == 0)
		atomicAdd(out_mt_counts + imt, 1);
	}

	// No __syncthreads() needed, since assert_equal_within_block() calls __syncthreads().
    }

    uint sentinel = 1U << 27;
    assert_equal_within_block(shmem, sentinel);
}


// Helper for test_plan_iterator()
template<int W>
static void launch_iterator_test_kernel(const Array<ulong> &plan_mt, Array<int> &mt_counts, int nmt_per_block)
{
    int nmt = plan_mt.size;
    int nblocks = (nmt + nmt_per_block - 1) / nmt_per_block;
    
    iterator_test_kernel<W> <<< nblocks, {32,W} >>>
	(plan_mt.data, nmt, nmt_per_block, mt_counts.data);
}


// -------------------------------------------------------------------------------------------------
//
// FIXME code below could get rewritten in python eventually


void test_plan_iterator(const Array<ulong> &plan_mt, uint nmt_per_block, int warps_per_threadblock)
{
    xassert(plan_mt.ndim == 1);
    xassert(plan_mt.is_fully_contiguous());
    xassert(plan_mt.size > 0);
    xassert(nmt_per_block > 0);

    Array<ulong> plan_mt_cpu = plan_mt.to_host();
    Array<ulong> plan_mt_gpu = plan_mt.to_gpu();
    
    long nmt = plan_mt_cpu.size;
    ulong *mt = plan_mt_cpu.data;

    // A little error checking on the plan
    
    for (long i = 1; i < nmt; i++) {
	uint icell0 = uint(mt[i-1]) & ((1<<20) - 1);
	uint icell1 = uint(mt[i]) & ((1<<20) - 1);
	xassert(icell0 <= icell1);
    }

    // Output array
    
    Array<int> mt_counts({nmt}, af_gpu | af_zero);
    
    // Launch kernel
    
    if (warps_per_threadblock == 4)
	launch_iterator_test_kernel<4> (plan_mt, mt_counts, nmt_per_block);
    else if (warps_per_threadblock == 8)
	launch_iterator_test_kernel<8> (plan_mt, mt_counts, nmt_per_block);
    else if (warps_per_threadblock == 12)
	launch_iterator_test_kernel<12> (plan_mt, mt_counts, nmt_per_block);
    else if (warps_per_threadblock == 16)
	launch_iterator_test_kernel<16> (plan_mt, mt_counts, nmt_per_block);
    else
	throw runtime_error("test_plan_iterator: unsupported value of warps_per_threadblock");

    CUDA_PEEK("iterator_test kernel launch");
    CUDA_CALL(cudaDeviceSynchronize());

    // Check results
    
    mt_counts = mt_counts.to_host();
    
    for (long i = 0; i < nmt; i++)
	xassert(mt_counts.data[i] == 1);
}


Array<ulong> make_random_plan_mt(long ncells, long min_nmt_per_cell, long max_nmt_per_cell)
{
    assert(ncells > 0);
    assert(ncells <= (1<<20));
    assert(min_nmt_per_cell >= 1);
    assert(min_nmt_per_cell <= max_nmt_per_cell);
    assert(max_nmt_per_cell <= 16*1024);
	   
    auto all_cells = rand_permutation(1<<20);
    
    vector<uint> cells(ncells);
    vector<uint> nmt_per_cell(ncells);
    long nmt_tot = 0;
    
    for (int i = 0; i < ncells; i++) {
	cells[i] = all_cells[i];
	int nmt = rand_int(min_nmt_per_cell, max_nmt_per_cell+1);
	nmt_per_cell[i] = nmt;
	nmt_tot += nmt;
    }

    std::sort(cells.begin(), cells.end());

    assert(nmt_tot <= (1<<30));
    Array<ulong> plan_mt({nmt_tot}, af_rhost);

    long imt = 0;
    for (int i = 0; i < ncells; i++) {
	ulong icell = cells[i];
	for (uint j = 0; j < nmt_per_cell[i]; j++) {
	    ulong icl = rand_int(0, 1<<26);
	    ulong sec = rand_int(0, 1<<18);  // FIXME arbitrary for now
	    assert(imt < nmt_tot);
	    plan_mt.data[imt++] = icell | (icl << 20) | (sec << 46);
	}
    }
    assert(imt == nmt_tot);

    return plan_mt;
}


}  // namespace gpu_mm
