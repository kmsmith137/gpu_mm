#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"  // ALL_LANES
#include "../include/PlanIterator.hpp"

#include <vector>
#include <iostream>
#include <algorithm> // std::sort
#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm;


// -------------------------------------------------------------------------------------------------


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
__global__ void strict_iterator_test_kernel(ulong *plan_mt, uint nmt, uint nmt_per_block, int *out_mt_counts, int *out_cell_counts)
{
    __shared__ uint shmem[W];
	
    assert(blockDim.x == 32);
    assert(blockDim.y == W);

    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    
    plan_iterator_strict<W,true> iterator;

    if (!iterator.init(plan_mt, nmt, nmt_per_block))
	return;

    do {
	uint icell = iterator.icell_curr;
	assert(icell < (1U << 20));
	assert_equal_within_block(shmem, icell);

	if ((warpId == 0) && (laneId == 0))
	    atomicAdd(out_cell_counts + iterator.icell_curr, 1);

	uint imt = iterator.imt_next;
	
	while (iterator.next_mt()) {
	    uint icl = iterator.icl_curr;
	    assert(icl < (1U << 26));
	    assert(imt < nmt);
	    
	    assert_equal_within_warp(imt);
	    assert_equal_within_warp(icl);
	    assert(iterator.icell_curr == icell);

	    ulong mt = plan_mt[imt];
	    uint mt_icell = uint(mt) & ((1U << 20) - 1);
	    uint mt_icl = uint(mt >> 20) & ((1U << 26) - 1);
	    
	    assert(icell == mt_icell);
	    assert(icl == mt_icl);

	    if (laneId == 0)
		atomicAdd(out_mt_counts + imt, 1);

	    imt = iterator.imt_next;
	}
    } while (iterator.next_cell());

    uint sentinel = 1U << 27;
    assert_equal_within_block(shmem, sentinel);
}


// -------------------------------------------------------------------------------------------------


static void test_strict_iterator(const Array<ulong> plan_mt, uint nmt_per_block)
{
    assert(plan_mt.ndim == 1);
    assert(plan_mt.is_fully_contiguous());
    assert(plan_mt.size > 0);
    assert(nmt_per_block > 0);
    assert((nmt_per_block % 32) == 0);

    Array<ulong> plan_mt_cpu = plan_mt.to_host();
    Array<ulong> plan_mt_gpu = plan_mt.to_gpu();
    
    long nmt = plan_mt_cpu.size;
    ulong *mt = plan_mt_cpu.data;

    // A little error checking on the plan
    for (long i = 1; i < nmt; i++) {
	uint icell0 = uint(mt[i-1]) & ((1<<20) - 1);
	uint icell1 = uint(mt[i]) & ((1<<20) - 1);
	assert(icell0 <= icell1);
    }

    // Output arrays
    Array<int> mt_counts({nmt}, af_gpu | af_zero);
    Array<int> cell_counts({1<<20}, af_gpu | af_zero);
    Array<int> expected_cell_counts({1<<20}, af_uhost | af_zero);

    // Initialize expected_cell_counts
    for (long i = 0; i < nmt; i++) {
	uint icell = uint(mt[i]) & ((1<<20) - 1);
	expected_cell_counts.data[icell] = 1;
    }
    
    // Launch kernel
    
    int nblocks = (nmt + nmt_per_block - 1) / nmt_per_block;
    constexpr int W = 16;
    
    strict_iterator_test_kernel<W> <<< nblocks, {32,W} >>>
	(plan_mt_gpu.data, nmt, nmt_per_block, mt_counts.data, cell_counts.data);

    CUDA_PEEK("iterator test kernel launch");
    CUDA_CALL(cudaDeviceSynchronize());
    
    mt_counts = mt_counts.to_host();
    cell_counts = cell_counts.to_host();

    // Check results
    
    for (long i = 0; i < nmt; i++)
	assert(mt_counts.data[i] == 1);
    for (long i = 0; i < (1<<20); i++)
	assert(cell_counts.data[i] == expected_cell_counts.data[i]);

    cout << "test_strict_iterator_strict: pass" << endl;
}


static Array<ulong> make_random_plan_mt(long ncells, long min_nmt_per_cell, long max_nmt_per_cell)
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


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    int num_iterations = 400;
    
    for (int i = 0; i < num_iterations; i++) {
	int ncells = rand_int(100, 1000);
	int min_nmt_per_cell = 1;
	int max_nmt_per_cell = 1000;
	int nmt_per_block = 32 * rand_int(1,20);
	Array<ulong> plan_mt = make_random_plan_mt(ncells, min_nmt_per_cell, max_nmt_per_cell);
	
	cout << "Random plan: ncells=" << ncells
	     << ", min_nmt_per_cell=" << min_nmt_per_cell
	     << ", max_nmt_per_cell=" << max_nmt_per_cell
	     << ", nmt=" << plan_mt.size
	     << ", nmt_per_block=" << nmt_per_block << endl;
	
	test_strict_iterator(plan_mt, nmt_per_block);
    }

    do {
	long nsamp = 256*1024*1024;
	long nypix = 8*1024;
	long nxpix = 32*1024;
	double scan_speed = 0.5;    // pixels per TOD sample
	double total_drift = 1024;  // x-pixels
	int nmt_per_block = 256*1024;
	
	ToyPointing<float> tp(nsamp, nypix, nxpix, scan_speed, total_drift);
	PointingPrePlan pp(tp.xpointing_gpu, nypix, nxpix);
	PointingPlan p(pp, tp.xpointing_gpu);
	Array<ulong> plan_mt = p.get_plan_mt(true);  // gpu=true
	
	test_strict_iterator(plan_mt, nmt_per_block);  // last argument is nmt_per_block
    } while (0);
	
    return 0;
}
