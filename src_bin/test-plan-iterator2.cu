#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"  // ALL_LANES
#include "../include/PlanIterator2.hpp"

#include <vector>
#include <iostream>
#include <algorithm> // std::sort
#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm;


int main(int argc, char **argv)
{
    int num_iterations = 400;
    
    for (int i = 0; i < num_iterations; i++) {
	int ncells = rand_int(100, 1000);
	int min_nmt_per_cell = 1;
	int max_nmt_per_cell = 1000;
	int nmt_per_block = rand_int(1,1000);
	int warps_per_threadblock = 1 << rand_int(2,5);
	Array<ulong> plan_mt = make_random_plan_mt(ncells, min_nmt_per_cell, max_nmt_per_cell);
	
	cout << "Random plan: ncells=" << ncells
	     << ", min_nmt_per_cell=" << min_nmt_per_cell
	     << ", max_nmt_per_cell=" << max_nmt_per_cell
	     << ", nmt=" << plan_mt.size
	     << ", nmt_per_block=" << nmt_per_block
	     << ", warps_per_threadblock=" << warps_per_threadblock
	     << endl;
	
	test_plan_iterator(plan_mt, nmt_per_block, warps_per_threadblock);
    }

    do {
	long nsamp = 256*1024*1024;
	long nypix = 8*1024;
	long nxpix = 32*1024;
	double scan_speed = 0.5;    // pixels per TOD sample
	double total_drift = 1024;  // x-pixels
	int nmt_per_block = 256*1024;
	int warps_per_threadblock = 16;
	
	ToyPointing<float> tp(nsamp, nypix, nxpix, scan_speed, total_drift);
	PointingPrePlan pp(tp.xpointing_gpu, nypix, nxpix);
	PointingPlan p(pp, tp.xpointing_gpu);
	Array<ulong> plan_mt = p.get_plan_mt(true);  // gpu=true
	
	test_plan_iterator(plan_mt, nmt_per_block, warps_per_threadblock);
    } while (0);
	
    return 0;
}
