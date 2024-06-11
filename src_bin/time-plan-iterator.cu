#include "../include/gpu_mm2.hpp"
#include "../include/gpu_mm2_internals.hpp"
#include "../include/PlanIterator.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>
#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm2;


template<int W, bool Debug>
__global__ void iterator_timing_kernel(ulong *plan_mt, uint nmt, uint nmt_per_block, uint *out)
{
    PlanIterator<W,Debug> iterator;
    uint ret = 0;

    if (!iterator.init(plan_mt, nmt, nmt_per_block))
	return;

    do {
	ret ^= iterator.icell_curr;	
	while (iterator.next_mt())
	    ret ^= iterator.icl_curr;
    } while (iterator.next_cell());

    int s = (32*W * blockIdx.x) + (32 * threadIdx.y) + threadIdx.x;
    out[s] = ret;
}


template<int W, bool Debug>
static void time_plan_iterator(const Array<ulong> &plan_mt, uint nmt_per_block)
{
    uint nmt = plan_mt.size;
    uint nblocks = (nmt + nmt_per_block - 1) / nmt_per_block;
    const char *dstr = Debug ? "true" : "false";

    Array<uint> out({nblocks,W,32}, af_gpu);

    for (int i = 0; i < 20; i++) {
	struct timeval tv0 = get_time();

	iterator_timing_kernel<W,Debug> <<<nblocks, {32,W}>>>
	    (plan_mt.data, nmt, nmt_per_block, out.data);
	
	CUDA_PEEK("iterator_timing_kernel launch");
	CUDA_CALL(cudaDeviceSynchronize());
	
	double dt = time_since(tv0);
	cout << "PlanIterator<Debug=" << dstr <<">: " << dt << " seconds" << endl;
    }

}


int main(int argc, char **argv)
{
    constexpr int W = 4;
    
    long nsamp = 256*1024*1024;
    long nypix = 8*1024;
    long nxpix = 32*1024;
    double scan_speed = 0.5;    // pixels per TOD sample
    double total_drift = 1024;  // x-pixels
    int nmt_per_block = 8192;

    ToyPointing<float> tp(nsamp, nypix, nxpix, scan_speed, total_drift);
    PointingPrePlan pp(tp.xpointing_gpu, nypix, nxpix);
    PointingPlan p(pp, tp.xpointing_gpu);
    Array<ulong> plan_mt = p.get_plan_mt(true);  // gpu=true

    time_plan_iterator<W,true> (plan_mt, nmt_per_block);
    time_plan_iterator<W,false> (plan_mt, nmt_per_block);

    return 0;
}
