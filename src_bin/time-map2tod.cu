#include "../include/gpu_mm.hpp"

#include <iostream>
#include <gputils/CudaStreamPool.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm;

int main(int argc, char **argv)
{
    if (argc != 2) {
	cerr << "Usage: " << argv[0] << " /home/kmsmith/xpointing/xpointing_0.npz" << endl;
	exit(2);
    }
    
    ActPointing ap(argv[1]);

    gputils::Array<float> xpointing_gpu = ap.xpointing.to_gpu();
    gputils::Array<float> map_gpu({3,ap.ndec,ap.nra}, af_gpu | af_zero);
    gputils::Array<float> tod_gpu({ap.ndet,ap.nt}, af_gpu | af_zero);

    double gb = 1.0e-9 * (tod_gpu.size + xpointing_gpu.size) * sizeof(float);
    int launches_per_callback = 20;
    int num_callbacks = 10;

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
    {
	for (int i = 0; i < launches_per_callback; i++)
	    launch_map2tod(tod_gpu, map_gpu, xpointing_gpu, stream);
    };

    gputils::CudaStreamPool sp(callback, num_callbacks, 1, "map2tod");  // nstreams=1
    sp.monitor_throughput("Global memory BW (GB/s, map not included)", gb * launches_per_callback);
    sp.monitor_time("Time per kernel (sec)", launches_per_callback);
    sp.run();
    
    return 0;
}
