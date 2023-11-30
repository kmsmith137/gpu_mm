#include "../include/gpu_mm.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>
#include <gputils/test_utils.hpp>

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

    gputils::Array<float> xpointing_cpu = ap.xpointing;
    gputils::Array<float> map_cpu({3,ap.ndec,ap.nra}, af_rhost | af_random);
    gputils::Array<float> tod_cpu({ap.ndet,ap.nt}, af_rhost | af_zero);

    reference_map2tod(tod_cpu, map_cpu, xpointing_cpu);
    
    gputils::Array<float> xpointing_gpu = xpointing_cpu.to_gpu();
    gputils::Array<float> map_gpu = map_cpu.to_gpu();
    gputils::Array<float> tod_gpu({ap.ndet,ap.nt}, af_gpu | af_zero);

    launch_map2tod(tod_gpu, map_gpu, xpointing_gpu);
    tod_gpu = tod_gpu.to_host();

    assert_arrays_equal(tod_cpu, tod_gpu, "tod_cpu", "tod_gpu", {"det","tod"});
    cout << "test-map2tod: pass" << endl;
    
    return 0;
}
