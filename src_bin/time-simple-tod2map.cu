#include "../include/gpu_mm2.hpp"

#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>
#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm2;

template<typename T>
static void time_simple_tod2map()
{
    long nsamp = 256*1024*1024;
    long nypix = 8*1024;
    long nxpix = 32*1024;
    
    double scan_speed = 0.5;    // pixels per TOD sample
    double total_drift = 1024;  // x-pixels

    ToyPointing<T> tp(nsamp, nypix, nxpix, scan_speed, total_drift);
    Array<T> map({3,nypix,nxpix}, af_gpu | af_zero);
    Array<T> tod({nsamp}, af_gpu | af_zero);

    for (int i = 0; i < 20; i++) {
	struct timeval tv0 = get_time();

	launch_simple_tod2map(map, tod, tp.xpointing_gpu);
	CUDA_CALL(cudaDeviceSynchronize());
	
	double dt = time_since(tv0);
	cout << "simple_tod2map<" << type_name<T>() << ">: " << dt << " seconds" << endl;
    }
}


int main(int argc, char **argv)
{
    time_simple_tod2map<float>();
    time_simple_tod2map<double>();
}
