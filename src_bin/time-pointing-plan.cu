#include "../include/gpu_mm2.hpp"

#include <iostream>
#include <gputils/cuda_utils.hpp>
#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;
using namespace gpu_mm2;


template<typename T>
static void time_pointing_plan()
{
    long nsamp = 256*1024*1024;
    long nypix = 8*1024;
    long nxpix = 32*1024;
    
    double scan_speed = 0.5;    // pixels per TOD sample
    double total_drift = 1024;  // x-pixels

    ToyPointing<T> tp(nsamp, nypix, nxpix, scan_speed, total_drift);
    PointingPrePlan pp(tp.xpointing_gpu, nypix, nxpix);

    Array<unsigned char> buf({pp.plan_nbytes}, af_gpu);
    Array<unsigned char> tmp_buf({pp.plan_constructor_tmp_nbytes}, af_gpu);
    
    for (int i = 0; i < 20; i++) {
	struct timeval tv0 = get_time();

	PointingPlan p(pp, tp.xpointing_gpu, buf, tmp_buf);
	CUDA_CALL(cudaDeviceSynchronize());
	
	double dt = time_since(tv0);
	cout << "PointingPlan<" << type_name<T>() << ">: " << dt << " seconds" << endl;
    }
}


int main(int argc, char **argv)
{
    time_pointing_plan<float>();
    // time_preplan<double>(); // FIXME blue
    return 0;
}
       
