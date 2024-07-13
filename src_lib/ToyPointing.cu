#include "../include/gpu_mm.hpp"

#include <iostream>
#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T>
inline Array<T> _alloc_xpointing(long ndet, long nt, int aflags)
{
    if (ndet <= 0) {
	check_nsamp(nt, "ToyPointing constructor");
	return Array<T> ({3,nt}, aflags);
    }
    else {
	long nsamp = ndet * nt;
	check_nsamp(nsamp, "ToyPointing constructor");
	return Array<T> ({3,ndet,nt}, aflags);
    }
}

				 
template<typename T>
ToyPointing<T>::ToyPointing(long ndet, long nt, long nypix_, long nxpix_, double scan_speed_, double total_drift_, bool noisy) :
    // Delegate to version of constructor which uses externally allocated arrays.
    ToyPointing(nypix_, nxpix_, scan_speed_, total_drift_,
		_alloc_xpointing<T> (ndet, nt, af_rhost),
		_alloc_xpointing<T> (ndet, nt, af_gpu),
		noisy)
{ }


template<typename T>
ToyPointing<T>::ToyPointing(long nypix_, long nxpix_, double scan_speed_, double total_drift_,
			    const Array<T> &xpointing_cpu_, const Array<T> &xpointing_gpu_, bool noisy) :
    nypix(nypix_), nxpix(nxpix_), scan_speed(scan_speed_), total_drift(total_drift_),
    drift_speed(total_drift / (3*xpointing_cpu_.size)),
    xpointing_cpu(xpointing_cpu_),
    xpointing_gpu(xpointing_gpu_)
{
    long nsamp;
    
    check_xpointing_and_init_nsamp(xpointing_cpu, nsamp, "ToyPointing constructor (xpointing_cpu)", false);  // on_gpu=false
    check_xpointing(xpointing_gpu, nsamp, "ToyPointing constructor (xpointing_gpu)", true);   // on_gpu=false
    check_nypix(nypix, "ToyPointing constructor");
    check_nxpix(nxpix, "ToyPointing constructor");
    
    xassert((scan_speed > 0.0) && (scan_speed <= 1.0));
    xassert((drift_speed > 0.0) && (drift_speed <= 1.0));
    struct timeval tv0 = get_time();

    if (noisy)
	cout << this->str() << ": constructor called" << endl;

    T *yp = xpointing_cpu.data;
    T *xp = xpointing_cpu.data + nsamp;
    T *ap = xpointing_cpu.data + 2*nsamp;

    double y = 1.0;
    double x = 1.0;
    double scan_vel = scan_speed;
    
    double y0 = 1.1 * scan_speed;
    double y1 = nypix-2 - (1.1 * scan_speed);
    
    double xmin = x;
    double xmax = x;
    double ymin = y;
    double ymax = y;
    
    for (long s = 0; s < nsamp; s++) {
	yp[s] = y;
	xp[s] = x;
	ap[s] = 1.0;   // FIXME do something more interesting here

	xmin = min(xmin, x);
	xmax = max(xmax, x);
	ymin = min(ymin, y);
	ymax = max(ymax, y);

	if ((scan_vel > 0) && (y > y1))
	    scan_vel = -scan_speed;
	if ((scan_vel < 0) && (y < y0))
	    scan_vel = scan_speed;

	y += scan_vel;
	x += scan_vel + drift_speed;

	if (x < 0.0)
	    x += nxpix;
	if (x > nxpix)
	    x -= nxpix;
    }

    // Copy CPU -> GPU
    xpointing_gpu.fill(xpointing_cpu);


    if (noisy)
	cout << this->str() << ": constructor done, time =" << time_since(tv0) << " seconds" << endl;
}


template<typename T>
string ToyPointing<T>::str() const
{
    stringstream ss;
    string tod_shape = gputils::shape_str(xpointing_cpu.ndim-1, xpointing_cpu.shape+1);
    
    ss << "ToyPointing(tod_shape=" << tod_shape
       << ", nypix=" << nypix << ", nxpix=" << nxpix
       << ", scan_speed=" << scan_speed << ", total_drift=" << total_drift
       << ")";
    
    return ss.str();
}


#define INSTANTIATE(T) \
    template ToyPointing<T>::ToyPointing(long ndet, long nt, long nypix, long nxpix, double scan_speed, double total_drift, bool noisy); \
    template ToyPointing<T>::ToyPointing(long nypix, long nxpix, double scan_speed, double total_drift, const Array<T> &xp_cpu, const Array<T> &xp_gpu, bool noisy); \
    template string ToyPointing<T>::str() const

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
