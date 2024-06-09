#include "../include/gpu_mm2.hpp"

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T>
ToyPointing<T>::ToyPointing(long nsamp_, long nypix_, long nxpix_, double scan_speed_, double drift_radians_) :
    nsamp(nsamp_), nypix(nypix_), nxpix(nxpix_), scan_speed(scan_speed_), drift_radians(drift_radians_),
    drift_speed(drift_radians / max(nsamp,1L))
{
    check_nsamp(nsamp, "ToyPointing constructor");
    check_nypix_nxpix(nypix, nxpix, "ToyPointing constructor");
    
    assert((scan_speed > 0.0) && (scan_speed <= 1.0));
    assert((drift_speed > 0.0) && (drift_speed <= 1.0));

    this->xpointing_cpu = Array<T> ({3,nsamp}, af_rhost);
    T *yp = xpointing_cpu.data;
    T *xp = xpointing_cpu.data + nsamp;
    T *ap = xpointing_cpu.data + 2*nsamp;

    double y = 1.0;
    double x = 1.0;
    double scan_vel = scan_speed;
	
    double y0 = 1.1 * scan_speed;
    double y1 = nypix - (2.1 * scan_speed);
    
    for (long s = 0; s < nsamp; s++) {
	yp[s] = y;
	xp[s] = x;
	ap[s] = 0.0;   // FIXME do something more interesting here

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

    this->xpointing_gpu = xpointing_cpu.to_gpu();
}


#define INSTANTIATE(T) \
    template ToyPointing<T>::ToyPointing(long nsamp, long nypix, long nxpix, double scan_speed, double drift_radians)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
