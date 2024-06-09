#include "../include/gpu_mm2.hpp"

#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T>
ToyPointing<T>::ToyPointing(long nsamp_, long nypix_, long nxpix_, double scan_speed_, double total_drift_) :
    nsamp(nsamp_), nypix(nypix_), nxpix(nxpix_), scan_speed(scan_speed_), total_drift(total_drift_),
    drift_speed(total_drift / max(nsamp,1L))
{
    check_nsamp(nsamp, "ToyPointing constructor");
    check_nypix(nypix, "ToyPointing constructor");
    check_nxpix(nxpix, "ToyPointing constructor");
    
    assert((scan_speed > 0.0) && (scan_speed <= 1.0));
    assert((drift_speed > 0.0) && (drift_speed <= 1.0));
    
    struct timeval tv0 = get_time();
    long nbytes_x = 3 * nsamp * sizeof(T);
    cout << "ToyPointing<" << type_name<T>() << "> constructor: start (" << nbytes_to_str(nbytes_x) << " xpointing)" << endl;

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
	ap[s] = 1.0;   // FIXME do something more interesting here

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
    cout << "ToyPointing<" << type_name<T>() << "> constructor: done, time = " << time_since(tv0) << " seconds" << endl;
}


#define INSTANTIATE(T) \
    template ToyPointing<T>::ToyPointing(long nsamp, long nypix, long nxpix, double scan_speed, double total_drift)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
