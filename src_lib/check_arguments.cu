#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"   // errflags

#include <iostream>
#include <gputils/cuda_utils.hpp>  // CUDA_CALL()

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// FIXME error-checking is pretty complete now, but the quality of the error messages
// could use some improvement.


void check_nsamp(long nsamp, const char *where)
{
    xassert(nsamp > 0);
    xassert(nsamp <= (1L << 31));
    xassert((nsamp % 32) == 0);
}


void check_nypix(long nypix, const char *where)
{
    xassert(nypix > 0);
    xassert(nypix <= 64*1024);
    xassert((nypix % 64) == 0);
}


void check_nxpix(long nxpix, const char *where)
{
    xassert(nxpix > 0);
    xassert(nxpix <= 64*1024);
    xassert((nxpix % 64) == 0);
}


void check_err(uint err, const char *where)
{
    if (err & errflag_pixel_outlier)
	throw runtime_error(string(where) + ": pixel coordinates in xpointing array are outside bounding box");
    if (err & errflag_inconsistent_nmt)
	throw runtime_error(string(where) + ": inconsistent value of nmt between preplan/plan?! (should never happen)");
}

void check_gpu_errflags(const uint *errflags_gpu, int nelts, const char *where, uint errflags_to_ignore)
{
    xassert(errflags_gpu != nullptr);
    xassert(nelts > 0);
    
    Array<uint> errflags_host({nelts}, af_rhost | af_zero);
    CUDA_CALL(cudaMemcpy(errflags_host.data, errflags_gpu, nelts * sizeof(uint), cudaMemcpyDefault));

    uint err = 0;
    for (int i = 0; i < nelts; i++)
	err |= errflags_host.data[i];

    check_err(err & ~errflags_to_ignore, where);
}

static void _check_location(int aflags, const char *where, const char *arr_name, bool on_gpu)
{
    if (on_gpu && gputils::af_on_gpu(aflags))
	return;
    if (!on_gpu && gputils::af_on_host(aflags))
	return;

    stringstream ss;
    ss << where << ": expected " << arr_name << " to be in " << (on_gpu ? "GPU" : "CPU") << " memory";
    throw runtime_error(ss.str());
}


template<typename T>
void check_map_and_init_npix(const Array<T> &map, long &nypix, long &nxpix, const char *where, bool on_gpu)
{
    xassert(map.ndim == 3);
    xassert(map.shape[0] == 3);
    xassert(map.is_fully_contiguous());

    nypix = map.shape[1];
    nxpix = map.shape[2];
    
    check_nypix(nypix, where);
    check_nxpix(nxpix, where);
    
    _check_location(map.aflags, where, "map", on_gpu);
}


template<typename T>
void check_map(const Array<T> &map, long nypix_expected, long nxpix_expected, const char *where, bool on_gpu)
{
    long nypix_actual, nxpix_actual;
    check_map_and_init_npix(map, nypix_actual, nxpix_actual, where, on_gpu);

    xassert_eq(nypix_expected, nypix_actual);
    xassert_eq(nypix_expected, nypix_actual);
}


template<typename T>
void check_local_map(const Array<T> &map, const LocalPixelization &local_pixelization, const char *where, bool on_gpu)
{
    // For local maps, we currently allow any shape which is contiguous and has the correct total size.
    xassert(map.is_fully_contiguous());
    xassert_eq(map.size, 3 * local_pixelization.npix);
    _check_location(map.aflags, where, "map", on_gpu);
}


template<typename T>
void check_tod_and_init_nsamp(const Array<T> &tod, long &nsamp, const char *where, bool on_gpu)
{
    xassert((tod.ndim == 1) || (tod.ndim == 2));
    xassert(tod.is_fully_contiguous());

    nsamp = (tod.ndim == 1) ? tod.shape[0] : (tod.shape[0] * tod.shape[1]);
    check_nsamp(nsamp, where);

    _check_location(tod.aflags, where, "tod", on_gpu);
}


template<typename T>
void check_tod(const Array<T> &tod, long nsamp_expected, const char *where, bool on_gpu)
{
    long nsamp_actual;
    check_tod_and_init_nsamp(tod, nsamp_actual, where, on_gpu);
    xassert_eq(nsamp_expected, nsamp_actual);
}


template<typename T>
void check_xpointing_and_init_nsamp(const Array<T> &xpointing, long &nsamp, const char *where, bool on_gpu)
{
    xassert((xpointing.ndim == 2) || (xpointing.ndim == 3));
    xassert(xpointing.shape[0] == 3);
    xassert(xpointing.is_fully_contiguous());

    nsamp = (xpointing.ndim == 2) ? xpointing.shape[1] : (xpointing.shape[1] * xpointing.shape[2]);
    check_nsamp(nsamp, where);

    _check_location(xpointing.aflags, where, "xpointing", on_gpu);
}


template<typename T>
void check_xpointing(const Array<T> &xpointing, long nsamp_expected, const char *where, bool on_gpu)
{
    long nsamp_actual;
    check_xpointing_and_init_nsamp(xpointing, nsamp_actual, where, on_gpu);
    xassert_eq(nsamp_expected, nsamp_actual);
}


void check_buffer(const Array<unsigned char> &buf, long min_nbytes, const char *where, const char *bufname)
{
    if ((buf.ndim != 1) || (buf.size < min_nbytes)) {
	stringstream ss;
	ss << where << ": expected '" << bufname << "' to be 1-d array of length >= " << min_nbytes << ", actual shape=" << buf.shape_str();
	throw runtime_error(ss.str());
    }

    if (!buf.on_gpu()) {
	stringstream ss;
	ss << where << ": array '" << bufname << "' must be on GPU";
	throw runtime_error(ss.str());
    }

    if (!buf.is_fully_contiguous()) {
	stringstream ss;
	ss << where << ": array '" << bufname << "' must be contiguous";
	throw runtime_error(ss.str());
    }
}


#define INSTANTIATE(T) \
    template void check_map(const Array<T> &map, long nypix, long nxpix, const char *where, bool on_gpu); \
    template void check_tod(const Array<T> &tod, long nsamp, const char *where, bool on_gpu); \
    template void check_xpointing(const Array<T> &xpointing, long nsamp, const char *where, bool on_gpu); \
    template void check_local_map(const Array<T> &map, const LocalPixelization &lpix, const char *where, bool on_gpu); \
    template void check_map_and_init_npix(const Array<T> &map, long &nypix, long &nxpix, const char *where, bool on_gpu); \
    template void check_tod_and_init_nsamp(const Array<T> &tod, long &nsamp, const char *where, bool on_gpu); \
    template void check_xpointing_and_init_nsamp(const Array<T> &xpointing, long &nsamp, const char *where, bool on_gpu)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
