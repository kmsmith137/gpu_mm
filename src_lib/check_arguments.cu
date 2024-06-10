#include "../include/gpu_mm2.hpp"

using namespace std;
using namespace gputils;

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// FIXME all functions in this file are placeholders for future expansion.

void check_nsamp(long nsamp, const char *where)
{
    assert(nsamp > 0);
    assert(nsamp <= (1L << 31));
    assert((nsamp % 32) == 0);
}


void check_nypix(long nypix, const char *where)
{
    assert(nypix > 0);
    assert(nypix <= 64*1024);
    assert((nypix % 64) == 0);
}


void check_nxpix(long nxpix, const char *where)
{
    assert(nxpix > 0);
    assert(nxpix <= 64*1024);
    assert((nxpix % 64) == 0);
}


void check_err(uint err, const char *where)
{
    if (err & 0x1)
	throw runtime_error(string(where) + ": ypix out of range");
    if (err & 0x2)
	throw runtime_error(string(where) + ": xpix out of range");
    if (err & 0x4)
	throw runtime_error(string(where) + ": inconsisent value of nmt between preplan/plan?! (should never happen)");
}


template<typename T>
void check_map(const Array<T> &map, long &nypix, long &nxpix, const char *where)
{
    assert(map.ndim == 3);
    assert(map.shape[0] == 3);
    assert(map.is_fully_contiguous());
    assert(map.on_gpu());
    
    check_nypix(map.shape[1], where);
    check_nxpix(map.shape[2], where);
    
    nypix = map.shape[1];
    nxpix = map.shape[2];
}


template<typename T>
void check_tod(const Array<T> &tod, long &nsamp, const char *where)
{
    assert(tod.ndim == 1);
    assert(tod.is_fully_contiguous());
    assert(tod.on_gpu());
    
    check_nsamp(tod.shape[0], where);
    nsamp = tod.shape[0];
}


template<typename T>
void check_xpointing(const Array<T> &xpointing, long &nsamp, const char *where)
{
    assert(xpointing.ndim == 2);
    assert(xpointing.shape[0] == 3);
    assert(xpointing.is_fully_contiguous());
    assert(xpointing.on_gpu());
    
    check_nsamp(xpointing.shape[1], where);
    nsamp = xpointing.shape[1];
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
    template void check_map(const Array<T> &map, long &nypix, long &nxpix, const char *where); \
    template void check_tod(const Array<T> &tod, long &nsamp, const char *where); \
    template void check_xpointing(const Array<T> &xpointing, long &nsamp, const char *where)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm2
