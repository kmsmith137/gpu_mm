#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"   // errflags

#include <iostream>
#include <ksgpu/cuda_utils.hpp>  // CUDA_CALL()

using namespace std;
using namespace ksgpu;

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


void check_nypix_global(long nypix_global, const char *where)
{
    xassert(nypix_global > 0);
    xassert(nypix_global <= 64*1024);
}


void check_nxpix_global(long nxpix_global, const char *where)
{
    xassert(nxpix_global > 0);
    xassert(nxpix_global <= 64*1024);
}


void check_err(uint err, const char *where, uint errflags_to_ignore)
{
    err &= ~errflags_to_ignore;
    
    // Note: errflag_bad_{xy}pix should come before errflag_not_in_pixelization.
    
    if (err & errflag_bad_ypix)
	throw runtime_error(string(where) + ": xpointing y-value is outside range [1,nypix_global-2].");
    if (err & errflag_bad_xpix)
	throw runtime_error(string(where) + ": xpointing x-value is outside range, perhaps you want the 'periodic_xcoord' flag?");
    if (err & errflag_inconsistent_nmt)
	throw runtime_error(string(where) + ": inconsistent value of nmt between preplan/plan?! (should never happen)");
    if (err & errflag_not_in_pixelization)
	throw runtime_error(string(where) + ": xpointing (y,x) value is not in LocalPixelization (perhaps pixelization is too small, or you want the 'partial_pixelization' flag");
    if (err)
	throw runtime_error(string(where) + ": bad errflags?! (should never happen)");
}

void check_cpu_errflags(const uint *errflags_cpu, int nelts, const char *where, uint errflags_to_ignore)
{
    uint err = 0;
    for (int i = 0; i < nelts; i++)
	err |= errflags_cpu[i];

    check_err(err, where, errflags_to_ignore);
}

void check_gpu_errflags(const uint *errflags_gpu, int nelts, const char *where, uint errflags_to_ignore)
{
    Array<uint> errflags_cpu({nelts}, af_rhost | af_zero);
    CUDA_CALL(cudaMemcpy(errflags_cpu.data, errflags_gpu, nelts * sizeof(uint), cudaMemcpyDefault));
    check_cpu_errflags(errflags_cpu.data, nelts, where, errflags_to_ignore);
}

static void _check_location(int aflags, const char *where, const char *arr_name, bool on_gpu)
{
    if (on_gpu && ksgpu::af_on_gpu(aflags))
	return;
    if (!on_gpu && ksgpu::af_on_host(aflags))
	return;

    stringstream ss;
    ss << where << ": expected " << arr_name << " to be in " << (on_gpu ? "GPU" : "CPU") << " memory";
    throw runtime_error(ss.str());
}


template<typename T>
void check_global_map_and_init_npix(const Array<T> &map, long &nypix_global, long &nxpix_global, const char *where, bool on_gpu)
{
    xassert(map.ndim == 3);
    xassert(map.shape[0] == 3);
    xassert(map.is_fully_contiguous());

    nypix_global = map.shape[1];
    nxpix_global = map.shape[2];
    
    check_nypix_global(nypix_global, where);
    check_nxpix_global(nxpix_global, where);
    
    _check_location(map.aflags, where, "map", on_gpu);
}


template<typename T>
void check_global_map(const Array<T> &map, long nypix_global_expected, long nxpix_global_expected, const char *where, bool on_gpu)
{
    long nypix_global_actual, nxpix_global_actual;
    check_global_map_and_init_npix(map, nypix_global_actual, nxpix_global_actual, where, on_gpu);

    xassert_eq(nypix_global_expected, nypix_global_actual);
    xassert_eq(nypix_global_expected, nypix_global_actual);
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


void check_cell_offsets_and_init_ncells(const Array<long> &cell_offsets, long &nycells, long &nxcells, const char *where, bool on_gpu)
{
    xassert(cell_offsets.ndim == 2);
    xassert(cell_offsets.is_fully_contiguous());

    nycells = cell_offsets.shape[0];
    nxcells = cell_offsets.shape[1];
    
    xassert(nycells > 0);
    xassert(nxcells > 0);
    xassert_le(nycells, 1024);
    xassert_le(nxcells, 1024);

    _check_location(cell_offsets.aflags, where, "cell_offsets", on_gpu);
}


void check_cell_offsets(const Array<long> &cell_offsets, long nycells_expected, long nxcells_expected, const char *where, bool on_gpu)
{
    long nycells_actual, nxcells_actual;
    check_cell_offsets_and_init_ncells(cell_offsets, nycells_actual, nxcells_actual, where, on_gpu);
    
    xassert_eq(nycells_expected, nycells_actual);
    xassert_eq(nxcells_expected, nxcells_actual);
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
    template void check_tod(const Array<T> &tod, long nsamp, const char *where, bool on_gpu); \
    template void check_xpointing(const Array<T> &xpointing, long nsamp, const char *where, bool on_gpu); \
    template void check_local_map(const Array<T> &map, const LocalPixelization &lpix, const char *where, bool on_gpu); \
    template void check_global_map(const Array<T> &map, long nypix_global, long nxpix_global, const char *where, bool on_gpu); \
    template void check_tod_and_init_nsamp(const Array<T> &tod, long &nsamp, const char *where, bool on_gpu); \
    template void check_xpointing_and_init_nsamp(const Array<T> &xpointing, long &nsamp, const char *where, bool on_gpu); \
    template void check_global_map_and_init_npix(const Array<T> &map, long &nypix_global, long &nxpix_global, const char *where, bool on_gpu)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
