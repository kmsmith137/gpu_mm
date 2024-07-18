#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T, int W>
__global__ void unplanned_tod2map_kernel(
    T *lmap,
    const T *tod,
    const T *xpointing,
    const long *cell_offsets,
    uint *errflags,
    long nsamp,
    uint nsamp_per_block,
    int nypix_global,
    int nxpix_global,
    int nycells,
    int nxcells,
    long ystride,
    long polstride,
    bool periodic_xcoord,
    bool partial_pixelization)
{
    // For write_errflags().
    __shared__ uint shmem[W];
    
    // Launch with {32,W} threads.
    const int warpId = threadIdx.y;
    const int laneId = threadIdx.x;

    // 'map_evaluator' and 'pixel_locator' are defined in gpu_mm_internals.hpp.
    map_accumulator<T,true> macc(lmap, cell_offsets, nycells, nxcells, ystride, polstride, partial_pixelization);
    pixel_locator<T> px(nypix_global, nxpix_global, periodic_xcoord);
    
    const long s0 = blockIdx.x * long(nsamp_per_block);
    const long s1 = min(nsamp, s0 + long(nsamp_per_block));
    uint err = 0;  // FIXME currently ignored

    for (long s = s0 + 32*warpId + laneId; s < s1; s += 32*W) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];
	T alpha = xpointing[s + 2*nsamp];
	T t = tod[s];

	T sin_2a, cos_2a;
	dtype<T>::xsincos(2*alpha, &sin_2a, &cos_2a);

	px.locate(ypix, xpix, err);
	macc.accum(px, t, t*cos_2a, t*sin_2a, err);
    }
    
    // No need for __syncthreads() before write_errflags(), since no one else uses shared memory.
    // Warning: write_errflags() assumes thread layout is {32,W,1}, and block layout is {B,1,1}!
    write_errflags(errflags, shmem, err);
}

template<typename T>
void launch_unplanned_tod2map(
    Array<T> &local_map,        // total size (3 * local_pixelization.npix)
    const Array<T> &tod,        // shape (nsamp,) or (ndet,nt)
    const Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization,
    Array<uint> &errflags,      // length nblocks, where nblocks is caller-supplied.
    bool partial_pixelization)
{
    static constexpr int W = 4;

    // FIXME some cut-and-paste with launch_unplanned_map2tod(), define helper function?
    
    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "unplanned_map2tod", true);            // on_gpu = true
    check_local_map(local_map, local_pixelization, "unplanned_map2tod", true);  // on_gpu = true
    check_xpointing(xpointing, nsamp, "unplanned_map2tod", true);               // on_gpu = true

    xassert(errflags.ndim == 1);
    xassert(errflags.is_fully_contiguous());
    xassert(errflags.on_gpu());

    long max_nblocks = errflags.size;
    long nsamp_per_block = (nsamp + max_nblocks - 1) / max_nblocks;
    long nsamp_per_thread = (nsamp_per_block + 32*W - 1) / (32*W);

    nsamp_per_thread = max(nsamp_per_thread, 8L);
    nsamp_per_block = 32*W * nsamp_per_thread;
    
    long nblocks = (nsamp + nsamp_per_block - 1) / nsamp_per_block;
    xassert(nblocks <= max_nblocks);

    unplanned_tod2map_kernel<T,W> <<< nblocks, {32,W} >>>
	(local_map.data,                            // T *lmap
	 tod.data,                                  // const T *tod
	 xpointing.data,                            // const T *xpointing
	 local_pixelization.cell_offsets_gpu.data,  // const long *cell_offsets
	 errflags.data,                             // uint *errflags
	 nsamp,                                     // long nsamp
	 nsamp_per_block,                           // uint nsamp_per_block
	 local_pixelization.nypix_global,           // int nypix_global
	 local_pixelization.nxpix_global,           // int nxpix_global
	 local_pixelization.nycells,                // int nycells
	 local_pixelization.nxcells,                // int nxcells
	 local_pixelization.ystride,                // long ystride
	 local_pixelization.polstride,              // long polstride
	 local_pixelization.periodic_xcoord,        // bool periodic_xcoord
	 partial_pixelization);                     // bool partial_pixelization

    CUDA_PEEK("unplanned_tod2map kernel launch");

    uint errflags_to_ignore = partial_pixelization ? errflag_not_in_pixelization : 0;
    check_gpu_errflags(errflags.data, nblocks, "unplanned_tod2map", errflags_to_ignore);
}


#define INSTANTIATE(T) \
    template void launch_unplanned_tod2map( \
	Array<T> &local_map, \
	const Array<T> &tod, \
	const Array<T> &xpointing, \
	const LocalPixelization &local_pixelization, \
	Array<uint> &errflags, \
	bool partial_pixelization)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
