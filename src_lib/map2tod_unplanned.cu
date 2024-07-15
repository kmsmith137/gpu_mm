#include "../include/gpu_mm.hpp"
#include "../include/gpu_mm_internals.hpp"

#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// This helper class isn't really necessary, but makes unplanned_map2tod_kernel() a bit more readable.
template<typename T>
struct map_evaluator
{
    const T *lmap;
    const long *cell_offsets;
    const int nycells;
    const int nxcells;
    const long ystride;
    const long polstride;

    __device__ map_evaluator(const T *lmap_, const long *cell_offsets_, int nycells_, int nxcells_, long ystride_, long polstride_)
	: lmap(lmap_), cell_offsets(cell_offsets_), nycells(nycells_), nxcells(nxcells_), ystride(ystride_), polstride(polstride_)
    { }

    __device__ T eval(int iy, int ix, T cos_2a, T sin_2a) const
    {
	int iycell = iy >> 6;
	int ixcell = ix >> 6;
	
	bool valid = (iy >= 0) && (ix >= 0) && (iycell < nycells) && (ixcell < nxcells);
	long offset = valid ? cell_offsets[iycell*nxcells + ixcell] : -1;
	__syncwarp();

	valid = (offset >= 0);
	offset += (ix - (ixcell << 6));
	offset += (iy - (iycell << 6)) * ystride;

	T t = valid ? lmap[offset] : 0;
	T q = valid ? lmap[offset + polstride] : 0;
	T u = valid ? lmap[offset + 2*polstride] : 0;
	__syncwarp();

	return t + cos_2a*q + sin_2a*u;
    }
};


template<typename T>
__global__ void unplanned_map2tod_kernel(
    T *tod,
    const T *lmap,
    const T *xpointing,
    const long *cell_offsets,
    uint nsamp,    // FIXME 32-bit overflow
    uint nsamp_per_block,
    int nycells,
    int nxcells,
    long ystride,
    long polstride)
{    
    map_evaluator<T> mev(lmap, cell_offsets, nycells, nxcells, ystride, polstride);
    
    const uint s0 = blockIdx.x * nsamp_per_block;
    const uint s1 = min(nsamp, s0 + nsamp_per_block);

    for (uint s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
	T ypix = xpointing[s];
	T xpix = xpointing[s + nsamp];
	T alpha = xpointing[s + 2*nsamp];

	T cos_2a, sin_2a;
	dtype<T>::xsincos(2*alpha, &sin_2a, &cos_2a);
	
	// quantize_pixel() is defined in gpu_mm_internals.hpp
	int iy = quantize_pixel(ypix, nycells << 6);   // satisfies -2 <= iy <= nypix
	int ix = quantize_pixel(xpix, nxcells << 6);   // satisfies -2 <= ix <= nxpix

	T dy = ypix - iy;
	T dx = xpix - ix;

	T t = (1-dy) * (1-dx) * mev.eval(iy,   ix,   cos_2a, sin_2a);
	t +=  (1-dy) *   (dx) * mev.eval(iy,   ix+1, cos_2a, sin_2a);
	t +=    (dy) * (1-dx) * mev.eval(iy+1, ix,   cos_2a, sin_2a);
	t +=    (dy) *   (dx) * mev.eval(iy+1, ix+1, cos_2a, sin_2a);

	tod[s] = t;
    }
}


template<typename T>
void launch_unplanned_map2tod(
    Array<T> &tod,              // shape (nsamp,) or (ndet,nt)
    const Array<T> &local_map,  // total size (3 * local_pixelization.npix)
    const Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization)
{
    long nsamp;
    check_tod_and_init_nsamp(tod, nsamp, "unplanned_map2tod", true);            // on_gpu = true
    check_local_map(local_map, local_pixelization, "unplanned_map2tod", true);  // on_gpu = true
    check_xpointing(xpointing, nsamp, "unplanned_map2tod", true);               // on_gpu = true

    int nthreads_per_block = 128;
    int nsamp_per_block = 1024;
    int nblocks = (nsamp + nsamp_per_block - 1) / nsamp_per_block;
    
    unplanned_map2tod_kernel <<< nblocks, nthreads_per_block >>>
	(tod.data,
	 local_map.data,
	 xpointing.data,
	 local_pixelization.cell_offsets_gpu.data,
	 nsamp,
	 nsamp_per_block,
	 local_pixelization.nycells,
	 local_pixelization.nxcells,
	 local_pixelization.ystride,
	 local_pixelization.polstride);

    CUDA_PEEK("unplanned_map2tod kernel launch");
}


#define INSTANTIATE(T) \
    template void launch_unplanned_map2tod(Array<T> &tod, const Array<T> &lmap, const Array<T> &xpointing, const LocalPixelization &lpix)

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
