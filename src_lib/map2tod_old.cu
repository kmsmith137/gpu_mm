#include "../include/gpu_mm.hpp"
#include <cassert>

#include <iostream>
#include <ksgpu/cuda_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


__global__ void old_map2tod_kernel(float *tod, const float *map, const float *xpointing, long nsamp, int ndec, int nra, int nt_per_block)
{
    // Number of blocks should be: ceil(ns / nt_per_block)
    long s0 = long(blockIdx.x) * long(nt_per_block);

    tod += s0;
    xpointing += s0;
    
    int n = min(nsamp-s0, long(nt_per_block));
    int npix = ndec * nra;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
	float px_dec = xpointing[i];
	float px_ra = xpointing[i + nsamp];
	float alpha = xpointing[i + 2*nsamp];
	float cos_2a = cosf(2.0f * alpha);
	float sin_2a = sinf(2.0f * alpha);

	int idec = int(px_dec);
	int ira = int(px_ra);

	assert(idec >= 0);
	assert(idec < ndec-1);
	assert(ira >= 0);
	assert(ira < nra-1);
	    
	int ipix = idec*nra + ira;
	float ddec = px_dec - float(idec);
	float dra = px_ra - float(ira);
	float out = 0.0f;

	float w = (1.0f - ddec) * (1.0f - dra);
	out += w * map[ipix];
	out += w * cos_2a * map[ipix+npix];
	out += w * sin_2a * map[ipix+2*npix];

	w = (1.0f - ddec) * (dra);
	out += w * map[ipix + 1];
	out += w * cos_2a * map[ipix+npix + 1];
	out += w * sin_2a * map[ipix+2*npix + 1];
	
	w = ddec * (1.0f - dra);
	out += w * map[ipix + nra];
	out += w * cos_2a * map[ipix+npix + nra];
	out += w * sin_2a * map[ipix+2*npix + nra];
	
	w = ddec * dra;
	out += w * map[ipix + nra+1];
	out += w * cos_2a * map[ipix+npix + nra+1];
	out += w * sin_2a * map[ipix+2*npix + nra+1];

	tod[i] = out;
    }
}


void launch_old_map2tod(Array<float> &tod, const Array<float> &map, const Array<float> &xpointing)
{
    static constexpr int nt_per_block = 16384;
    static constexpr int nthreads_per_block = 512;

    long nsamp, ndec, nra;
    check_tod_and_init_nsamp(tod, nsamp, "old_map2tod", true);     // on_gpu=true
    check_xpointing(xpointing, nsamp, "old_map2tod", true);        // on_gpu=true
    check_global_map_and_init_npix(map, ndec, nra, "old_map2tod", true);  // on_gpu=true

    long nblocks = (long(nsamp) + nt_per_block - 1) / nt_per_block;
    xassert(nblocks < (1L << 31));

    old_map2tod_kernel<<< nblocks, nthreads_per_block >>>
	(tod.data, map.data, xpointing.data, nsamp, ndec, nra, nt_per_block);

    CUDA_PEEK("old_map2tod_kernel");
}


}  // namespace gpu_mm
