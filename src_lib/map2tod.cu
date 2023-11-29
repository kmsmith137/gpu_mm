#include "../include/gpu_mm.hpp"
#include <cassert>

using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif



static void _check_map2tod_args(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    assert(tod != nullptr);
    assert(map != nullptr);
    assert(xpointing != nullptr);
    
    assert(ndet > 0);
    assert(nt > 0);
    assert(ndec > 0);
    assert(nra > 0);

    assert((nt % 32) == 0);
    assert((ndec % 64) == 0);
    assert((nra % 64) == 0);
}


static void _check_map2tod_args(Array<float> &tod, const Array<float> &map, const Array<float> &xpointing)
{
    assert(tod.on_gpu());
    assert(tod.ndim == 2);
    assert(tod.is_fully_contiguous());
    
    assert(map.on_gpu());
    assert(map.ndim == 3);
    assert(map.shape[0] == 3);
    assert(map.is_fully_contiguous());
    
    assert(xpointing.on_gpu());
    assert(xpointing.ndim == 3);
    assert(xpointing.shape[0] == 3);
    assert(xpointing.shape[1] == tod.shape[0]);
    assert(xpointing.shape[2] == tod.shape[1]);
    assert(xpointing.is_fully_contiguous());
}


void reference_map2tod(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    _check_map2tod_args(tod, map, xpointing, ndet, nt, ndec, nra);
    
    long ns = long(ndet) * long(ndec);

    for (long s = 0; s < ns; s++) {
	float px_dec = xpointing[s];
	float px_ra = xpointing[s + ns];
	float alpha = xpointing[s + 2*ns];
	float cos_2a = cosf(2*alpha);
	float sin_2a = sinf(2*alpha);

	int idec = int(px_dec);
	int ira = int(px_ra);
	float ddec = px_dec - float(idec);
	float dra = px_ra - float(ira);
	
	assert(idec > 0);
	assert(idec < ndec-1);
	assert(ira > 0);
	assert(ira < nra-1);
	
	long ipix = idec*nra + ira;
	float out = 0.0;
	
	float w = (1.0-ddec) * (1.0-dra);
	out += w * map[ipix];
	out += w * cos_2a * map[ipix+ns];
	out += w * sin_2a * map[ipix+2*ns];

	w = (1.0-ddec) * (dra);
	out += w * map[ipix + 1];
	out += w * cos_2a * map[ipix+ns + 1];
	out += w * sin_2a * map[ipix+2*ns + 1];
	
	w = ddec * (1.0 - dra);
	out += w * map[ipix + nra];
	out += w * cos_2a * map[ipix+ns + nra];
	out += w * sin_2a * map[ipix+2*ns + nra];
	
	w = ddec * dra;
	out += w * map[ipix + nra+1];
	out += w * cos_2a * map[ipix+ns + nra+1];
	out += w * sin_2a * map[ipix+2*ns + nra+1];

	tod[s] = out;
    }
}


void reference_map2tod(Array<float> &tod, const Array<float> &map, const Array<float> &xpointing)
{
    _check_map2tod_args(tod, map, xpointing);
    
    reference_map2tod(tod.data, map.data, xpointing.data,
		      tod.shape[0], tod.shape[1], map.shape[1], map.shape[2]);
}


// -------------------------------------------------------------------------------------------------


// Number of blocks should be: 
//   ceil(ndet * nt / nt_per_block)

__global__ void map2tod_kernel(float *tod, const float *map, const float *xpointing,
			       int ndet, int nt, int ndec, int nra, int nt_per_block)
{
    long s0 = long(blockIdx.x) * long(nt_per_block);
    long ns = long(ndet) * long(nt);
    
    tod += s0;
    map += s0;
    xpointing += s0;

    int n = min(ns - s0, long(nt_per_block));
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
	float px_dec = xpointing[i];
	float px_ra = xpointing[i + ns];
	float alpha = xpointing[i + 2*ns];
	float cos_2a = cosf(2*alpha);
	float sin_2a = sinf(2*alpha);

	int idec = int(px_dec);
	int ira = int(px_ra);
	int ipix = idec*nra + ira;
	float ddec = px_dec - float(idec);
	float dra = px_ra - float(ira);
	float out = 0.0;

	float w = (1.0-ddec) * (1.0-dra);
	out += w * map[ipix];
	out += w * cos_2a * map[ipix+ns];
	out += w * sin_2a * map[ipix+2*ns];

	w = (1.0-ddec) * (dra);
	out += w * map[ipix + 1];
	out += w * cos_2a * map[ipix+ns + 1];
	out += w * sin_2a * map[ipix+2*ns + 1];
	
	w = ddec * (1.0 - dra);
	out += w * map[ipix + nra];
	out += w * cos_2a * map[ipix+ns + nra];
	out += w * sin_2a * map[ipix+2*ns + nra];
	
	w = ddec * dra;
	out += w * map[ipix + nra+1];
	out += w * cos_2a * map[ipix+ns + nra+1];
	out += w * sin_2a * map[ipix+2*ns + nra+1];

	tod[i] = out;
    }
}


void launch_map2tod(float *tod, const float *map, const float *xpointing,
		    int ndet, int nt, int ndec, int nra, cudaStream_t stream,
		    int nthreads_per_block, int nt_per_block)
{
    _check_map2tod_args(tod, map, xpointing, ndet, nt, ndec, nra);
    
    assert(nthreads_per_block > 0);
    assert((nthreads_per_block % 32) == 0);
    assert(nthreads_per_block <= 1024);
    assert(nt_per_block > 0);

    int m = nt_per_block;
    long nblocks = (long(ndet) * long(nt) + m - 1) / m;
    assert(nblocks < (1L << 31));

    map2tod_kernel<<< nblocks, nthreads_per_block, 0, stream >>>
	(tod, map, xpointing, ndet, nt, ndec, nra, nt_per_block);
}


void launch_map2tod(Array<float> &tod, Array<float> &map, Array<float> &xpointing,
		    cudaStream_t stream, int nthreads_per_block, int nt_per_block)
{
    _check_map2tod_args(tod, map, xpointing);
    
    launch_map2tod(tod.data, map.data, xpointing.data,
		   tod.shape[0], tod.shape[1], map.shape[1], map.shape[2],   // (ndet, nt, ndec, nra)
		   stream, nthreads_per_block, nt_per_block);
}


}  // namespace gpu_mm
