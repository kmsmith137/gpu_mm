#include "../include/gpu_mm.hpp"

#include <cassert>
#include <ksgpu/cuda_utils.hpp>

using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// out.shape = (R,2)
// index_map.shape = (R,5)
// signal.shape = (ndet,nt)
//
// Launch with 32 threads/block, and R threadblocks.
// Note that value of R is implicitly supplied via gridDim.x
//
// This code is poorly optimized, but according to Sigurd,
// "this code doesn't need to be fast, it just needs to not be slow".

__device__ __forceinline__ float reduce_add(float x)
{
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x01);
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x02);
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x04);
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x08);
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x10);
    return x;
}
    

__global__ void border_means_kernel(float *out, const float *signal, const int *index_map, int ndet, int nt)
{
    int ri = blockIdx.x;
    int di = index_map[5*ri];
    int b00 = index_map[5*ri + 1];
    int b01 = index_map[5*ri + 2];
    int b10 = index_map[5*ri + 3];
    int b11 = index_map[5*ri + 4];

    signal += long(di) * long(nt);

    float sum0 = 0.0f;    
    for (int t = b00 + threadIdx.x; t < b01; t += blockDim.x)
	sum0 += signal[t];
    
    float sum1 = 0.0f;
    for (int t = b10 + threadIdx.x; t < b11; t += blockDim.x)
	sum1 += signal[t];

    sum0 = reduce_add(sum0);
    sum1 = reduce_add(sum1);

    if (b00 < b01)
	sum0 /= (b01-b00);
    if (b10 < b11)
	sum1 /= (b11-b10);
    if (b00 >= b01)
	sum0 = sum1;
    if (b10 >= b11)
	sum1 = sum0;

    if (threadIdx.x == 0) {
	out[2*ri] = sum0;
	out[2*ri + 1] = sum1;
    }
}


void get_border_means(
    ksgpu::Array<float> &out,            // shape (R,2) 
    const ksgpu::Array<float> &signal,   // shape (ndet,ntime)
    const ksgpu::Array<int> &index_map)  // shape (R,5)
{
    xassert((out.ndim == 2) && (out.shape[1] == 2));
    xassert(out.is_fully_contiguous());
    xassert(out.on_gpu());

    xassert((index_map.ndim == 2) && (index_map.shape[1] == 5));
    xassert(index_map.is_fully_contiguous());
    xassert(index_map.on_gpu());
    
    xassert(signal.ndim == 2);
    xassert(signal.is_fully_contiguous());
    xassert(signal.on_gpu());

    // 'out' should have shape (R,2), and 'index_map' should have shape (R,5).
    xassert_eq(out.shape[0], index_map.shape[0]);
    
    int R = out.shape[0];
    int ndet = signal.shape[0];
    int ntime = signal.shape[1];
    
    border_means_kernel<<<R,32>>> (out.data, signal.data, index_map.data, ndet, ntime);
    CUDA_PEEK("border_means kernel launch");

    // FIXME errflags
}


}  // namespace gpu_mm
