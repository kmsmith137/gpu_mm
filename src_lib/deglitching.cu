#include "../include/gpu_mm.hpp"

#include <cassert>
#include <ksgpu/cuda_utils.hpp>

using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T>
static void check_gpu_array(const ksgpu::Array<T> &arr, const char *kernel_name, const char *arr_name, int expected_ndim)
{
    if (arr.ndim != expected_ndim) {
	std::stringstream ss;
	ss << kernel_name << ": got " << arr_name << ".ndim=" << arr.ndim << ", expected ndim=" << expected_ndim;
	throw std::runtime_error(ss.str());
    }

    if (!arr.is_fully_contiguous()) {
	std::stringstream ss;
	ss << kernel_name << ": expected array '" << arr_name << "' to be fully contiguous";
	throw std::runtime_error(ss.str());
    }

    if (!arr.on_gpu()) {
	std::stringstream ss;
	ss << kernel_name << ": array '" << arr_name << "' was not on the GPU as expected";
	throw std::runtime_error(ss.str());
    }
}


__device__ __forceinline__ float reduce_add(float x)
{
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x01);
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x02);
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x04);
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x08);
    x += __shfl_sync(0xffffffff, x, threadIdx.x ^ 0x10);
    return x;
}


// -------------------------------------------------------------------------------------------------
//
// out.shape = (R,2)
// index_map.shape = (R,5)
// signal.shape = (ndet,nt)
//
// Launch with 32 threads/block, and R threadblocks.
// Note that value of R is implicitly supplied via gridDim.x.
//
// This code is poorly optimized, but according to Sigurd,
// "this code doesn't need to be fast, it just needs to not be slow".


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
    check_gpu_array(out, "get_border_means", "out", 2);
    check_gpu_array(signal, "get_border_means", "signal", 2);
    check_gpu_array(index_map, "get_border_means", "index_map", 2);

    // 'out' should have shape (R,2), and 'index_map' should have shape (R,5).
    xassert_eq(out.shape[1], 2);
    xassert_eq(index_map.shape[1], 5);
    xassert_eq(out.shape[0], index_map.shape[0]);
    
    int R = out.shape[0];
    int ndet = signal.shape[0];
    int ntime = signal.shape[1];
    
    border_means_kernel<<<R,32>>> (out.data, signal.data, index_map.data, ndet, ntime);
    CUDA_PEEK("border_means kernel launch");

    // FIXME errflags
}


// -------------------------------------------------------------------------------------------------
//
// signal = shape (ndet,nt)
// bvals = shape (R,2) array (output from get_border_means)
// cumj = length-R array
// index_map2 = shape (R,4)
// 
//   index_map2[r,0] = detector index
//   index_map2[r,1] = first r-value with same detector index
//   index_map2[r,2] = starting time index
//   index_map2[r,3] = ending time index
//
// Launch with 128 threads/block, and R threadblocks.
// Note that value of R is implicitly supplied via gridDim.x.
//
// This code isn't well-optimized, but according to Sigurd,
// "this code doesn't need to be fast, it just needs to not be slow".


__global__ void deglitch_kernel(float *signal, const float *bvals, const float *cumj, const int *index_map2, int ndet, int nt)
{
    int R = gridDim.x;
    int ri = blockIdx.x;
    int di = index_map2[4*ri];
    int r0 = index_map2[4*ri + 1];
    int t0 = index_map2[4*ri + 2];
    int t1 = index_map2[4*ri + 3];
    int t2 = nt;

    float bval = bvals[2*ri + 1];
    float cval = cumj[ri];
    
    if (r0 > 0)
	cval -= cumj[r0-1];

    if (ri < R-1) {
	int dnext = index_map2[4*ri + 4];
	if (dnext == di)
	    t2 = index_map2[4*ri + 6];
    }

    signal += long(di) * long(nt);

    // signal[t0:t1] = bval - cval
    // signal[t1:t2] -= cval

    for (int t = t0 + threadIdx.x; t < t1; t += blockDim.x)
	signal[t] = bval - cval;

    __syncwarp();
    
    for (int t = t1 + threadIdx.x; t < t2; t += blockDim.x)
	signal[t] -= cval;
}


void deglitch(
    ksgpu::Array<float> &signal,          // shape (ndet,ntime)
    const ksgpu::Array<float> &bvals,     // shape (R,2)
    const ksgpu::Array<float> &cumj,      // shape (R,)
    const ksgpu::Array<int> &index_map2)  // shape (R,4)
{
    check_gpu_array(signal, "deglitch", "signal", 2);
    check_gpu_array(bvals, "deglitch", "bvals", 2);
    check_gpu_array(cumj, "deglitch", "cumj", 1);
    check_gpu_array(index_map2, "deglitch", "index_map2", 2);

    // bvals.shape = (R,2)
    // cumj.shape = (R,)
    // index_map2.shape = (R,4)
    xassert_eq(bvals.shape[1], 2);
    xassert_eq(index_map2.shape[1], 4);
    xassert_eq(bvals.shape[0], cumj.shape[0]);
    xassert_eq(index_map2.shape[0], cumj.shape[0]);
    
    int R = cumj.shape[0];
    int ndet = signal.shape[0];
    int ntime = signal.shape[1];

    deglitch_kernel<<<R,128>>> (signal.data, bvals.data, cumj.data, index_map2.data, ndet, ntime);
    CUDA_PEEK("deglitch kernel launch");

    // FIXME errflags
}


}  // namespace gpu_mm
