#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__
void extract_ranges_gpu(float * tod, int nsamp, float * rdata, int * ostart, int nrange, int * det, int * istart, int * len) {
	for(int r = blockIdx.x; r < nrange; r += gridDim.x)
		for(int i = threadIdx.x; i < len[r]; i += blockDim.x)
			rdata[ostart[r]+i] = tod[det[r]*nsamp+istart[r]+i];
}

__global__
void insert_ranges_gpu(float * tod, int nsamp, float * rdata, int * ostart, int nrange, int * det, int * istart, int * len) {
	for(int r = blockIdx.x; r < nrange; r += gridDim.x)
		for(int i = threadIdx.x; i < len[r]; i += blockDim.x)
			tod[det[r]*nsamp+istart[r]+i] = rdata[ostart[r]+i];
}

__global__
void clear_ranges_gpu(float * tod, int nsamp, int nrange, int * det, int * istart, int * len) {
	for(int r = blockIdx.x; r < nrange; r += gridDim.x)
		for(int i = threadIdx.x; i < len[r]; i += blockDim.x)
			tod[det[r]*nsamp+istart[r]+i] = 0;
}

extern "C" {

	void extract_ranges(float * tod, int nsamp, float * rdata, int * ostart, int nrange, int * det, int * istart, int * len) {
		extract_ranges_gpu<<<128,128>>>(tod, nsamp, rdata, ostart, nrange, det, istart, len);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	void insert_ranges(float * tod, int nsamp, float * rdata, int * ostart, int nrange, int * det, int * istart, int * len) {
		insert_ranges_gpu<<<128,128>>>(tod, nsamp, rdata, ostart, nrange, det, istart, len);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

	void clear_ranges(float * tod, int nsamp, int nrange, int * det, int * istart, int * len) {
		clear_ranges_gpu<<<128,128>>>(tod, nsamp, nrange, det, istart, len);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
	}

}
