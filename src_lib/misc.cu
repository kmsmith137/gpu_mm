//nvcc -o libgpu_point.so gpu_point.cu -shared -lcublas -Xcompiler -fPIC -lgomp

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void clip_kernel(float * arr, long n, float vmin, float vmax)
{
    long off  = blockIdx.x*blockDim.x+threadIdx.x;
    long step = blockDim.x*gridDim.x;
    for(long i = off; i < n; i+= step)
        arr[i] = fmaxf(vmin, fminf(vmax, arr[i]));
}

extern "C" void clip(float * arr, long n, float vmin, float vmax) {
    clip_kernel<<<128,128>>>(arr, n, vmin, vmax);
}
