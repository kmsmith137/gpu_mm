#include "../include/gpu_mm.hpp"
#include <gputils/cuda_utils.hpp>  // CUDA_CALL

extern "C"
{
#if 0
}  // pacify emacs c-mode
#endif


void py_reference_map2tod(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    gpu_mm::reference_map2tod(tod, map, xpointing, ndet, nt, ndec, nra);
}

void py_gpu_map2tod(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra)
{
    // FIXME here and elsewhere in this file, get rid of cudaDeviceSynchronize()
    CUDA_CALL(cudaDeviceSynchronize());
    gpu_mm::launch_map2tod(tod, map, xpointing, ndet, nt, ndec, nra);
    CUDA_CALL(cudaDeviceSynchronize());
}


}  // extern "C"
