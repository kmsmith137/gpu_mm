#ifndef _GPU_MM_HPP
#define _GPU_MM_HPP

#include <gputils/Array.hpp>

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// map2tod high-level interface
//
// Warning!! Caller must ensure that
//   - All elements of xpointing[0,:,:] are in { 0, ..., ndec-2}
//   - All elements of xpointing[1,:,:] are in { 0, ..., nra-2}
//
// If this condition is not satisfied, then the map2tod kernel will segfault or return nonsense.

extern void launch_map2tod(
    gputils::Array<float> &tod,              // shape (ndet, nt)
    const gputils::Array<float> &map,        // shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &xpointing,  // shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    cudaStream_t stream = nullptr,
    int nthreads_per_block = 512,
    int nt_per_block = 16384
);


// map2tod low-level interface

extern void launch_map2tod(
    float *tod,              // shape (ndet, nt)
    const float *map,        // shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const float *xpointing,  // shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    int ndet,                // Number of detectors
    int nt,                  // Number of time samples per detector
    int ndec,                // Length of map declination axis
    int nra,                 // Length of map RA axis
    cudaStream_t stream = nullptr,
    int nthreads_per_block = 512,
    int nt_per_block = 16384
);


} // namespace gpu_mm

#endif //  _GPU_MM_HPP
