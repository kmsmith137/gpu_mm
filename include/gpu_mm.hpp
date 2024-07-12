#ifndef _GPU_MM_HPP
#define _GPU_MM_HPP

#include <cassert>
#include <gputils/Array.hpp>

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// map2tod
//
// Warning!! Caller must ensure that
//   - All elements of xpointing[0,:,:] are in [0.0, nra-1)
//   - All elements of xpointing[1,:,:] are in [0.0, ndec-1)
//
// If this condition is not satisfied, then the map2tod kernel will segfault or return nonsense.
// In particular, map2tod doesn't know that the ra coordinate should be periodic.


extern void launch_old_map2tod(
    gputils::Array<float> &tod,              // shape (ndet, nt)
    const gputils::Array<float> &map,        // shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &xpointing   // shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
);


// Slow single-threaded CPU map2tod, for testing

extern void reference_map2tod(
    gputils::Array<float> &tod,              // shape (ndet, nt)
    const gputils::Array<float> &map,        // shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &xpointing   // shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
);


// -------------------------------------------------------------------------------------------------
//
// tod2map
//
// Note: tod2map() accumulates its output! (rather than overwriting existing output array)
//
// The GPU version of tod2map() works like this:
//
//   - The map is logically divided into 64-by-64 cells.
//
//   - The tod is viewed as a "flattened" 1-d array of length (ndet * nt), and logically
//     divided into length-32 cache lines. (We use the term "sample" to refer to an index
//     in this 1-d array, i.e. a sample is a (detector,time) pair.)
//
//   - The GPU kernel needs a "plan", or data structure which lists the map cells which
//     overlap with the tod, and allows easy lookup of the tod cache lines which overlap
//     with a given map cell.
//
// The plan consists of two arrays:
//
//   - plan_cltod_list[]: a 1-d array of tod cache line indices. Here, a "cache line index"
//     is an integer 'cltod' which represents samples [32*cltod:32*(cltod+1)].
//
//     The plan_cltod_list[] array should be organized so that all tod cache line indices which
//     overlap with the first nonempty map cell appears first, followed by tod cache lines
//     which overlap with the second cell, etc. If a tod cache line overlaps with multiple
//     map cells, then it will appear in the plan_cltod_list[] array multiple times.
//
//   - plan_quadruples[]: a 1-d array of "quadruples", corresponding to nonempty map cells.
//     Each quadruple consists of four integers:
//
//         int cell_idec = plan_quadruples[4*q];
//         int cell_ira = plan_quadruples[4*q+1];
//         int icl_start = plan_quadruples[4*q+2];
//         int icl_end = plan_quadruples[4*q+3];
//
//     The values of (cell_idec, cell_ira) must be multiples of 64, and specify a nonempty
//     map cell. The values of (icl_start, icl_end) specify an index range in the
//     plan_cltod_list[] array. This index range should correspond to the cache linex
//     which overlap with the given map cell.


extern void launch_old_tod2map(
    gputils::Array<float> &map,                  // Shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &tod,            // Shape (ndet, nt)
    const gputils::Array<float> &xpointing,      // Shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
    const gputils::Array<int> &plan_cltod_list,  // Shape (plan_ncltod,)
    const gputils::Array<int> &plan_quadruples   // Shape (plan_nquadruples, 4)
);


// Slow single-threaded CPU tod2map, for testing.

extern void reference_tod2map(
    gputils::Array<float> &map,              // Shape (3, ndec, nra)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &tod,        // Shape (ndet, nt)
    const gputils::Array<float> &xpointing   // Shape (3, ndet, nt)    where axis 0 = {px_dec, px_ra, alpha}
);


// -------------------------------------------------------------------------------------------------


// Temporary hack: construct pointing plans on CPU.
struct OldPointingPlan
{
    OldPointingPlan(const gputils::Array<float> &xpointing, int ndec, int nra, bool verbose=true);

    long ncl_uninflated = 0;
    long ncl_inflated = 0;     // same as 'plan_ncltod' argument to tod2map()
    long num_quadruples = 0;   // same as 'plan_nquadruples' argument to tod2map()
    
    gputils::Array<int> plan_cltod_list;    // 1-d contiguous array of shape (ncl_inflated,)
    gputils::Array<int> plan_quadruples;    // 2-d contiguous array of shape (num_quadruples,4)
};


} // namespace gpu_mm

#endif //  _GPU_MM_HPP
