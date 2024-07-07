#ifndef _GPU_MM2_HPP
#define _GPU_MM2_HPP

#include <cassert>
#include <iostream>
#include <gputils/Array.hpp>

namespace gpu_mm2 {
#if 0
}   // pacify editor auto-indent
#endif


// --------------------------------------------------
//             (GLOBAL) PIXEL-SPACE MAPS
// --------------------------------------------------
//
// A "global" pixel-space map is an array
//
//    map[3][nypix][nxpix].
//
// The 'y' coordinate is non-periodic, and is usually declination.
// The 'x' coordinate is periodic, and is usually RA.
//
// Currently we require nypix to be a multiple of 64, and nxpix
// to be a multiple of 128. (FIXME: I hope to improve this soon.)
// We also require nxcells and nycells to be <= 1024.
//
// We divide maps into 64-by-64 cells. Thus the number of cells is:
//   nycells = ceil(nypix / 64)
//   nxcells = ceil(nxpix / 64)
//
// --------------------------------------------------
//               LOCAL PIXEL-SPACE MAPS
// --------------------------------------------------
//
// A "local" pixel-space map represents a subset of a global map
// held on one GPU. We currently require the local map to be defined
// by a subset of cells
//
//    map_loc[3][ncells_loc][64][64]
//
// The local -> global cell mapping could be represented as:
//
//    int ncells_loc;
//    int icell_loc[ncells_loc];   // 20-bit global cell index
//
// However, the GPU kernels use the global -> local cell mapping instead,
// which is represented as:
//
//    int ncells_loc;
//    int icell_glo[2^20];
//
// where elements of icell_glo[] are either integers in [0:ncells_loc),
// or (-1) if a given global cell is not in the local map.
//
// --------------------------------------------------
//                   TIMESTREAMS
// --------------------------------------------------
//
// As far as map <-> tod kernels are concerned, timestreams are
// just 1-d arrays
//
//    tod[nsamp];   // indexed by "sample"
//
// In a larger map-maker, the sample index may be a "flattened"
// index representing multiple sub-indices, e.g. tod[ndetectors][ntod].
//
// In our current implementation, 'nsamp' must be <= 2^31.
// (FIXME what am I assuming about divisibility?)
//
// --------------------------------------------------
//                   XPOINTING
// --------------------------------------------------
//
// We use "exploded pointing" throughout, represented as an array
//
//   xpointing[3][nsamp];   // axis 0 is {y,x,alpha}
//
//

// -----------------------------------------------------------------------------
//
// PointingPrePlan and PointingPlan.
//
// We factor plan creation into two steps:
//
//   - Create PointingPrePlan from xpointing array
//   - Create PointingPlan from PointingPrePlan + xpointing array.
//
// These steps have similar running times, but the PointingPrePlan is much
// smaller in memory (a few KB versus ~100 MB). Therefore, PointingPrePlans
// can be retained (per-TOD) for the duration of the program, whereas
// PointingPlans will typically be created and destroyed on the fly.


struct PointingPrePlan
{
    template<typename T>
    PointingPrePlan(const gputils::Array<T> &xpointing_gpu, long nypix, long nxpix);

    long nsamp = 0;
    long nypix = 0;
    long nxpix = 0;

    long plan_nbytes = 0;                  // length of 'buf' argument to PointingPlan constructor
    long plan_constructor_tmp_nbytes = 0;  // length of 'tmp_buf' argument to PointingPlan constructor
    // Forthcoming: plan_map2tod_tmp_nbytes

    // Used internally.
    long rk = 0;            // number of TOD samples per threadblock is (2^rk)
    long nblocks = 0;       // number of threadblocks 
    long plan_nmt = 0;      // total number of mt-pairs in plan
    size_t cub_nbytes = 0;  // number of bytes used in cub radix sort 'd_temp_storage'

    // Cumulative count of mt-pairs per threadblock.
    // 1-d array of length nblocks, in GPU memory.
    // FIXME should be able to swap between host/GPU memory.
    gputils::Array<uint> nmt_cumsum;

    std::string str() const;

    // Reminder: the plan will need to allocate
    //   plan_mt[]
    //   plan_tt[]
    //   tmp buf for map2tod
    //   sort input buffer
    //   sort d_temp_storage
};


struct PointingPlan
{
    const long nsamp;
    const long nypix;
    const long nxpix;
    
    const PointingPrePlan pp;

    gputils::Array<unsigned char> buf;
    ulong *plan_mt = nullptr;
    uint *err_gpu = nullptr;

    gputils::Array<uint> err_cpu;

    // This constructor uses externally allocated GPU memory.
    template<typename T>
    PointingPlan(const PointingPrePlan &pp,
		 const gputils::Array<T> &xpointing_gpu,
		 const gputils::Array<unsigned char> &buf,
		 const gputils::Array<unsigned char> &tmp_buf);

    // This constructor allocates GPU memory.
    template<typename T>
    PointingPlan(const PointingPrePlan &pp,
		 const gputils::Array<T> &xpointing_gpu);

    // Used in unit tests.
    gputils::Array<ulong> get_plan_mt(bool gpu) const;
};


// -----------------------------------------------------------------------------
//
// Testing


template<typename T>
struct ToyPointing
{
    // Scans currently go at 45 degrees, and cover the full y-range.

    // Version of constructor which allocates xpointing arrays.
    ToyPointing(long nsamp, long nypix, long nxpix,
		double scan_speed,     // map pixels per TOD sample
		double total_drift,    // total drift over full TOD, in x-pixels
		bool noisy = true);

    // Version of constructor with externally allocated xpointing arrays (intended for python)
    ToyPointing(long nsamp, long nypix, long nxpix,
		double scan_speed, double total_drift,
		const gputils::Array<T> &xpointing_cpu,
		const gputils::Array<T> &xpointing_gpu,
		bool noisy = true);

    long nsamp;
    long nypix;
    long nxpix;
    double scan_speed;    // map pixels per TOD sample
    double total_drift;   // total drift over full TOD, in x-pixels
    double drift_speed;   // drift (in x-pixels) per TOD sample

    // Since ToyPointing is only used in unit tests, assume the caller
    // wants array copies on both CPU and GPU.
    
    gputils::Array<T> xpointing_cpu;
    gputils::Array<T> xpointing_gpu;

    std::string str() const;
};


template<typename T>
extern void launch_simple_tod2map(gputils::Array<T> &map, const gputils::Array<T> &tod, const gputils::Array<T> &xpointing);


// Argument checking (defined in check_arguments.cu)

extern void check_nsamp(long nsamp, const char *where);
extern void check_nypix(long nypix, const char *where);
extern void check_nxpix(long nxpix, const char *where);
extern void check_err(uint err, const char *where);

// The 'nsamp' argument to check_xpointing() has the following semantics:
//   - if nsamp > 0, then exception will be thrown if xpointing array has mismatched size
//   - if nsamp == 0, then check_xpointing() will set the value of nsamp.

template<typename T> extern void check_map(const gputils::Array<T> &map, long &nypix, long &nxpix, const char *where);
template<typename T> extern void check_tod(const gputils::Array<T> &tod, long &nsamp, const char *where);
template<typename T> extern void check_xpointing(const gputils::Array<T> &xpointing, long &nsamp, const char *where, bool on_gpu=true);

extern void check_buffer(const gputils::Array<unsigned char> &buf, long min_nbytes, const char *where, const char *bufname);


// QuantizedPointing: A utility class used in unit tests.
//
// Given an 'xpointing' array on the GPU, determine which map pixel (iy, ix)
// each time sample falls into, and store the result in two length-nsamp arrays
// (iypix_cpu, ixpix_cpu). (Since this class is only used in unit tests, we assume
// that the caller wants these arrays on the CPU.)

struct QuantizedPointing
{
    template<typename T>
    QuantizedPointing(const gputils::Array<T> &xpointing_gpu, long nypix, long nxpix);
    
    long nsamp = 0;
    long nypix = 0;
    long nxpix = 0;

    gputils::Array<int> iypix_cpu;   // length nsamp
    gputils::Array<int> ixpix_cpu;   // length nsamp

    // For testing PointingPrePlan.
    // The 'rk' argument has the same meaning as PointingPrePlan::rk.
    // The returned array has the same meaning as PointingPrePlan::nmt_cumsum.
    gputils::Array<uint> compute_nmt_cumsum(int rk) const;
    
    std::string str() const;
};


} // namespace gpu_mm2

#endif //  _GPU_MM2_HPP
