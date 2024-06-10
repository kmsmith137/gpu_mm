#ifndef _GPU_MM2_HPP
#define _GPU_MM2_HPP

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


struct PointingPrePlan
{
    long nsamp = 0;
    long nypix = 0;
    long nxpix = 0;

    long plan_nbytes = 0;
    long plan_constructor_tmp_nbytes = 0;
    // Forthcoming: plan_map2tod_tmp_nbytes

    // Used internally
    long rk = 0;            // number of TOD samples per threadblock is (2^rk)
    long nblocks = 0;       // number of threadblocks 
    long plan_nmt = 0;      // total number of mt-pairs in plan
    size_t cub_nbytes = 0;  // number of bytes used in cub radix sort 'd_temp_storage'

    // Cumulative count of mt-pairs per threadblock.
    // 1-d array of length nblocks, in GPU memory.
    gputils::Array<uint> nmt_cumsum;
    
    template<typename T>
    PointingPrePlan(const gputils::Array<T> &xpointing_gpu, long nypix, long nxpix);

    void show(std::ostream &os = std::cout) const;
    // Reminder: the plan will need to allocate
    //   plan_mt[]
    //   plan_tt[]
    //   tmp buf for map2tod
    //   sort input buffer
    //   sort d_temp_storage
};


struct PointingPlan
{
    long nsamp = 0;
    long nypix = 0;
    long nxpix = 0;
    
    const PointingPrePlan pp;
    
    gputils::Array<ulong> plan_mt;
    // Forthcoming: plan_tt

    template<typename T>
    PointingPlan(const PointingPrePlan &pp,
		 const gputils::Array<T> &xpointing_gpu,
		 const gputils::Array<unsigned char> &buf,
		 const gputils::Array<unsigned char> &tmp_buf);
};


template<typename T>
struct ToyPointing
{
    // Scans currently go at 45 degrees, and cover the full y-range.
    
    ToyPointing(long nsamp, long nypix, long nxpix,
		double scan_speed,     // map pixels per sample
		double total_drift);   // total drift over full TOD, in x-pixels

    const long nsamp;
    const long nypix;
    const long nxpix;
    const double scan_speed;
    const double total_drift;
    const double drift_speed;

    // Since ToyPointing is only used in unit tests, assume the caller
    // wants array copies on both CPU and GPU.
    
    gputils::Array<T> xpointing_cpu;
    gputils::Array<T> xpointing_gpu;
};


template<typename T>
extern void launch_simple_tod2map(gputils::Array<T> &map, const gputils::Array<T> &tod, const gputils::Array<T> &xpointing);


// Argument checking (defined in check_arguments.cu)

extern void check_nsamp(long nsamp, const char *where);
extern void check_nypix(long nypix, const char *where);
extern void check_nxpix(long nxpix, const char *where);
extern void check_err(int err, const char *where);

template<typename T> extern void check_map(const gputils::Array<T> &map, long &nypix, long &nxpix, const char *where);
template<typename T> extern void check_tod(const gputils::Array<T> &tod, long &nsamp, const char *where);
template<typename T> extern void check_xpointing(const gputils::Array<T> &xpointing, long &nsamp, const char *where);

extern void check_buffer(const gputils::Array<unsigned char> &buf, long min_nbytes, const char *where, const char *bufname);


// QuantizedPointing: A utility class used in unit tests.

struct QuantizedPointing
{
    long nsamp = 0;
    long nypix = 0;
    long nxpix = 0;
    
    // Since QuantizedPointing is only used in unit tests, assume the caller
    // wants the 'iypix' and 'ixpix' arrays on the CPU.

    gputils::Array<int> iypix_cpu;
    gputils::Array<int> ixpix_cpu;

    template<typename T>
    QuantizedPointing(const gputils::Array<T> &xpointing_gpu, long nypix, long nxpix);
};


} // namespace gpu_mm2

#endif //  _GPU_MM2_HPP
