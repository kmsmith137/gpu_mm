#ifndef _GPU_MM2_HPP
#define _GPU_MM2_HPP

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
//
// We divide maps into 64-by-64 cells. Thus the number of cells is:
//   nycells = ceil(nypix / 64)
//   nxcells = ceil(nxpix / 64)
//
// We sometimes represent cells by a 20-bit cell index:
//   icell = (iycell << 10) | (ixcell)
//
// The code assumes this fits into 10 bits, and also reserves icell=-1
// as a sentinel value. Therefore, nxcells and nycells must be <= 1024,
// and they can't both be equal to 1024.
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


template<typename T>
struct ToyPointing
{
    // Scans currently go at 45 degrees, and cover the full y-range.
    // Usually use 
    
    ToyPointing(long nsamp, long nypix, long nxpix,
		double scan_speed,      // map pixels per sample
		double drift_radians);  // total drift over full TOD

    const long nsamp;
    const long nypix;
    const long nxpix;
    const double scan_speed;
    const double drift_radians;
    const double drift_speed;

    // Since ToyPointing is only used in unit tests, assume the caller
    // wants array copies on both CPU and GPU.
    
    gputils::Array<T> xpointing_cpu;
    gputils::Array<T> xpointing_gpu;
};


// Argument checking (defined in check_arguments.cu)
extern void check_nsamp(long nsamp, const char *where);
extern void check_nypix_nxpix(long nypix, long nxpix, const char *where);


} // namespace gpu_mm2

#endif //  _GPU_MM2_HPP
