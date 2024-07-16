#ifndef _GPU_MM_HPP
#define _GPU_MM_HPP

#include <gputils/Array.hpp>

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// --------------------------------------------------
//             (GLOBAL) PIXEL-SPACE MAPS
// --------------------------------------------------
//
// A "global" pixel-space map is an array
//
//    map[3][nypix_global][nxpix_global].
//
// The 'y' coordinate is non-periodic, and is usually declination.
// The 'x' coordinate is periodic, and is usually RA.
//
// Currently we require nypix_global to be a multiple of 64, and nxpix_global
// to be a multiple of 128. (FIXME: I hope to improve this soon.)
// We also require nxcells and nycells to be <= 1024.
//
// We divide maps into 64-by-64 cells. Thus the number of cells is:
//   nycells = ceil(nypix_global / 64)
//   nxcells = ceil(nxpix_global / 64)
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



struct LocalPixelization
{
    // This constructor is intended to be called from C++.
    // The 'cell_offsets' array can be either on the CPU or GPU.
    
    LocalPixelization(long nypix_global, long nxpix_global,
		      const gputils::Array<long> &cell_offsets,
		      long ystride, long polstride,
		      bool periodic_xcoord = true);

    // This constructor is intended to be called from python.
    // The caller is responsible for ensuring that the 'cell_offsets_cpu'
    // and 'cell_offsets_gpu' arrays have the same contents!
    
    LocalPixelization(long nypix_global, long nxpix_global,
		      const gputils::Array<long> &cell_offsets_cpu,
		      const gputils::Array<long> &cell_offsets_gpu,
		      long ystride, long polstride,
		      bool periodic_xcoord = true);

    // Global pixelization
    const long nypix_global;
    const long nxpix_global;
    const bool periodic_xcoord;

    // Local pixelization
    const gputils::Array<long> cell_offsets_cpu;
    const gputils::Array<long> cell_offsets_gpu;
    const long ystride;
    const long polstride;
    
    long nycells;   // same as cell_offsets.shape[0]
    long nxcells;   // same as cell_offsets.shape[1]
    long npix;      // counts only pixels in occupied cells, does not include factor 3 from TQU.
};


struct PointingPrePlan
{
    static constexpr int preplan_size = 1024;

    // This constructor allocates GPU memory, and is intended to be called from C++.
    
    template<typename T>
    PointingPrePlan(const gputils::Array<T> &xpointing_gpu, long nypix_global, long nxpix_global,
		    bool periodic_xcoord = true, bool debug = false);
    
    // This constructor uses externally allocated GPU memory, and is intended to be called from python.
    // The 'nmt_gpu' and 'err_gpu' arrays should have length preplan_size.
    
    template<typename T>
    PointingPrePlan(const gputils::Array<T> &xpointing_gpu, long nypix_global, long nxpix_global,
		    const gputils::Array<uint> &nmt_gpu, const gputils::Array<uint> &err_gpu,
		    bool periodic_xcoord = true, bool debug = false);
    
    long nsamp = 0;
    long nypix_global = 0;
    long nxpix_global = 0;
    bool periodic_xcoord;

    long plan_nbytes = 0;                  // length of 'buf' argument to PointingPlan constructor
    long plan_constructor_tmp_nbytes = 0;  // length of 'tmp_buf' argument to PointingPlan constructor
    double overhead = 0.0;                 // typically ~0.3 (meaning that cell decomposition is a ~30% overhead)

    // Used when launching planner/preplanner kernels.
    long ncl_per_threadblock = 0;
    long planner_nblocks = 0;

    // Used when launching pointing (tod2map/map2tod) kernels.
    long nmt_per_threadblock = 0;
    long pointing_nblocks = 0;

    // Used internally
    long plan_nmt = 0;                     // total number of mt-pairs in plan
    size_t cub_nbytes = 0;                 // number of bytes used in cub radix sort 'd_temp_storage'

    // Cumulative count of mt-pairs per threadblock.
    gputils::Array<uint> nmt_cumsum;

    // Copies nmt_cumsum array to host, and returns it as a numpy array.
    // Temporary hack, used in tests.test_pointing_preplan().
    gputils::Array<uint> get_nmt_cumsum() const;
    
    std::string str() const;
};


struct PointingPlan
{
    const long nsamp;
    const long nypix_global;
    const long nxpix_global;
    const bool periodic_xcoord;

    const PointingPrePlan pp;
    
    gputils::Array<unsigned char> buf;

    // The 'buf' array logically consists of two buffers:
    //   ulong plan_mt[nmt];    // where nmt = pp.plan_nmt
    //   uint err[B];           // where B = max(pp.planner_nblocks, pp.pointing_nblocks)
    //
    // The 'plan_mt' and 'err_gpu' pointers point directly to these buffers, for convenience.
   
    ulong *plan_mt = nullptr;
    uint *err_gpu = nullptr;    // 128-byte aligned

    // This constructor uses externally allocated GPU memory.
    template<typename T>
    PointingPlan(const PointingPrePlan &pp,
		 const gputils::Array<T> &xpointing_gpu,
		 const gputils::Array<unsigned char> &buf,
		 const gputils::Array<unsigned char> &tmp_buf,
		 bool debug = false);

    // This constructor allocates GPU memory.
    template<typename T>
    PointingPlan(const PointingPrePlan &pp,
		 const gputils::Array<T> &xpointing_gpu,
		 bool debug = false);


    // All arrays must be on the GPU.
    template<typename T>
    void map2tod(gputils::Array<T> &tod,
		 const gputils::Array<T> &local_map,
		 const gputils::Array<T> &xpointing,
		 const LocalPixelization &local_pixelization,
		 bool allow_outlier_pixels = false,
		 bool debug = false) const;
    
    // All arrays must be on the GPU.
    template<typename T>
    void tod2map(gputils::Array<T> &local_map,
		 const gputils::Array<T> &tod,
		 const gputils::Array<T> &xpointing,
		 const LocalPixelization &local_pixelization,
		 bool allow_outlier_pixels = false,
		 bool debug = false) const;

    
    // Used in unit tests.
    gputils::Array<ulong> get_plan_mt(bool gpu) const;

    std::string str() const;
};


// -----------------------------------------------------------------------------
//
// Internals + testing


// Slow single-threaded CPU map2tod/tod2map, for testing.

template<typename T>
extern void reference_map2tod(
    gputils::Array<T> &tod,              // shape (nsamp,) or (ndet,nt)
    const gputils::Array<T> &local_map,  // total size (3 * local_pixelization.npix)
    const gputils::Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization,
    bool periodic_xcoord,
    bool partial_pixelization
);

template<typename T>
extern void reference_tod2map(
    gputils::Array<T> &local_map,        // total size (3 * local_pixelization.npix)
    const gputils::Array<T> &tod,        // Shape (nsamp,) or (ndet,nt)
    const gputils::Array<T> &xpointing,  // Shape (3,nsamp) or (3,ndet,nt)     where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization,
    bool periodic_xcoord,
    bool partial_pixelization
);


// Array interface to map2tod.
// You probably want to call PointingPlan::map2tod(), not this function!

template<typename T>
extern void launch_map2tod(gputils::Array<T> &tod,
			   const gputils::Array<T> &local_map,
			   const gputils::Array<T> &xpointing,
			   const LocalPixelization &local_pixelization,
			   const ulong *plan_mt, uint *errflags,
			   long nmt, long nmt_per_block, long nblocks,
			   bool allow_outlier_pixels, bool debug);

// Bare-pointer interface to tod2map.
// You probably want to call PointingPlan::tod2map(), not this function!

template<typename T>
extern void launch_tod2map(gputils::Array<T> &local_map,
			   const gputils::Array<T> &tod,
			   const gputils::Array<T> &xpointing,
			   const LocalPixelization &local_pixelization,
			   const ulong *plan_mt, uint *errflags,
			   long nmt, long nmt_per_block, long nblocks,
			   bool allow_outlier_pixels, bool debug);

// "Unplanned" SHTs (intended for testing)
// Note: only allow_outlier_pixels=true is implemented!

template<typename T>
extern void launch_unplanned_map2tod(
    gputils::Array<T> &tod,              // shape (nsamp,) or (ndet,nt)
    const gputils::Array<T> &local_map,  // total size (3 * local_pixelization.npix)
    const gputils::Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization
);

template<typename T>
extern void launch_unplanned_tod2map(
    gputils::Array<T> &local_map,  // total size (3 * local_pixelization.npix)
    const gputils::Array<T> &tod,  // shape (nsamp,) or (ndet,nt)
    const gputils::Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization
);

template<typename T>
extern void launch_unplanned_tod2map(gputils::Array<T> &map, const gputils::Array<T> &tod, const gputils::Array<T> &xpointing);


template<typename T>
struct ToyPointing
{
    // Scans currently go at 45 degrees, and cover the full y-range.

    // Version of constructor which allocates xpointing arrays.
    // If ndet <= 0, then the xpointing arrays will have shape (3,nt).
    // If ndet > 0, then the xpointing arrays will have shape (3,ndet,nt).
    
    ToyPointing(long ndet, long nt,
		long nypix_global,
		long nxpix_global,
		double scan_speed,     // map pixels per TOD sample
		double total_drift,    // total drift over full TOD, in x-pixels
		bool noisy = true);

    // Version of constructor with externally allocated xpointing arrays (intended for python)
    ToyPointing(long nypix_global, long nxpix_global,
		double scan_speed, double total_drift,
		const gputils::Array<T> &xpointing_cpu,
		const gputils::Array<T> &xpointing_gpu,
		bool noisy = true);

    long nypix_global;
    long nxpix_global;
    double scan_speed;    // map pixels per TOD sample
    double total_drift;   // total drift over full TOD, in x-pixels
    double drift_speed;   // drift (in x-pixels) per TOD sample

    // Since ToyPointing is only used in unit tests, assume the caller
    // wants array copies on both CPU and GPU.
    
    gputils::Array<T> xpointing_cpu;
    gputils::Array<T> xpointing_gpu;

    std::string str() const;
};


// Argument checking (defined in check_arguments.cu)
// Note that TODs can have either shape (nsamp,) or (ndet,nt).
// Similarly, xpointing arrays can have either shape (3,nsamp) or (3,ndet,nt).

extern void check_nsamp(long nsamp, const char *where);
extern void check_nypix_global(long nypix_global, const char *where);
extern void check_nxpix_global(long nxpix_global, const char *where);

extern void check_err(uint err, const char *where);
extern void check_gpu_errflags(const uint *errflags_gpu, int nelts, const char *where, uint errflags_to_ignore = 0);

// Check arrays, in cases where we know the dimensions in advance.
template<typename T> extern void check_map(const gputils::Array<T> &map, long nypix_global, long nxpix_global, const char *where, bool on_gpu);
template<typename T> extern void check_tod(const gputils::Array<T> &tod, long nsamp, const char *where, bool on_gpu);
template<typename T> extern void check_xpointing(const gputils::Array<T> &xpointing, long nsamp, const char *where, bool on_gpu);
template<typename T> extern void check_local_map(const gputils::Array<T> &map, const LocalPixelization &lpix, const char *where, bool on_gpu);
extern void check_cell_offsets(const gputils::Array<long> &cell_offsets, long nycells_expected, long nxcells_expected, const char *where, bool on_gpu);
extern void check_buffer(const gputils::Array<unsigned char> &buf, long min_nbytes, const char *where, const char *bufname);

// Check arrays, in cases where we do not know the dimensions in advance.
template<typename T> extern void check_map_and_init_npix(const gputils::Array<T> &map, long &nypix_global, long &nxpix_global, const char *where, bool on_gpu);
template<typename T> extern void check_tod_and_init_nsamp(const gputils::Array<T> &tod, long &nsamp, const char *where, bool on_gpu);
template<typename T> extern void check_xpointing_and_init_nsamp(const gputils::Array<T> &xpointing, long &nsamp, const char *where, bool on_gpu);
extern void check_cell_offsets_and_init_ncells(const gputils::Array<long> &cell_offsets, long &nycells, long &nxcells, const char *where, bool on_gpu);


// ReferencePointingPlan: used in unit tests.
//
// Given an 'xpointing' array on the GPU, determine which map pixel (iy, ix)
// each time sample falls into, and store the result in two length-nsamp arrays
// (iypix_cpu, ixpix_cpu). (Since this class is only used in unit tests, we assume
// that the caller wants these arrays on the CPU.)

struct ReferencePointingPlan
{
    // Version of constructor which allocates temporary arrays.
    template<typename T>
    ReferencePointingPlan(const PointingPrePlan &pp, const gputils::Array<T> &xpointing_gpu);

    // Version of constructor with externally allocated tmp array (intended for python)
    template<typename T>
    ReferencePointingPlan(const PointingPrePlan &pp,
			  const gputils::Array<T> &xpointing_gpu,
			  const gputils::Array<unsigned char> &tmp);    

    // Same meaning as in PointingPrePlan.
    long nsamp = 0;
    long nypix_global = 0;
    long nxpix_global = 0;
    long plan_nmt = 0;
    long ncl_per_threadblock = 0;
    long planner_nblocks = 0;

    // All arrays are on the CPU.
    // (iypix, ixpix) = which map pixel does each time sample fall into?
    // (Computed on GPU and copied to CPU, in order to guarantee roundoff consistency with other GPU code.)
    
    gputils::Array<int> iypix_arr;   // length nsamp
    gputils::Array<int> ixpix_arr;   // length nsamp

    gputils::Array<uint> nmt_cumsum;  // length planner_nblocks, same meaning as PointingPrePlan::nmt_cumsum.
    gputils::Array<ulong> sorted_mt;  // length plan_nmt, see PointingPlan for 'mt' format.

    // Used temporarily in constructor.
    int _tmp_cells[128];
    int _ntmp_cells = 0;
    void _add_tmp_cell(int iypix, int ixcpix);

    std::string str() const;

    // Helpers for python constructor logic.
    static long get_constructor_tmp_nbytes(const PointingPrePlan &pp);
    static constexpr int warps_per_threadblock = 4;
    static constexpr int nsamp_per_block = 1024;
};


// -------------------------------------------------------------------------------------------------
//
// map2tod/tod2map (old version)
//
// Warning!! Caller must ensure that
//   - All elements of xpointing[0,:,:] are in [0.0, nra-1)
//   - All elements of xpointing[1,:,:] are in [0.0, ndec-1)
//
// If this condition is not satisfied, then the map2tod kernel will segfault or return nonsense.
// In particular, map2tod doesn't know that the ra coordinate should be periodic.
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




extern void launch_old_map2tod(
    gputils::Array<float> &tod,              // shape (nsamp,) or (ndet,nt)
    const gputils::Array<float> &map,        // shape (3,nypix_global,nxpix_global)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &xpointing   // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
);


extern void launch_old_tod2map(
    gputils::Array<float> &map,                  // Shape (3,nypix_global,nxpix_global)   where axis 0 = {I,Q,U}
    const gputils::Array<float> &tod,            // Shape (nsamp,) or (ndet,nt)
    const gputils::Array<float> &xpointing,      // Shape (3,nsamp) or (3,ndet,nt)     where axis 0 = {y,x,alpha}
    const gputils::Array<int> &plan_cltod_list,  // Shape (plan_ncltod,)
    const gputils::Array<int> &plan_quadruples   // Shape (plan_nquadruples, 4)
);


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
