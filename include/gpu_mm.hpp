#ifndef _GPU_MM_HPP
#define _GPU_MM_HPP

#include <ksgpu/Array.hpp>

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
		      const ksgpu::Array<long> &cell_offsets,
		      long ystride, long polstride,
		      bool periodic_xcoord = true);

    // This constructor is intended to be called from python.
    // The caller is responsible for ensuring that the 'cell_offsets_cpu'
    // and 'cell_offsets_gpu' arrays have the same contents!
    
    LocalPixelization(long nypix_global, long nxpix_global,
		      const ksgpu::Array<long> &cell_offsets_cpu,
		      const ksgpu::Array<long> &cell_offsets_gpu,
		      long ystride, long polstride,
		      bool periodic_xcoord = true);

    // Global pixelization
    const long nypix_global;
    const long nxpix_global;
    const bool periodic_xcoord;

    // Local pixelization
    const ksgpu::Array<long> cell_offsets_cpu;
    const ksgpu::Array<long> cell_offsets_gpu;
    const long ystride;
    const long polstride;
    
    long nycells;   // same as cell_offsets.shape[0]
    long nxcells;   // same as cell_offsets.shape[1]
    long npix;      // counts only local pixels, does not include factor 3 from TQU.
};


struct PointingPrePlan
{
    static constexpr int preplan_size = 1024;

    // This constructor allocates GPU memory, and is intended to be called from C++.
    
    template<typename T>
    PointingPrePlan(const ksgpu::Array<T> &xpointing_gpu, long nypix_global, long nxpix_global,
		    bool periodic_xcoord = true, bool debug = false);
    
    // This constructor uses externally allocated GPU memory, and is intended to be called from python.
    // The 'nmt_gpu' and 'err_gpu' arrays should have length preplan_size.
    
    template<typename T>
    PointingPrePlan(const ksgpu::Array<T> &xpointing_gpu, long nypix_global, long nxpix_global,
		    const ksgpu::Array<uint> &nmt_gpu, const ksgpu::Array<uint> &err_gpu,
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
    ksgpu::Array<uint> nmt_cumsum;

    // Copies nmt_cumsum array to host, and returns it as a numpy array.
    // Temporary hack, used in tests.test_pointing_preplan().
    ksgpu::Array<uint> get_nmt_cumsum() const;
    
    std::string str() const;
};


struct PointingPlan
{
    const long nsamp;
    const long nypix_global;
    const long nxpix_global;
    const bool periodic_xcoord;

    const PointingPrePlan pp;
    
    ksgpu::Array<unsigned char> buf;

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
		 const ksgpu::Array<T> &xpointing_gpu,
		 const ksgpu::Array<unsigned char> &buf,
		 const ksgpu::Array<unsigned char> &tmp_buf,
		 bool debug = false);

    // This constructor allocates GPU memory.
    template<typename T>
    PointingPlan(const PointingPrePlan &pp,
		 const ksgpu::Array<T> &xpointing_gpu,
		 bool debug = false);
    
    // Used in unit tests.
    ksgpu::Array<ulong> get_plan_mt(bool gpu) const;

    std::string str() const;
};


template<typename T>
extern void launch_planned_map2tod(
    ksgpu::Array<T> &tod,                       // shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<T> &local_map,           // total size (3 * local_pixelization.npix)
    const ksgpu::Array<T> &xpointing,           // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization, 
    const PointingPlan &plan,
    bool partial_pixelization,
    bool debug
);


template<typename T>
extern void launch_planned_tod2map(
    ksgpu::Array<T> &local_map,                 // total size (3 * local_pixelization.npix)
    const ksgpu::Array<T> &tod,                 // shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<T> &xpointing,           // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization, 
    const PointingPlan &plan,
    bool partial_pixelization,
    bool debug
);


template<typename T>
extern void launch_unplanned_map2tod(
    ksgpu::Array<T> &tod,              // shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<T> &local_map,  // total size (3 * local_pixelization.npix)
    const ksgpu::Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization,
    ksgpu::Array<uint> &errflags,      // length nblocks, where nblocks is caller-supplied.
    bool partial_pixelization
);


template<typename T>
extern void launch_unplanned_tod2map(
    ksgpu::Array<T> &local_map,        // total size (3 * local_pixelization.npix)
    const ksgpu::Array<T> &tod,        // shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization,
    ksgpu::Array<uint> &errflags,      // length nblocks, where nblocks is caller-supplied.
    bool partial_pixelization
);


// Slow, single-threaded, CPU implementation for testing.
template<typename T>
extern void reference_map2tod(
    ksgpu::Array<T> &tod,              // shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<T> &local_map,  // total size (3 * local_pixelization.npix)
    const ksgpu::Array<T> &xpointing,  // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization,
    bool partial_pixelization
);


// Slow, single-threaded, CPU implementation for testing.
template<typename T>
extern void reference_tod2map(
    ksgpu::Array<T> &local_map,        // total size (3 * local_pixelization.npix)
    const ksgpu::Array<T> &tod,        // Shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<T> &xpointing,  // Shape (3,nsamp) or (3,ndet,nt)     where axis 0 = {y,x,alpha}
    const LocalPixelization &local_pixelization,
    bool partial_pixelization
);


// -----------------------------------------------------------------------------
//
// Internals + testing


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
		const ksgpu::Array<T> &xpointing_cpu,
		const ksgpu::Array<T> &xpointing_gpu,
		bool noisy = true);

    long nypix_global;
    long nxpix_global;
    double scan_speed;    // map pixels per TOD sample
    double total_drift;   // total drift over full TOD, in x-pixels
    double drift_speed;   // drift (in x-pixels) per TOD sample

    // Since ToyPointing is only used in unit tests, assume the caller
    // wants array copies on both CPU and GPU.
    
    ksgpu::Array<T> xpointing_cpu;
    ksgpu::Array<T> xpointing_gpu;

    std::string str() const;
};


// Argument checking (defined in check_arguments.cu)
// Note that TODs can have either shape (nsamp,) or (ndet,nt).
// Similarly, xpointing arrays can have either shape (3,nsamp) or (3,ndet,nt).

extern void check_nsamp(long nsamp, const char *where);
extern void check_nypix_global(long nypix_global, const char *where);
extern void check_nxpix_global(long nxpix_global, const char *where);

extern void check_err(uint err, const char *where, uint errflags_to_ignore = 0);
extern void check_cpu_errflags(const uint *errflags_cpu, int nelts, const char *where, uint errflags_to_ignore = 0);
extern void check_gpu_errflags(const uint *errflags_gpu, int nelts, const char *where, uint errflags_to_ignore = 0);

// Check arrays, in cases where we know the dimensions in advance.
template<typename T> extern void check_tod(const ksgpu::Array<T> &tod, long nsamp, const char *where, bool on_gpu);
template<typename T> extern void check_xpointing(const ksgpu::Array<T> &xpointing, long nsamp, const char *where, bool on_gpu);
template<typename T> extern void check_global_map(const ksgpu::Array<T> &map, long nypix_global, long nxpix_global, const char *where, bool on_gpu);
template<typename T> extern void check_local_map(const ksgpu::Array<T> &map, const LocalPixelization &lpix, const char *where, bool on_gpu);
extern void check_cell_offsets(const ksgpu::Array<long> &cell_offsets, long nycells_expected, long nxcells_expected, const char *where, bool on_gpu);
extern void check_buffer(const ksgpu::Array<unsigned char> &buf, long min_nbytes, const char *where, const char *bufname);

// Check arrays, in cases where we do not know the dimensions in advance.
template<typename T> extern void check_tod_and_init_nsamp(const ksgpu::Array<T> &tod, long &nsamp, const char *where, bool on_gpu);
template<typename T> extern void check_xpointing_and_init_nsamp(const ksgpu::Array<T> &xpointing, long &nsamp, const char *where, bool on_gpu);
template<typename T> extern void check_global_map_and_init_npix(const ksgpu::Array<T> &map, long &nypix_global, long &nxpix_global, const char *where, bool on_gpu);
extern void check_cell_offsets_and_init_ncells(const ksgpu::Array<long> &cell_offsets, long &nycells, long &nxcells, const char *where, bool on_gpu);


// PointingPlanTester: used in unit tests.
//
// Given an 'xpointing' array on the GPU, determine which map pixel (iy, ix)
// each time sample falls into, and store the result in two length-nsamp arrays
// (iypix_cpu, ixpix_cpu). (Since this class is only used in unit tests, we assume
// that the caller wants these arrays on the CPU.)

struct PointingPlanTester
{
    // Version of constructor which allocates temporary arrays.
    template<typename T>
    PointingPlanTester(const PointingPrePlan &pp, const ksgpu::Array<T> &xpointing_gpu);

    // Version of constructor with externally allocated tmp array (intended for python)
    template<typename T>
    PointingPlanTester(const PointingPrePlan &pp,
		       const ksgpu::Array<T> &xpointing_gpu,
		       const ksgpu::Array<unsigned char> &tmp);    

    // Same meaning as in PointingPrePlan.
    long nsamp = 0;
    long nypix_global = 0;
    long nxpix_global = 0;
    bool periodic_xcoord;
    
    long plan_nmt = 0;
    long ncl_per_threadblock = 0;
    long planner_nblocks = 0;

    // All arrays are on the CPU.
    // (iypix, ixpix) = which map pixel does each time sample fall into?
    // (Computed on GPU and copied to CPU, in order to guarantee roundoff consistency with other GPU code.)
    
    ksgpu::Array<int> iypix_arr;   // length nsamp
    ksgpu::Array<int> ixpix_arr;   // length nsamp

    ksgpu::Array<uint> nmt_cumsum;  // length planner_nblocks, same meaning as PointingPrePlan::nmt_cumsum.
    ksgpu::Array<ulong> sorted_mt;  // length plan_nmt, see PointingPlan for 'mt' format.

    // Used temporarily in constructor.
    int _tmp_cells[128];
    int _ntmp_cells = 0;
    void _add_tmp_cell(int iypix, int ixcpix);

    std::string str() const;

    // Helpers for python constructor logic.
    static long get_constructor_tmp_nbytes(const PointingPrePlan &pp);
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
    ksgpu::Array<float> &tod,              // shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<float> &map,        // shape (3,nypix_global,nxpix_global)   where axis 0 = {I,Q,U}
    const ksgpu::Array<float> &xpointing   // shape (3,nsamp) or (3,ndet,nt)    where axis 0 = {y,x,alpha}
);


extern void launch_old_tod2map(
    ksgpu::Array<float> &map,                  // Shape (3,nypix_global,nxpix_global)   where axis 0 = {I,Q,U}
    const ksgpu::Array<float> &tod,            // Shape (nsamp,) or (ndet,nt)
    const ksgpu::Array<float> &xpointing,      // Shape (3,nsamp) or (3,ndet,nt)     where axis 0 = {y,x,alpha}
    const ksgpu::Array<int> &plan_cltod_list,  // Shape (plan_ncltod,)
    const ksgpu::Array<int> &plan_quadruples   // Shape (plan_nquadruples, 4)
);


// Temporary hack: construct pointing plans on CPU.
struct OldPointingPlan
{
    OldPointingPlan(const ksgpu::Array<float> &xpointing, int ndec, int nra, bool verbose=true);

    long ncl_uninflated = 0;
    long ncl_inflated = 0;     // same as 'plan_ncltod' argument to tod2map()
    long num_quadruples = 0;   // same as 'plan_nquadruples' argument to tod2map()
    
    ksgpu::Array<int> plan_cltod_list;    // 1-d contiguous array of shape (ncl_inflated,)
    ksgpu::Array<int> plan_quadruples;    // 2-d contiguous array of shape (num_quadruples,4)
};


} // namespace gpu_mm

#endif //  _GPU_MM_HPP
