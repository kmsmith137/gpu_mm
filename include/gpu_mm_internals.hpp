#ifndef _GPU_MM_INTERNALS_HPP
#define _GPU_MM_INTERNALS_HPP

#include <cassert>

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif

static constexpr unsigned int ALL_LANES = 0xffffffffU;


inline long align128(long n)
{
    assert(n >= 0);
    return (n + 127L) & ~127L;
}


// -------------------------------------------------------------------------------------------------
//
// Flags for communicating errors from GPU to CPU.


static constexpr int errflag_bad_ypix = 0x1;
static constexpr int errflag_bad_xpix = 0x2;
static constexpr int errflag_inconsistent_nmt = 0x4;
static constexpr int errflag_not_in_pixelization = 0x8;


// write_errflags(): called from device code, to write errflags to global CPU memory.
// Warning: Assumes thread layout is {32,W,1}, and block layout is {B,1,1}!
// Warning: caller may need to call __syncthreads() before and/or after.

static __device__ void write_errflags(uint *gp, uint *sp, uint local_err)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;
    int nwarps = blockDim.y;
    
    local_err = __reduce_or_sync(ALL_LANES, local_err);
    
    if (laneId == 0)
	sp[warpId] = local_err;
    
    __syncthreads();

    if (warpId != 0)
	return;
    
    local_err = (laneId < nwarps) ? sp[laneId] : 0;
    local_err = __reduce_or_sync(ALL_LANES, local_err);
    
    if (laneId == 0)
	gp[blockIdx.x] = local_err;
}


// -------------------------------------------------------------------------------------------------
//
// Some boilerplate, used to support T=float and T=double with the same C++ template.


template<typename T> struct dtype {};

template<> struct dtype<float>
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
    static __host__ __device__ void xsincos(float x, float *sptr, float *cptr) { sincosf(x, sptr, cptr); }
    static __device__ float *get_shmem() { extern __shared__ float shmem_f[]; return shmem_f; }
};


template<> struct dtype<double>
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html
    static __host__ __device__ void xsincos(double x, double *sptr, double *cptr) { sincos(x, sptr, cptr); }
    static __device__ double *get_shmem() { extern __shared__ double shmem_d[]; return shmem_d; }
};

template<> struct dtype<int>
{
    static __device__ int *get_shmem() { extern __shared__ int shmem_i[]; return shmem_i; }
};


// -------------------------------------------------------------------------------------------------
//
// The rest of this file consists of some __device__ inline classes:
//
//  - pixel_locator: (ypix, xpix) -> (iy0, iy1, ix0, ix1, dy, dx)
//      [ given nypix_global, nx_global, periodic_xcoord ]
//
//  - cell_enumerator: called by planner to map (ypix, xpix) -> (up to 4 cells)
//
//  - map_evaluator: (local_map, pixel_locator output) -> (interpolated map value)
//      [ used in unplanned_map2tod(), reference_map2tod(), but not the "main" mapt2tod() ]
//
//  - map_accumulator: (local_map, pixel_locator output, tod value) -> (updates map)
//      [ used in unplanned_tod2map(), reference_tod2map(), but not the "main" mapt2tod() ]


template<typename T>
struct pixel_locator
{
    const int nypix_global;
    const int nxpix_global;
    const bool periodic_xcoord;
    
    // Pixel indices and offsets are omputed in locate().
    // Note: if (ypix, xpix) are bad, then locate() sets the pixel indices (iy0, iy1, ix0, ix1)
    // to something in-range, but sets errflag_bad_{xy}pix, so that an exception gets thrown later.
    
    int iy0, iy1;  // will always satisfy (0 <= iy < nypix_global), even if 'ypix' is bad.
    int ix0, ix1;  // will always satisfy (0 <= iy < nypix_global), even if 'xpix' is bad.
    T dy, dx;
    
    __host__ __device__ inline pixel_locator(int nypix_global_, int nxpix_global_, bool periodic_xcoord_)
	: nypix_global(nypix_global_), nxpix_global(nxpix_global_), periodic_xcoord(periodic_xcoord_)
    { }

    // Locate pixel in global coordinates.
    __host__ __device__ inline void locate(T ypix, T xpix, uint &err)
    {
	iy0 = int(ypix);
	iy0 = max(iy0, 0);
	iy0 = min(iy0, nypix_global-2);
	iy1 = iy0 + 1;
	dy = ypix - iy0;

	bool yvalid = (ypix >= 0) && (ypix <= nypix_global-1);
	err = yvalid ? err : (err | errflag_bad_ypix);
	
	const int ix0_max = periodic_xcoord ? (nxpix_global-1) : (nxpix_global-2);
	const T xmin = periodic_xcoord ? (-nxpix_global) : 0;
	const T xmax = periodic_xcoord ? (2*nxpix_global) : (nxpix_global-1);

	xpix = (xpix >= 0) ? xpix : (xpix + nxpix_global);
	xpix = (xpix <= nxpix_global) ? xpix : (xpix - nxpix_global);
	
	ix0 = int(xpix);
	ix0 = max(ix0, 0);
	ix0 = min(ix0, ix0_max);
	ix1 = ix0 + 1;
	ix1 = (ix1 < nxpix_global) ? ix1 : 0;   // wrap around (periodic case only)
	dx = xpix - ix0;
	
	bool xvalid = (xpix >= xmin) && (xpix <= xmax);
	err = xvalid ? err : (err | errflag_bad_xpix);
    }

    // Locate pixel in "cell coordinates".
    __host__ __device__ inline void locate(T ypix, T xpix, int iycell, int ixcell, uint &err)
    {
	locate(ypix, xpix, err);
	
	iy0 -= (iycell << 6);
	iy1 -= (iycell << 6);
	ix0 -= (ixcell << 6);
	ix1 -= (ixcell << 6);
    }
};


// Used in preplanner, planner.
template<typename T, bool Debug>
struct cell_enumerator
{
    // Cell indices, not pixel indices!
    int iy0, iy1;
    int ix0, ix1, ix2;

    pixel_locator<T> _px;

    __device__ inline cell_enumerator(int nypix_global_, int nxpix_global_, bool periodic_xcoord_)
	: _px(nypix_global_, nxpix_global_, periodic_xcoord_)
    { }

    // Helper for enumerate()
    __device__ inline void _process_pair(int icell0, int icell1, int &icell_even, int &icell_odd)
    {
	icell1 = (icell0 != icell1) ? icell1 : -1;
	
	bool flag = (icell0 & 1);
	icell_even = flag ? icell1 : icell0;
	icell_odd = flag ? icell0 : icell1;
    }
				   
    __device__ inline void enumerate(T ypix, T xpix, uint &err)
    {
	_px.locate(ypix, xpix, err);

	int iycell0 = (_px.iy0 >> 6);
	int iycell1 = (_px.iy1 >> 6);
	int ixcell0 = (_px.ix0 >> 6);
	int ixcell1 = (_px.ix1 >> 6);

	_process_pair(iycell0, iycell1, iy0, iy1);
	_process_pair(ixcell0, ixcell1, ix0, ix1);

	ix2 = (ix0 != 0) ? -1 : 0;
	ix0 = (ix0 != 0) ? ix0 : -1;
	
	if constexpr (Debug) {
	    assert((iycell0 == iy0) || (iycell0 == iy1));
	    assert((iycell1 == iy0) || (iycell1 == iy1));	    
	    assert((ixcell0 == ix0) || (ixcell0 == ix1) || (ixcell0 == ix2));
	    assert((ixcell1 == ix0) || (ixcell1 == ix1) || (ixcell1 == ix2));
		    
	    assert((iy0 < 0) || !(iy0 & 1));  // negative or even
	    assert((iy1 < 0) || (iy1 & 1));   // negative or odd
	    assert((ix0 < 0) || (ix0 && !(ix0 & 1)));  // negative or even nonzero
	    assert((ix1 < 0) || (ix1 & 1));   // negative or odd
	    assert(ix2 <= 0);                 // negative or zero
	}
    }
};


// map_evaluator: used in unplanned_map2tod(), reference_map2tod().
// (Not used in the "main" map2tod().)

template<typename T, bool Device>
struct map_evaluator
{
    const T *lmap;
    const long *cell_offsets;
    const int nycells;
    const int nxcells;
    const long ystride;
    const long polstride;
    const bool partial_pixelization;

    __host__ __device__ inline map_evaluator(const T *lmap_, const long *cell_offsets_, int nycells_, int nxcells_, long ystride_, long polstride_, bool partial_pixelization_)
	: lmap(lmap_), cell_offsets(cell_offsets_), nycells(nycells_), nxcells(nxcells_), ystride(ystride_), polstride(polstride_), partial_pixelization(partial_pixelization_)
    { }

    __host__ __device__ inline T eval(int iy, int ix, T cos_2a, T sin_2a, uint &err) const
    {
	int iycell = iy >> 6;
	int ixcell = ix >> 6;
	
	bool valid = (iy >= 0) && (ix >= 0) && (iycell < nycells) && (ixcell < nxcells);
	long offset = valid ? cell_offsets[iycell*nxcells + ixcell] : -1;

	if constexpr (Device)
	    __syncwarp();

	valid = (offset >= 0);
	offset += (ix - (ixcell << 6));
	offset += (iy - (iycell << 6)) * ystride;

	T t = valid ? lmap[offset] : 0;
	T q = valid ? lmap[offset + polstride] : 0;
	T u = valid ? lmap[offset + 2*polstride] : 0;

	if constexpr (Device)
	    __syncwarp();

	err = (valid || partial_pixelization) ? err : (err | errflag_not_in_pixelization);
	return t + cos_2a*q + sin_2a*u;
    }

    __host__ __device__ inline T eval(const pixel_locator<T> &px, T cos_2a, T sin_2a, uint &err) const
    {
	T ret = (1-px.dy) * (1-px.dx) * eval(px.iy0, px.ix0, cos_2a, sin_2a, err);
	ret +=  (1-px.dy) *   (px.dx) * eval(px.iy0, px.ix1, cos_2a, sin_2a, err);
        ret +=    (px.dy) * (1-px.dx) * eval(px.iy1, px.ix0, cos_2a, sin_2a, err);
        ret +=    (px.dy) *   (px.dx) * eval(px.iy1, px.ix1, cos_2a, sin_2a, err);
	
	return ret;
    }
};


// map_accumulator: used in unplanned_tod2map(), reference_tod2map(),
// (Not used in the "main" tod2map().)

template<typename T, bool Device>
struct map_accumulator
{
    T *lmap;
    const long *cell_offsets;
    const int nycells;
    const int nxcells;
    const long ystride;
    const long polstride;
    const bool partial_pixelization;

    __host__ __device__ inline map_accumulator(T *lmap_, const long *cell_offsets_, int nycells_, int nxcells_, long ystride_, long polstride_, bool partial_pixelization_)
	: lmap(lmap_), cell_offsets(cell_offsets_), nycells(nycells_), nxcells(nxcells_), ystride(ystride_), polstride(polstride_), partial_pixelization(partial_pixelization_)
    { }

    __host__ __device__ inline void accum(int iy, int ix, T t, T q, T u, T w, uint &err) const
    {
	int iycell = iy >> 6;
	int ixcell = ix >> 6;
	
	bool valid = (iy >= 0) && (ix >= 0) && (iycell < nycells) && (ixcell < nxcells);
	long offset = valid ? cell_offsets[iycell*nxcells + ixcell] : -1;

	if constexpr (Device)
	    __syncwarp();

	valid = (offset >= 0);
	offset += (ix - (ixcell << 6));
	offset += (iy - (iycell << 6)) * ystride;

	if (valid) {
	    if constexpr (Device) {
		atomicAdd(lmap + offset, w*t);
		atomicAdd(lmap + offset + polstride, w*q);
		atomicAdd(lmap + offset + 2*polstride, w*u);
	    }
	    else {
		lmap[offset] += w*t;
		lmap[offset + polstride] += w*q;
		lmap[offset + 2*polstride] += w*u;
	    }
	}

	if constexpr (Device)
	    __syncwarp();

	err = (valid || partial_pixelization) ? err : (err | errflag_not_in_pixelization);
    }

    __host__ __device__ inline void accum(const pixel_locator<T> &px, T t, T q, T u, uint &err) const
    {
	accum(px.iy0, px.ix0, t, q, u, (1-px.dy) * (1-px.dx), err);
	accum(px.iy0, px.ix1, t, q, u, (1-px.dy) *   (px.dx), err);
	accum(px.iy1, px.ix0, t, q, u,   (px.dy) * (1-px.dx), err);
	accum(px.iy1, px.ix1, t, q, u,   (px.dy) *   (px.dx), err);
    }
};


} // namespace gpu_mm

#endif //  _GPU_MM_INTERNALS_HPP
