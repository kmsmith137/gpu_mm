#ifndef _GPU_MM_INTERNALS_HPP
#define _GPU_MM_INTERNALS_HPP

#include <cassert>
#include <gputils/Array.hpp>

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
static constexpr int errflag_pixel_outlier = 0x10;   // FIXME remove


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
// FIXME these functions could use more comments!


// Returns integer in the range -2 <= ix <= npix.
template<typename T>
static __host__ __device__ inline int quantize_pixel(T xpix, int npix)
{
    // Clamp xpix so that integer conversion can't go haywire, ensuring roundoff-robustness.
    xpix = max(xpix, T(-1.5));
    xpix = min(xpix, npix + T(0.5));

    // The +/- 2 ensures that int(...) always rounds down (since it gets applied to a positive number).
    return int(xpix + T(2)) - 2;
}


template<typename T>
static __device__ void range_check_ypix(T ypix, int nypix, uint &err)
{
    bool valid = (ypix >= 0) && (ypix <= nypix-1);
    err = valid ? err : (err | errflag_pixel_outlier);
}


template<typename T>
static __device__ void range_check_xpix(T xpix, int nxpix, uint &err)
{
    bool valid = (xpix >= 0) && (xpix <= nxpix-1);
    err = valid ? err : (err | errflag_pixel_outlier);
}


template<typename T>
static __device__ void quantize_ypix(int &iypix0, int &iypix1, T ypix, int nypix)
{
    // Assumes range_check_ypix() has been called.
    iypix0 = int(ypix);
    iypix0 = (iypix0 >= 0) ? iypix0 : 0;
    iypix0 = (iypix0 <= nypix-2) ? iypix0 : (nypix-2);
    iypix1 = iypix0 + 1;
}


template<typename T>
static __device__ void quantize_xpix(int &ixpix0, int &ixpix1, T xpix, int nxpix)
{
    // Assumes range_check_xpix() and normalize_xpix() have been called.
    ixpix0 = int(xpix);
    ixpix0 = (ixpix0 >= 0) ? ixpix0 : 0;
    ixpix0 = (ixpix0 <= nxpix-1) ? ixpix0 : (nxpix-1);
    ixpix1 = (ixpix0 < (nxpix-1)) ? (ixpix0+1) : 0;
}


static __device__ void set_up_cell_pair(int &icell_e, int &icell_o, int ipix0, int ipix1)
{
    int icell0 = ipix0 >> 6;
    int icell1 = ipix1 >> 6;
    icell1 = (icell0 != icell1) ? icell1 : -1;

    bool flag = (icell0 & 1);
    icell_e = flag ? icell1 : icell0;
    icell_o = flag ? icell0 : icell1;
}


template<typename T>
struct pixel_locator
{
    const int nypix;
    const int nxpix;
    const bool periodic_xcoord;
    
    // Pixel indices and offsets are omputed in locate().
    // Note: if (ypix, xpix) are bad, then locate() sets the pixel indices (iy0, iy1, ix0, ix1)
    // to something in-range, but sets errflag_bad_{xy}pix, so that an exception gets thrown later.
    
    int iy0, iy1;  // will always satisfy (0 <= iy < nypix), even if 'ypix' is bad.
    int ix0, ix1;  // will always satisfy (0 <= iy < nypix), even if 'xpix' is bad.
    T dy, dx;
    
    __host__ __device__ inline pixel_locator(int nypix_, int nxpix_, bool periodic_xcoord_)
	: nypix(nypix_), nxpix(nxpix_), periodic_xcoord(periodic_xcoord_)
    { }

    __host__ __device__ inline void locate(T ypix, T xpix, uint &err)
    {
	iy0 = int(ypix);
	iy0 = max(iy0, 0);
	iy0 = min(iy0, nypix-2);
	iy1 = iy0 + 1;
	dy = ypix - iy0;

	bool yvalid = (ypix >= 0) && (ypix <= nypix-1);
	err = yvalid ? err : (err | errflag_bad_ypix);
	
	const int ix0_max = periodic_xcoord ? (nxpix-1) : (nxpix-2);
	const T xmin = periodic_xcoord ? (-nxpix) : 0;
	const T xmax = periodic_xcoord ? (2*nxpix) : (nxpix-1);

	xpix = (xpix >= 0) ? xpix : (xpix + nxpix);
	xpix = (xpix <= nxpix) ? xpix : (xpix - nxpix);
	
	ix0 = int(xpix);
	ix0 = max(ix0, 0);
	ix0 = min(ix0, ix0_max);
	ix1 = ix0 + 1;
	ix1 = (ix1 < nxpix) ? ix1 : 0;   // wrap around (periodic case only)
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
struct cell_enumerator
{
    int iy0, iy1;
    int ix0, ix1;

    template<typename T>
    __device__ cell_enumerator(T ypix, T xpix, int nypix, int nxpix, uint &err)
    {
	range_check_ypix(ypix, nypix, err);  // defined in gpu_mm_internals.hpp
	range_check_xpix(xpix, nxpix, err);  // defined in gpu_mm_internals.hpp
	
	int iypix0, iypix1, ixpix0, ixpix1;
	quantize_ypix(iypix0, iypix1, ypix, nypix);  // defined in gpu_mm_internals.hpp
	quantize_xpix(ixpix0, ixpix1, xpix, nxpix);  // defined in gpu_mm_internals.hpp

	set_up_cell_pair(iy0, iy1, iypix0, iypix1);  // defined in gpu_mm_internals.hpp
	set_up_cell_pair(ix0, ix1, ixpix0, ixpix1);  // defined in gpu_mm_internals.hpp
    }
};


// map_evaluator: used in unplanned_map2tod(), reference_map2tod(),
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
};


} // namespace gpu_mm

#endif //  _GPU_MM_INTERNALS_HPP
