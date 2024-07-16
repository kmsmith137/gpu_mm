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


// Flags for communicating errors from GPU to CPU.
static constexpr int errflag_pixel_outlier = 0x1;
static constexpr int errflag_inconsistent_nmt = 0x4;


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
// FIXME these functions could use some comments!


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


} // namespace gpu_mm

#endif //  _GPU_MM_INTERNALS_HPP
