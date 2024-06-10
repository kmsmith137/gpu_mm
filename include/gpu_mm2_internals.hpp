#ifndef _GPU_MM2_INTERNALS_HPP
#define _GPU_MM2_INTERNALS_HPP

#include <gputils/Array.hpp>

namespace gpu_mm2 {
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
// FIXME these functions could use some comments!


template<typename T>
static __device__ void range_check_ypix(T ypix, int nypix, uint &err)
{
    bool valid = (ypix >= 0) && (ypix <= nypix-1);
    err = valid ? err : (err | 1);
}


template<typename T>
static __device__ void range_check_xpix(T xpix, int nxpix, uint &err)
{
    bool valid = (xpix >= -nxpix) && (xpix <= 2*nxpix);
    err = valid ? err : (err | 2);
}


template<typename T>
static __device__ void normalize_xpix(T &xpix, int nxpix)
{
    xpix = (xpix >= 0) ? xpix : (xpix + nxpix);
    xpix = (xpix <= nxpix) ? xpix : (xpix - nxpix);
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


static __device__ uint count_nmt(int iycell, int ixcell)
{
    uint icell = (iycell << 10) | ixcell;
    bool valid = (iycell >= 0) && (ixcell >= 0);

    int laneId = threadIdx.x & 31;
    uint lmask = (1U << laneId) - 1;   // all lanes lower than current lane
    uint mmask = __match_any_sync(ALL_LANES, icell);  // all matching lanes
    bool is_lowest = ((mmask & lmask) == 0);
	
    return (valid && is_lowest) ? 1 : 0;
}


// FIXME refactor common code in count_nmt(), analyze_cell_pair().
static __device__ void analyze_cell_pair(int iycell, int ixcell, uint &icell, uint &amask, int &na)
{
    icell = (iycell << 10) | ixcell;
    bool valid = (iycell >= 0) && (ixcell >= 0);

    int laneId = threadIdx.x & 31;
    uint lmask = (1U << laneId) - 1;   // all lanes lower than current lane
    uint mmask = __match_any_sync(ALL_LANES, icell);  // all matching lanes
    bool is_lowest = ((mmask & lmask) == 0);

    amask = __ballot_sync(ALL_LANES, valid && is_lowest);
    na = __popc(amask);
}


} // namespace gpu_mm2

#endif //  _GPU_MM2_INTERNALS_HPP
