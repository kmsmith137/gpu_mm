#include "../include/gpu_mm.hpp"

using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


template<typename T>
void local_map_to_global(const LocalPixelization &local_pixelization, Array<T> &dst, const Array<T> &src)
{
    long nypix_global = local_pixelization.nypix_global;
    long nxpix_global = local_pixelization.nxpix_global;
    long nycells = local_pixelization.nycells;
    long nxcells = local_pixelization.nxcells;
    long ystride = local_pixelization.ystride;
    long polstride = local_pixelization.polstride;
    const long *cell_offsets = local_pixelization.cell_offsets_cpu.data;

    check_global_map(dst, nypix_global, nxpix_global, "local_map_to_global", false);  // on_gpu=false
    check_local_map(src, local_pixelization, "local_map_to_global", false);           // on_gpu=false
	
    for (long iycell = 0; iycell < nycells; iycell++) {
	for (long ixcell = 0; ixcell < nxcells; ixcell++) {
	    long off = cell_offsets[iycell*nxcells + ixcell];
	    if (off < 0)
		continue;

	    long iy0 = iycell * 64;
	    long ix0 = ixcell * 64;
	    long ny = min(nypix_global, iy0+64) - iy0;
	    long nx = min(nxpix_global, ix0+64) - ix0;

	    T *pdst = dst.data + (iy0 * nxpix_global) + ix0;
	    const T *psrc = src.data + off;

	    for (int p = 0; p < 3; p++)
		for (long iy = 0; iy < ny; iy++)
		    for (long ix = 0; ix < nx; ix++)
			pdst[p*nypix_global*nxpix_global + iy*nxpix_global + ix] = psrc[p*polstride + iy*ystride + ix];
	}
    }
}


#define INSTANTIATE(T) \
    template void local_map_to_global(const LocalPixelization &local_pixelization, Array<T> &dst, const Array<T> &src);

INSTANTIATE(float);
INSTANTIATE(double);


}  // namespace gpu_mm
