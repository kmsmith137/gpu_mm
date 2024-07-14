#include "../include/gpu_mm.hpp"
#include <gputils/xassert.hpp>

using namespace std;
using namespace gputils;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


LocalPixelization::LocalPixelization(const Array<long> &cell_offsets, long ystride_, long polstride_)
{
    xassert(cell_offsets.ndim == 2);
    xassert(cell_offsets.is_fully_contiguous());
    
    this->ystride = ystride_;
    this->polstride = polstride_;
    this->nycells = cell_offsets.shape[0];
    this->nxcells = cell_offsets.shape[1];

    xassert(nycells > 0);
    xassert(nxcells > 0);
    xassert_le(nycells, 1024);
    xassert_le(nxcells, 1024);

    // Some checks on the strides.
    
    xassert(ystride > 0);
    xassert((ystride % 64) == 0);
    
    xassert(polstride > 0);
    xassert((polstride % 64) == 0);

    bool case1 = (ystride % (3*polstride)) == 0;
    bool case2 = (polstride % (64*ystride)) == 0;
    
    if (!case1 && !case2)
	throw runtime_error("LocalPixelization constructor: expected either ystride to be a multiple of (3*polstride), or polstride to be a multiple of (64*ystride)");

    this->cell_offsets_cpu = cell_offsets.to_host();
    this->cell_offsets_gpu = cell_offsets.to_gpu();
    
    // The rest of the constructor intiializes npix, and does a consistency check on the cell_offsets.
    // This isn't a complete consistency check, but it should catch mis-specified cell_offsets in practice.
    // FIXME some day, I may strengthen this check.

    long max_offset = 0;
    this->npix = 0;
    
    for (int iycell = 0; iycell < nycells; iycell++) {
	for (int ixcell = 0; ixcell < nxcells; ixcell++) {
	    long cell_offset = cell_offsets_cpu.data[iycell*nxcells + ixcell];
	    if (cell_offset < 0)
		continue;

	    xassert((cell_offset % 64) == 0);
	    max_offset = max(cell_offset, max_offset);
	    this->npix += 64*64;
	}
    }

    max_offset += 2*polstride;
    max_offset += 63*ystride;
    max_offset += 63;

    if (max_offset != (3*npix-1))
	throw runtime_error("LocalPixelization constructor: something is wrong with the cell_offsets -- the cells do not fit together contiguously");
}


}  // namespace gpu_mm
