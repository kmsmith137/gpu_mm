#include "../include/gpu_mm.hpp"

#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


LocalPixelization::LocalPixelization(
    long nypix_global_,
    long nxpix_global_,
    const Array<long> &cell_offsets_,
    long ystride_,
    long polstride_,
    bool periodic_xcoord_)

    : LocalPixelization(nypix_global_,
			nxpix_global_,
			cell_offsets_.to_host(),   // cell_offsets_cpu
			cell_offsets_.to_gpu(),    // cell_offsets_gpu
			ystride_,
			polstride_,
			periodic_xcoord_)
{ }


LocalPixelization::LocalPixelization(
    long nypix_global_,
    long nxpix_global_,
    const Array<long> &cell_offsets_cpu_,
    const Array<long> &cell_offsets_gpu_,
    long ystride_,
    long polstride_,
    bool periodic_xcoord_)
    
    : nypix_global(nypix_global_),
      nxpix_global(nxpix_global_),
      cell_offsets_cpu(cell_offsets_cpu_),
      cell_offsets_gpu(cell_offsets_gpu_),
      ystride(ystride_),
      polstride(polstride_),
      periodic_xcoord(periodic_xcoord_)
{
    check_nypix_global(nypix_global, "LocalPixelization constructor");
    check_nxpix_global(nxpix_global, "LocalPixelization constructor");
    
    check_cell_offsets_and_init_ncells(cell_offsets_cpu, nycells, nxcells, "LocalPixelization constructor", false);  // on_gpu=false
    check_cell_offsets(cell_offsets_gpu, nycells, nxcells, "LocalPixelization constructor", true);   // on_gpu=true

    // Some checks on the strides.
    
    xassert(ystride > 0);
    xassert((ystride % 64) == 0);
    
    xassert(polstride > 0);
    xassert((polstride % 64) == 0);

    bool case1 = (ystride % (3*polstride)) == 0;
    bool case2 = (polstride % (64*ystride)) == 0;
    
    if (!case1 && !case2)
	throw runtime_error("LocalPixelization constructor: expected either ystride to be a multiple of (3*polstride), or polstride to be a multiple of (64*ystride)");

    this->_init_npix("LocalPixelization constructor");
}


void LocalPixelization::_init_npix(const char *where)
{
    // Intialize 'npix', and do a consistency check on the cell_offsets.
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

    if ((npix > 0) && (max_offset != (3*npix-1)))
	throw runtime_error(string(where) + ": something is wrong with the cell_offsets -- the cells do not fit together contiguously");
}


void LocalPixelization::copy_gpu_offsets_to_cpu()
{
    CUDA_CALL(cudaMemcpy(this->cell_offsets_cpu.data, this->cell_offsets_gpu.data, nycells * nxcells * sizeof(long), cudaMemcpyDefault));
    this->_init_npix("LocalPixelization::copy_gpu_offsets_to_cpu()");
}

void LocalPixelization::copy_cpu_offsets_to_gpu()
{
    this->_init_npix("LocalPixelization::copy_cpu_offsets_to_gpu");
    CUDA_CALL(cudaMemcpy(this->cell_offsets_gpu.data, this->cell_offsets_cpu.data, nycells * nxcells * sizeof(long), cudaMemcpyDefault));
}

	
}  // namespace gpu_mm
