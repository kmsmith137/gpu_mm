#include "../include/gpu_mm.hpp"

#include <cassert>
#include <ksgpu/cuda_utils.hpp>

using namespace ksgpu;

namespace gpu_mm {
#if 0
}   // pacify editor auto-indent
#endif


// Template parameter N is the number of 'plan_mt' values per threadblock.
// Use 1-d warp grid {32*W,1,1}.
// Use 1-d block grid {B,1,1}.
    
template<uint N>
__global__ void dynamic_map_expander(
    uint *global_ncells,   // scalar, updated atomically by all threadblocks
    long *cell_offsets,    // output array
    uint nycells,
    uint nxcells,
    const ulong *plan_mt,  // input array
    uint nmt)
{
    // Shared memory layout:
    //  uint icell_in[N+1];      // note +1 here
    //  uint local_ncells = 0;   // updated atomically
    //  uint global_ncells = 0;  // if needed
    //  uint pad[29];            // always zero
    //  uint icell_out[N];
    
    __shared__ uint shmem[2*N+32];

    // Global -> shared.
    for (int i = threadIdx.x; i < N+32; i += blockDim.x) {
	uint imt0 = (blockIdx.x * N) + i;
	uint imt1 = (imt0 < nmt) ? imt0 : (nmt-1);
	ulong mt = plan_mt[imt1];

	// Reminder: mt bit layout is
	//   Low 10 bits = Global xcell index
	//   Next 10 bits = Global ycell index
	//     ...
	
	uint icell = uint(mt) & ((1U << 20) - 1);
	shmem[i] = (i < N+1) ? icell : 0;   // write zero to 'local_ncells' and 'pad'.
    }

    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
	uint imt = (blockIdx.x * N) + i;
	
	uint icell = shmem[i];
	uint icell_next = shmem[i+1];
	uint ixcell = icell & ((1U << 10) - 1);;
	uint iycell = icell >> 10;
	uint ioff = iycell*nxcells + ixcell;  // index in 'cell_offsets' table.

	bool candidate = (imt < nmt) && ((icell != icell_next) || (imt == (nmt-1)));
	candidate = candidate && (ixcell < nxcells) && (iycell < nycells);

	if (candidate) {
	    // Check 'cell_offsets' in global memory.
	    // Note that 'cell_offsets' has special semantics: (-1) is "targeted", (-2) is "untargeted"
	    if (cell_offsets[ioff] == -1) {          // targeted
		uint n = atomicAdd(&shmem[N+2], 1);  // update local_ncells
		shmem[N+32+n] = icell;
	    }
	}

	__syncwarp();
    }

    __syncthreads();

    // The value of 'local_ncells' is the same on all threads.
    uint local_ncells = shmem[N+2];

    if (local_ncells == 0)
	return;

    // On thread 0, increment 'global_ncells', and write its value to shared memory.
    if (threadIdx.x == 0)
	shmem[N+3] = atomicAdd(global_ncells, local_ncells);

    __syncthreads();

    // Value of 'nglo' is the same on all threads.
    uint nglo = shmem[N+3];

    for (int i = threadIdx.x; i < local_ncells; i += blockDim.x) {
	uint icell = shmem[N+32+i];
	uint ixcell = icell & ((1U << 10) - 1);;
	uint iycell = icell >> 10;
	uint ioff = iycell*nxcells + ixcell;  // index in 'cell_offsets' table.
	// assert(ioff < nxcells*nycells);
	cell_offsets[ioff] = long(nglo+i) * long(3*64*64);
    }
}


// Returns updated value of 'global_ncells'.
uint expand_dynamic_map(
    ksgpu::Array<uint> &global_ncells,        // shape (1,) on GPU
    ksgpu::Array<long> &cell_offsets,         // shape (nycells, nxcells) on GPU
    const ksgpu::Array<ulong> &plan_mt)       // shape (nmt,) on GPU
{
    constexpr uint N = 1024 - 32;   // number of 'plan_mt' values per threadblock
    constexpr uint W = 4;           // warps per threadblock
    
    xassert(global_ncells.on_gpu());
    xassert(global_ncells.shape_equals({1}));
    
    long nycells, nxcells;
    check_cell_offsets_and_init_ncells(cell_offsets, nycells, nxcells, "launch_dynamic_map_expander", true);  // on_gpu=true

    xassert(plan_mt.on_gpu());
    xassert(plan_mt.ndim == 1);
    xassert(plan_mt.is_fully_contiguous());

    uint nmt = plan_mt.shape[0];
    uint nblocks = (nmt + N - 1) / N;

    dynamic_map_expander<N> <<< nblocks, 32*W >>>
	(global_ncells.data,  // uint *global_ncells
	 cell_offsets.data,   // const ulong *cell_offsets
	 nycells,             // uint nycells
	 nxcells,             // uint nxcells
	 plan_mt.data,        // const ulong *plan_mt
	 nmt);                // uint nmt

    CUDA_PEEK("dynamic_map_expander kernel launch");

    Array<uint> ret({1}, af_rhost | af_zero);
    CUDA_CALL(cudaMemcpy(ret.data, global_ncells.data, sizeof(uint), cudaMemcpyDefault));
    
    return ret.data[0];
}


// FIXME temporary kludge that will go away later.
// Returns updated value of 'global_ncells'.
uint expand_dynamic_map2(
    ksgpu::Array<uint> &global_ncells,        // shape (1,) on GPU
    LocalPixelization &local_pixelization,
    const PointingPlan &plan)
{
    xassert(local_pixelization.nypix_global == plan.nypix_global);
    xassert(local_pixelization.nxpix_global == plan.nxpix_global);
    xassert(local_pixelization.periodic_xcoord == plan.periodic_xcoord);
    
    return expand_dynamic_map(
        global_ncells,
	local_pixelization.cell_offsets_gpu,
	plan.get_plan_mt(true)   // on_gpu=true
    );
}


}  // namespace gpu_mm
