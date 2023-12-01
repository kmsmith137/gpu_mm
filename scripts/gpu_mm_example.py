import gpu_mm
import numpy as np
import cupy as cp

# This script runs two version of map2tod() and compares the results:
#
#   - Slow CPU single-threaded reference version (gpu_mm.reference_map2tod())
#   - GPU version (gpu_mm.gpu_map2tod())
#
# For more info, see comments in this file, or docstrings in the gpu_mm module.
#
# Currently, this script must be run from the gpu_mm/scripts directory.
#
# The following procedure works for me:
#
#   # In bashrc: use nvidia cuda-toolkit (12.3), not ubuntu cuda-toolkit (11.5)
#   export PATH=/usr/local/cuda/bin:$PATH
#   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#
#   conda create -n cupy -c conda-forge cupy
#   conda activate cupy
#
#   cd ~/git/gpu_mm
#   make -j
#   cd scripts
#   python gpu_mm_example.py


def analyze_xpointing_axis(t, name):
    """
    Infer map dimension 'ndec' or 'nra' from the entries of the 'xpointing' array.
    See below for usage.
    """
    
    assert t.ndim == 2
    assert(np.min(t) > 0.1)

    nmin = int(np.max(t) + 1.1)
    npad = ((nmin + 63) // 64) * 64
    print(f'n{name} = {npad} (padded from {nmin})')
    assert npad <= 16*1024  # larger value probably indicates a bug somewhere
    return npad


if __name__ == '__main__':
    filename = '/data/gpu_map_making/xpointing/xpointing_0.npz'

    # Both CPU and GPU versions of map2tod() operate on "exploded pointing", which
    # we represent as a float32 array 'xpointing' with shape (3, ndetectors, ntimes).
    #
    # The length-3 axis of the 'xpointing' array is {px_dec, px_ra, alpha}, where:
    #
    #   - px_dec is declination in "pixel units", i.e. in (0, ndec-1).
    #     WARNING: if an array element is outside this range, the kernel
    #     will either return garbage or segfault! (This can be fixed later.)
    #
    #   - px_ra is right ascension in "pixel units", i.e. in (0, ndec-1).
    #     WARNING: if an array element is outside this range, the kernel
    #     will either return garbage or segfault! (This can be fixed later.)
    #
    #   - alpha is detector angle in radians.
    #
    # This script reads its 'xpointing' array from a file 'xpointing_0.npz',
    # which I precomputed using Sigurd's tods_full arrays. (See make-xpointing-*
    # scripts in this directory.)

    print(f'Reading {filename}')
    xpointing_cpu = np.load('/data/gpu_map_making/xpointing/xpointing_0.npz')['xpointing']
    assert xpointing_cpu.ndim == 3
    assert xpointing_cpu.shape[0] == 3

    # WARNING: xpointing files use axis ordering (ra, dec, alpha).
    # Internally, we use (dec, ra, alpha), so we permute axes here.
    xpointing_cpu = np.array([ xpointing_cpu[1], xpointing_cpu[0], xpointing_cpu[2] ])
    
    ndet, nt = xpointing_cpu.shape[1:]
    ndec = analyze_xpointing_axis(xpointing_cpu[0], 'dec')
    nra = analyze_xpointing_axis(xpointing_cpu[1], 'ra')

    map_cpu = np.random.uniform(size=(3,ndec,nra))
    map_cpu = np.array(map_cpu, dtype=np.float32)
    tod_cpu = np.zeros((ndet,nt), dtype=np.float32)

    print('\nCalling gpu_mm.reference_map2tod()')
    gpu_mm.reference_map2tod(tod_cpu, map_cpu, xpointing_cpu)
    print(tod_cpu)

    xpointing_gpu = cp.asarray(xpointing_cpu)  # copy CPU -> GPU
    map_gpu = cp.asarray(map_cpu)              # copy CPU -> GPU
    tod_gpu = cp.zeros((ndet,nt), dtype=cp.float32)
    
    print('\nCalling gpu_mm.gpu_map2tod()')
    gpu_mm.gpu_map2tod(tod_gpu, map_gpu, xpointing_gpu)
    print('Done')

    tod_gpu_copy = cp.asnumpy(tod_gpu)  # copy result GPU -> CPU
    print(tod_gpu_copy)

    maxdiff = np.max(np.abs(tod_cpu - tod_gpu_copy))
    print(f'\nMax difference = {maxdiff}')
