from gpu_mm import gpu_mm

import numpy as np
import cupy as cp
import time

# This script has two steps.
#
#  Step 1. Run two version of map2tod() and compare the results:
#
#   - Slow CPU single-threaded reference version (gpu_mm.reference_map2tod())
#   - GPU version (gpu_mm.gpu_map2tod())
#
#  Step 2. Run two version of tod2map() and compare the results:
#
#   - Slow CPU single-threaded reference version (gpu_mm.reference_tod2map())
#   - GPU version (gpu_mm.gpu_tod2map())
#
#   - This step also needs to create a "plan" for the GPU tod2map() operation.
#     Currently, plan creation is done on the CPU and is slow. I hope to move
#     plan creation to the GPU soon!
#
# For more info, see comments in this file, or docstrings in the gpu_mm module.
#
# Currently, this script must be run from the gpu_mm/scripts directory.
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
    
    # Both CPU and GPU code currently uses "exploded pointing", which we
    # represent as a float32 array 'xpointing' with shape (3, ndetectors, ntimes).
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
    
    filename = '/data/gpu_map_making/xpointing/xpointing_0.npz'

    print(f'Reading {filename}')
    xpointing_cpu = np.load('/data/gpu_map_making/xpointing/xpointing_0.npz')['xpointing']
    assert xpointing_cpu.ndim == 3
    assert xpointing_cpu.shape[0] == 3

    # WARNING: xpointing files use axis ordering (ra, dec, alpha).
    # Internally, we use (dec, ra, alpha), so we permute axes here.
    xpointing_cpu = np.array([ xpointing_cpu[1], xpointing_cpu[0], xpointing_cpu[2] ])
    xpointing_gpu = cp.asarray(xpointing_cpu)  # copy CPU -> GPU
    
    ndet, nt = xpointing_cpu.shape[1:]
    ndec = analyze_xpointing_axis(xpointing_cpu[0], 'dec')
    nra = analyze_xpointing_axis(xpointing_cpu[1], 'ra')

    #  Step 1. Run two version of map2tod() and compare the results:
    #
    #   - Slow CPU single-threaded reference version (gpu_mm.reference_map2tod())
    #   - GPU version (gpu_mm.gpu_map2tod())

    map_cpu = np.random.uniform(size=(3,ndec,nra), low=-1.0, high=1.0)
    map_cpu = np.array(map_cpu, dtype=np.float32)   # float64 -> float32
    tod_cpu = np.zeros((ndet,nt), dtype=np.float32)

    print('\nCalling gpu_mm.reference_map2tod()')
    gpu_mm.reference_map2tod(tod_cpu, map_cpu, xpointing_cpu)
    print(tod_cpu)

    map_gpu = cp.asarray(map_cpu)       # copy input CPU -> GPU
    tod_gpu = cp.zeros((ndet,nt), dtype=cp.float32)
    
    print('\nCalling gpu_mm.gpu_map2tod()')
    gpu_mm.gpu_map2tod(tod_gpu, map_gpu, xpointing_gpu)
    print('Done')

    tod_gpu_copy = cp.asnumpy(tod_gpu)  # copy result GPU -> CPU
    print(tod_gpu_copy)

    maxdiff = np.max(np.abs(tod_cpu - tod_gpu_copy))
    print(f'\nMax difference = {maxdiff}')

    del map_cpu, tod_cpu, map_gpu, tod_gpu, tod_gpu_copy
    
    
    #  Step 2. Run two version of tod2map() and compare the results:
    #
    #   - Slow CPU single-threaded reference version (gpu_mm.reference_tod2map())
    #   - GPU version (gpu_mm.gpu_tod2map())
    #
    #   - This step also needs to create a "plan" for the GPU tod2map() operation.
    #     Currently, plan creation is done on the CPU and is slow. I hope to move
    #     plan creation to the GPU soon!
    
    tod_cpu = np.random.uniform(size=(ndet,nt), low=-1.0, high=1.0)
    tod_cpu = np.array(tod_cpu, dtype=np.float32)   # float64 -> float32
    map_cpu = np.zeros((3,ndec,nra), dtype=np.float32)

    print('\nCalling gpu_mm.reference_tod2map()')
    gpu_mm.reference_tod2map(map_cpu, tod_cpu, xpointing_cpu)
    print(f'{np.sum(map_cpu)=}')

    print('\nMaking OldPointingPlan (currently done on CPU and slow)')
    t0 = time.time()
    plan = gpu_mm.OldPointingPlan(xpointing_cpu, ndec, nra)
    print(f'Plan creation time = {time.time()-t0} seconds')

    tod_gpu = cp.asarray(tod_cpu)   # copy input CPU -> GPU
    map_gpu = cp.zeros((3,ndec,nra), dtype=np.float32)
    
    print('\nCalling gpu_mm.gpu_tod2map()')
    gpu_mm.gpu_tod2map(map_gpu, tod_gpu, xpointing_gpu, plan)
    map_gpu_copy = cp.asnumpy(map_gpu)   # copy result GPU -> CPU
    print(f'{np.sum(map_gpu_copy)=}')

    maxdiff = np.max(np.abs(map_cpu - map_gpu_copy))
    print(f'\nMax difference = {maxdiff}')

