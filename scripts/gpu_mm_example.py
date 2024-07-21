import gpu_mm
import numpy as np
import cupy as cp
import time


if __name__ == '__main__':

    # Read xpointing (="exploded pointing") array.
    # For more info, see 'xpointing' in the toplevel gpu_mm docstring.
    #
    # This script reads its 'xpointing' array from a file 'xpointing_0.npz',
    # which I precomputed using Sigurd's tods_full arrays. (See make-xpointing-*
    # scripts in this directory.)
    #
    # Note that storing xpointing arrays in npz files is a kludge that 
    # I'm hoping to phase out soon.
    
    filename = '/data/gpu_map_making/xpointing/xpointing_0.npz'
    xpointing_cpu = gpu_mm.read_xpointing_npzfile(filename)
    xpointing_gpu = cp.asarray(xpointing_cpu)  # copy CPU -> GPU

    y, x, alpha = xpointing_cpu         # axis 0 is {y,x,alpha}
    ndet, nt = xpointing_cpu.shape[1:]  # axes 1,2 are det,time
    
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    print(f'{ymin=} {ymax=} {xmin=} {xmax=}')

    # Define global pixelization. (For more info, see 'global maps' in
    # toplevel gpu_mm docstring.) This part is a little ad hoc!

    nypix_global = 64*int((ymax-3)/64.) + 64   # use enough y-pixels to cover the xpointing with a little padding
    nxpix_global = 360*60*2                    # ACT maps use 0.5 arcminute pixels
    periodic_xcoord = True

    # Define local pixelization. (For more info, see 'local maps' in
    # toplevel gpu_mm docstring.)
    #
    # For simplicity, we use a trivial local pixelization which covers
    # the entire sky. (Doing something fancier here would save GPU memory.)

    local_pix = gpu_mm.LocalPixelization.make_rectangle(nypix_global, nxpix_global, periodic_xcoord)

    # Make GPU pointing "plan". (For more info, see 'pointing plans' in
    # toplevel gpu_mm docstring, including info on how to speed up
    # plan creation with preallocated buffers.)

    preplan = gpu_mm.PointingPrePlan(xpointing_gpu, nypix_global, nxpix_global, periodic_xcoord=periodic_xcoord)
    plan = gpu_mm.PointingPlan(preplan, xpointing_gpu)

    #  Step 1. Run two version of map2tod() and compare the results:
    #
    #   - Slow CPU single-threaded reference version (plan='reference')
    #   - GPU version

    map_cpu = np.random.uniform(size=(3,nypix_global,nxpix_global), low=-1.0, high=1.0)
    map_cpu = np.array(map_cpu, dtype=np.float32)   # float64 -> float32
    tod_cpu = np.zeros((ndet,nt), dtype=np.float32)
    map_gpu = cp.asarray(map_cpu)
    tod_gpu = cp.zeros((ndet, nt), dtype=cp.float32)
    
    print('\nCalling CPU-based reference map2tod()')
    local_map = gpu_mm.LocalMap(local_pix, map_cpu)  # local map on CPU
    gpu_mm.map2tod(tod_cpu, local_map, xpointing_cpu, plan='reference')
    print('Done')

    print('\nCalling GPU-based map2tod()')
    local_map = gpu_mm.LocalMap(local_pix, map_gpu)  # local map on GPU
    t0 = time.time()
    gpu_mm.map2tod(tod_gpu, local_map, xpointing_gpu, plan=plan)
    dt = time.time()-t0
    print(f'Done [{1000*dt} milliseconds]')

    # Compare results
    tod_gpu_copy = cp.asnumpy(tod_gpu)  # copy result GPU -> CPU
    maxdiff = np.max(np.abs(tod_cpu - tod_gpu_copy))
    print(f'\nMax difference = {maxdiff}')

    del map_cpu, tod_cpu, map_gpu, tod_gpu, tod_gpu_copy
    
    #  Step 2. Run two version of tod2map() and compare the results:
    #
    #   - Slow CPU single-threaded reference version (plan='reference')
    #   - GPU version
    
    tod_cpu = np.random.uniform(size=(ndet,nt), low=-1.0, high=1.0)
    tod_cpu = np.array(tod_cpu, dtype=np.float32)   # float64 -> float32
    map_cpu = np.zeros((3,nypix_global,nxpix_global), dtype=np.float32)
    tod_gpu = cp.asarray(tod_cpu)   # copy input CPU -> GPU
    map_gpu = cp.zeros((3,nypix_global,nxpix_global), dtype=np.float32)

    print('\nCalling CPU-based reference tod2map()')
    local_map = gpu_mm.LocalMap(local_pix, map_cpu)  # local map on CPU
    gpu_mm.tod2map(local_map, tod_cpu, xpointing_cpu, plan='reference')
    print('Done\n')
    
    print('\nCalling GPU tod2map()')
    local_map = gpu_mm.LocalMap(local_pix, map_gpu)  # local map on GPU
    t0 = time.time()
    gpu_mm.tod2map(local_map, tod_gpu, xpointing_gpu, plan)
    dt = time.time()-t0
    print(f'Done [{1000*dt} milliseconds]')

    map_gpu_copy = cp.asnumpy(map_gpu)  # copy result GPU -> CPU
    maxdiff = np.max(np.abs(map_cpu - map_gpu_copy))
    print(f'\nMax difference = {maxdiff}')

