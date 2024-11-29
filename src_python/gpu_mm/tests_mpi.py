import os
import time
import functools
import cupy as cp
import numpy as np

from . import gpu_mm
from . import tests

from mpi4py import MPI

comm = MPI.COMM_WORLD
ntasks = comm.Get_size()
my_rank = comm.Get_rank()

        
def test_mpi_pixelization(niter=100):
    for _ in range(niter):
        nypix_global, nycells = tests.make_random_npix(32)
        nxpix_global, nxcells = tests.make_random_npix(32)
        periodic_xcoord = False    # FIXME!!!!
        dtype = np.float32         # placeholder for future expansion
    
        t = nypix_global, nxpix_global, periodic_xcoord, nycells, nxcells
        nypix_global, nxpix_global, periodic_xcoord, nycells, nxcells = comm.bcast(t)   # bcast() not Bcast()!
        
        # FIXME temporary kludge needed for random DynamicPixelization below
        if (nypix_global % 64) == 1:
            nypix_global += 1
        if (nxpix_global % 64) == 1:
            nxpix_global += 1

        if my_rank == 0:
            print(f'test_mpi_pixelization(): {nypix_global=} {nxpix_global=} {periodic_xcoord=} {nycells=} {nxcells=}')
        
        # Make a random DynamicPixelization
        # FIXME improve this code and put it somewhere more general
        # FIXME test case where ncells_hit==0?

        hit_prob = np.linspace(0.0, 1.0, nycells*nxcells)
        hit_prob = np.reshape(hit_prob, (nycells,nxcells))
        cells_2d = (np.random.uniform(size=(nycells,nxcells)) < hit_prob)
        cells_2d[np.random.randint(0,nycells), np.random.randint(0,nxcells)] = True
        if np.random.uniform() < 0.1/ntasks:
            cells_2d[:,:] = True

        cells_y, cells_x = np.where(cells_2d)
        ncells_hit = len(cells_y)
        perm = np.random.permutation(ncells_hit)
        cells_y, cells_x = cells_y[perm], cells_x[perm]

        xpointing = np.zeros((3,ncells_hit,32), dtype=np.float32)
        xpointing[0,:,:] = np.reshape(64*cells_y + 0.5, (-1,1))
        xpointing[1,:,:] = np.reshape(64*cells_x + 0.5, (-1,1))
        xpointing = cp.asarray(xpointing)  # CPU -> GPU
        preplan = gpu_mm.PointingPrePlan(xpointing, nypix_global, nxpix_global, periodic_xcoord=periodic_xcoord)
        plan = gpu_mm.PointingPlan(preplan, xpointing)
        dmap = gpu_mm.DynamicMap(nypix_global, nxpix_global, cell_mask=np.ones((nycells,nxcells),dtype=bool), dtype=dtype, periodic_xcoord=periodic_xcoord)
        dmap.expand(plan)
        
        dmap._unstable_pixelization.copy_gpu_offsets_to_cpu()
        assert np.all( (dmap._unstable_pixelization.cell_offsets_cpu >= 0) == cells_2d)
        
        mpix = gpu_mm.MpiPixelization(dmap, noisy=False)
        n = 3*64*64
        
        # Test 1. check that (mpix.aux_counts) and (mpix.wbuf_counts) are consistent.
        all_aux = mpix._allgather(mpix.aux_counts)
        all_wbuf = mpix._allgather(mpix.wbuf_counts)
        assert np.all(all_aux == np.transpose(all_wbuf))

        # Test 2. broadcast() a "toplevel" test map, and verify that the result is a "working" test map.
        tm_t = tests.make_test_map(mpix.toplevel_pixelization, dtype)
        tm_w = tests.make_test_map(mpix.working_pixelization, dtype)
        tm_b = mpix.broadcast(tm_t)
        assert np.all(tm_b.arr == tm_w.arr)

        # Test 3. gather() a "toplevel" test map, and verify that the result is a "root" test map.
        tm_g = mpix.gather(tm_t)

        if (mpix.my_rank == 0):
            tm_r = tests.make_test_map(mpix.root_pixelization, dtype)
            assert np.all(tm_g.arr == tm_r.arr)
        
        
        # Test 4. verify that broadcast() and reduce() are adjoints, in the formal linear algebra sense.
        
        v = tests.make_random_map(mpix.toplevel_pixelization, dtype)
        w = tests.make_random_map(mpix.working_pixelization, dtype)
        Bv = mpix.broadcast(v)
        Rw = mpix.reduce(w)

        dot_w_Bv, dot_Rw_v, dot_v_v, dot_w_w, dot_Bv_Bv, dot_Rw_Rw = mpix._allreduce([
            np.dot(w.arr,Bv.arr), np.dot(Rw.arr,v.arr), np.dot(v.arr,v.arr),
            np.dot(w.arr,w.arr), np.dot(Bv.arr,Bv.arr), np.dot(Rw.arr,Rw.arr)
        ])
        
        eps = np.abs(dot_w_Bv - dot_Rw_v) / (dot_v_v * dot_w_w * dot_Bv_Bv * dot_Rw_Rw)**0.25
        assert eps < 1.0e-7   # FIXME use dtype-dependent threshold
