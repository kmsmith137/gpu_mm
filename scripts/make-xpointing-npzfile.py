#!/usr/bin/env python

import sys
import numpy as np


def pixelize(arr, name):
    eps = 0.5 * (np.pi/180./60.)   # 0.5 arcmin
    amin = np.min(arr) - 0.5 * eps
    amax = np.max(arr) + 0.5 * eps
    print(f'    {name}: min angle = {amin} radians, max angle = {amax} radians')
    
    npix_min = int((amax-amin)/eps + 2)
    print(f'    {name}: {npix_min=}')
    
    px_arr = (arr-amin)/eps
    px_min, px_max = np.min(px_arr), np.max(px_arr)
    print(f'    {name}: {px_min=} {px_max=}')
    
    assert np.all(px_min > 0.25)
    assert np.all(px_max < npix_min-1.25)
    return px_arr


####################################################################################################


if len(sys.argv) != 3:
    print('usage: make-pointing-npzfile.py <infile.npz> <outfile.npz>', file=sys.stderr)
    sys.exit(2)

print(f'Reading {sys.argv[1]}')
npz = np.load(sys.argv[1])
pointing = npz['pointing']
print(f'    {pointing.shape=}')
assert len(pointing.shape)==3 and pointing.shape[0]==3
    
# Note: in pointing files, the length-3 axis is ordered { ra, dec, alpha }.
# (Internally in C++/cuda code, we usually re-order to { dec, ra, alpha }.)
ra, dec, alpha = pointing

mean_xy = np.mean(np.cos(ra)), np.mean(np.sin(ra))
mean_ra = np.arctan2(mean_xy[1], mean_xy[0])
print(f'    {mean_ra=}')

# Shift RA so that mean=pi
ra = np.fmod(ra - mean_ra + 5*np.pi, 2*np.pi)

px_ra = pixelize(ra, "ra")
px_dec = pixelize(dec, "dec")

# Print some diagnostic info
ra_step = px_ra[:,1:] - px_ra[:,:-1]
dec_step = px_dec[:,1:] - px_dec[:,:-1]
tot_step = np.sqrt(ra_step**2 + dec_step**2)
print(f'    mean scan speed = {np.mean(tot_step)} pixels/sample')
print(f'    max scan jump = {np.max(tot_step)} pixels')

# Note: in pointing files, the length-3 axis is ordered { ra, dec, alpha }.
# (Internally in C++/cuda code, we usually re-order to { dec, ra, alpha }.)
print(f'Writing {sys.argv[2]}')
np.savez(sys.argv[2], xpointing = np.array([px_ra, px_dec, alpha]))
