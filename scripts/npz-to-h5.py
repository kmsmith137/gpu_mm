#!/usr/bin/env python3

import sys
import h5py
import numpy as np

if len(sys.argv) != 3:
    print('Usage: npz-to-h5.py <infile.npz> <outfile.h5>', file=sys.stderr)
    sys.exit(2)


####################################################################################################


def dtype_is_convertible_to_hdf5(dtype):
    try:
        h5py.h5t.py_create(dtype, logical=1)
    except:
        return False
    return True


print(f'Reading input file {sys.argv[1]}')
infile = np.load(sys.argv[1])

print(f'Opening output file {sys.argv[2]}')

with h5py.File(sys.argv[2], 'w') as outfile:
    for k in infile.keys():
        arr = infile[k]
        dtype_ok = dtype_is_convertible_to_hdf5(arr.dtype)
        line = f'    {k}: shape={arr.shape} dtype={arr.dtype}'

        if not dtype_ok:
            line += '. This dtype is not convertible to hdf5, array will not be written to output file!'

        print(line)

        if dtype_ok:
            outfile.create_dataset(k, data=arr)

print(f'Wrote {sys.argv[2]}')
