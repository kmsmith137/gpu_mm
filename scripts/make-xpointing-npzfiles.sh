#!/bin/bash
set -e
set -x

SDIR=/home/sigurdkn/gpu_mapmaker/tods_full/
DDIR=$HOME/xpointing

mkdir -p $DDIR

./make-xpointing-npzfile.py $SDIR/tod_1507610629.1507621563.ar5_f090.npz $DDIR/xpointing_0.npz
./make-xpointing-npzfile.py $SDIR/tod_1507611287.1507621748.ar5_f090.npz $DDIR/xpointing_1.npz
