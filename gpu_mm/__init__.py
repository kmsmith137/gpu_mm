# Note: 'import ksgpu' must precede 'import gpu_mm_pybind11'.
# (This is because 'import ksgpu' pulls in the libraries ksgpu.so and ksgpu_pybind11...so,
# using the "ctypes trick" to make their symbols globally visible.)
import ksgpu

import cupy.cublas
import cupy.cuda.cufft

from .gpu_mm import *
from . import gpu_mm_pybind11
from . import gpu_pointing
from . import gpu_utils
from . import pycufft as cufft
from . import tests

# Not imported by default
# from . import tests_mpi

# Pull in the long toplevel docstring from gpu_mm.py,
# which explains the main data structures.
__doc__ = gpu_mm.__doc__
