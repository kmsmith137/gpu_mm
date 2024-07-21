## gpu_mm: tools for CMB map-making on GPUs.

### Installation

1. Make sure you have cuda, cupy, cufft, cublas, curand, pybind11
installed. This conda environment ("heavycuda") works for me:
```
conda create -c conda-forge -n heavycuda \
         cupy scipy matplotlib pybind11 \
         cuda-nvcc libcublas-dev libcufft-dev libcurand-dev
```

2. Install the `ksgpu` library (https://github.com/kmsmith137/ksgpu).
(This library was previously named `gputils`, but I renamed it since that
name was taken on pypi etc.)

3. Install `gpu_mm` with:

    make install

to install in python site-packages. (Alternately, you can just do `make`,
and the subdirectory `src_python` will behave like an importable python
package.)

### Documentation

Please see:

  - The example script `scripts/gpu_mm_example.py`

  - The long docstring at the top of `src_python/gpu_mm/gpu_mm.py`
    (from within python: `import gpu_mm; help(gpu_mm)`)

  - Docstrings for individual classes/functions.

### TODO list

  - Right now the code is not very well tested! I think testing is my 
    next priority.

  - Currently, nypix_global and nxpix_global must be multiples of 64.
    (There's no longer a good reason for this, and it would be easy to change.)

  - Currently, the number of TOD samples 'nsamp' must be a multiple of 32.
    (I'd like to change this, but it's not totally trivial, and there are a
     few minor issues I'd like to chat about.)

  - Helper functions for converting maps between different pixelizations
    (either local or global, with or without wrapping logic).

  - A DynamicLocalPixelization class which adds map cells on-the-fly,
    as tod2map() gets called sequentially with different TODs. This could
    be used on the first iteration of a map maker to assign a LocalPixelization
    to each GPU.

  - An MPIPixelization class with all-to-all logic for distirbuting/reducing
    maps across GPUs.

  - Kernels should be launched on current cupy stream (I'm currently launching
    on the default cuda stream).

  - Support both float32 and float64.

  - There are still some optimizations I'd like to explore, for making map2tod()
    and tod2map() even faster, but I put this on the back-burner since they're
    pretty fast already.

  - Not currently pip-installable (or conda-installable). This turned out to 
    be a real headache. I put it "on pause" but I plan to go back to it later.

  - New feature I'd like to implement some day: full quaternion-based pointing
    computation on the GPU (rather than computing on the CPU and using an
    interpolator).

  - If decompressing data files on the CPU turns out to be a bottleneck, we
    could probably move this to the GPU (https://developer.nvidia.com/nvcomp).

None of these todo items should be a lot of work individually, but I'm not sure 
how to prioritize.

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
