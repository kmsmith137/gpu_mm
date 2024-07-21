### gpu_mm: tools for CMB map-making on GPUs.

Installation.

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

For documentation, please see:

  - The example script `scripts/gpu_mm_example.py`

  - The long docstring at the top of `src_python/gpu_mm/gpu_mm.py`
    (from python: `import gpu_mm; help(gpu_mm.gpu_mm)`)

  - Docstrings for individual classes/functions.

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
