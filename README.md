Here's one way to install:
```
conda create -n gpu_mm -f conda_env.yaml
# Or: conda create -c conda-forge -n gpu_mm cupy scipy matplotlib meson-python pybind11

conda activate gpu_mm

# Compile gputils library (note -b flag to select 'python' branch)
git clone https://github.com/kmsmith137/gputils -b python   
cd gputils
pip install --no-cache-dir --no-build-isolation -v .   # note weird pip flags
cd ..


# Compile gpu_mm library (note -b flag to select 'pip' branch)
git clone https://github.com/kmsmith137/gpu_mm -b pip
cd gpu_mm
pip install --no-cache-dir --no-build-isolation -v .    # note weird pip flags
cd ..
```
