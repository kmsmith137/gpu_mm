project(
    'gpu_mm',
    'cuda', 'cpp',
     default_options: ['python.install_env=auto','optimization=3','debug=false','b_ndebug=false','cpp_std=c++17']
)

header_files_toplevel = [
    'include/gpu_mm.hpp',
    'include/gpu_mm_internals.hpp',
    'include/plan_iterator.hpp'
]

lib_source_files = [
    'src_lib/LocalPixelization.cu',
    'src_lib/OldPointingPlan.cu',
    'src_lib/PointingPlan.cu',
    'src_lib/PointingPlanTester.cu',
    'src_lib/PointingPrePlan.cu',
    'src_lib/ToyPointing.cu',
    'src_lib/check_arguments.cu',
    'src_lib/cuts.cu',
    'src_lib/gpu_point.cu',
    'src_lib/gpu_utils.cu',
    'src_lib/map2tod.cu',
    'src_lib/map2tod_old.cu',
    'src_lib/map2tod_reference.cu',
    'src_lib/map2tod_unplanned.cu',
    'src_lib/misc.cu',
    'src_lib/pycufft.cu',
    'src_lib/test_plan_iterator.cu',
    'src_lib/tod2map.cu',
    'src_lib/tod2map_old.cu',
    'src_lib/tod2map_reference.cu',
    'src_lib/tod2map_unplanned.cu'
]

pyext_source_files = [
     'src_pybind11/gpu_mm_pybind11.cu'
]

python_files_toplevel = [
    'python/__init__.py',
    'python/__main__.py',
    'python/gpu_mm.py',
    'python/gpu_pointing.py',
    'python/gpu_utils.py',
    'python/pycufft.py',
    'python/tests.py'
]    

# In src_bin/
# local_executables = [ ]


####################################################################################################


fs = import('fs')
cpp = meson.get_compiler('cpp')
nvcc = meson.get_compiler('cuda')

cuda_args = ['-std=c++17', '-m64']
cuda_args += ['-gencode','arch=compute_80,code=sm_80']
cuda_args += ['-gencode','arch=compute_86,code=sm_86']
cuda_args += ['-gencode','arch=compute_89,code=sm_89']
link_args = ['-m64']

py3 = import('python').find_installation('python3', pure: false)
py3_install_dir = py3.get_install_dir()
# message('py3.path = ', py3.path())
# message('py3.install_dir = ', py3_install_dir)

# Show sys.path for debugging
# ret = run_command(py3.path(), '-c', 'import sys; print(sys.path)', check: true)
# message('Debug: py3.sys.path = ', ret.stdout().strip())


####################################################################################################


# Find cublas and cufft libraries.
#
# Note: nvcc.find_library(...) may find a different version of cublas/cufft than the
# one used by cupy. So far this has been harmless, but it seems like a potential source
# of problems. (There's no good general solution -- one possible solution is to build in
# a conda env which includes cupy, cuda-nvcc, libcublas-dev, libcufft-dev.)
#
# If you want to see which version of cufft (say) is used by cupy, here's one way to do it:
#
#   LD_DEBUG=libs python -c 'import cupy.cuda.cufft' 2>libs_cufft.txt
#   grep libcufft.so libs_cufft.txt


cublas = nvcc.find_library('cublas')   # no 'dirs' argument needed (I think)
cufft = nvcc.find_library('cufft')


####################################################################################################


ksgpu_ret = run_command(py3.path(), '-c', 'import ksgpu; print(ksgpu.__file__)', check: true)
ksgpu_ret = ksgpu_ret.stdout().strip()
# message('ksgpu_ret = ', ksgpu_ret)

ksgpu_incdir = fs.parent(ksgpu_ret) / 'include'
ksgpu_libdir = fs.parent(ksgpu_ret)

if not fs.exists(ksgpu_incdir / 'ksgpu.hpp')
    error('Couldn\'t find ksgpu.hpp in expected location -- time to debug')
endif

if not fs.exists(ksgpu_libdir / 'libksgpu.so')
    error('Couldn\'t find libksgpu.so in expected location -- time to debug')
endif

ksgpu_pyext_ret = run_command(py3.path(), '-c', 'import ksgpu; print(ksgpu.ksgpu_pybind11.__file__)', check: true)
ksgpu_pyext_ret = ksgpu_pyext_ret.stdout().strip()
# message('ksgpu_pyext_ret = ', ksgpu_pyext_ret)

ksgpu_pyext_dir = fs.parent(ksgpu_pyext_ret)
ksgpu_pyext_basename = fs.name(ksgpu_pyext_ret)
# message('ksgpu_pyext_dir = ', ksgpu_pyext_dir)
# message('ksgpu_pyext_basename =', ksgpu_pyext_basename)

if not ksgpu_pyext_basename.endswith('.so')
    error('ksgpu.ksgpu_pybind11.__file__ did not end with .so as expected -- time to debug')
endif

ksgpu_pyext_stem = ksgpu_pyext_basename.substring(0,-3)
# message('ksgpu_pyext_stem =', ksgpu_pyext_stem)

# FIXME using cpp compiler here, since using cuda compiler doesn't seem to work!
ksgpu_lib = cpp.find_library('ksgpu', dirs: ksgpu_libdir)
ksgpu_pyext = cpp.find_library(ksgpu_pyext_stem, dirs: ksgpu_libdir)


####################################################################################################


pybind11_ret = run_command(py3.path(), '-m', 'pybind11', '--includes', check: true)
pybind11_ret = pybind11_ret.stdout().strip()
# message('Debug: Output of python -m pybind11 --includes: ', pybind11_ret)

pybind11_include_dirs = [ ]
foreach include_dir: pybind11_ret.split()
    include_dir = include_dir.strip().strip('-I')
    pybind11_include_dirs += include_dir
endforeach

# message('Debug: pybind11_include_dirs = ', pybind11_include_dirs)
pybind11_dep = declare_dependency(include_directories: pybind11_include_dirs)


####################################################################################################


numpy_incdir = run_command(py3.path(), '-c', 'import numpy; print(numpy.get_include())', check: true)
numpy_incdir = numpy_incdir.stdout().strip()

# message('Debug: numpy.get_include() = ', numpy_incdir)
numpy_dep = declare_dependency(include_directories: numpy_incdir)


####################################################################################################


gpu_mm_lib = library(
    'gpu_mm',
    lib_source_files,
    cuda_args: cuda_args,
    link_args: link_args,
    include_directories: ksgpu_incdir,
    dependencies: [ksgpu_lib, cublas, cufft],
    install: true,
    install_dir: py3.get_install_dir() / 'gpu_mm'
)

py3.extension_module(
    'gpu_mm_pybind11',     # compiled extension will be named gpu_mm_pybind11.so
    sources: pyext_source_files,
    dependencies : [pybind11_dep, numpy_dep, ksgpu_pyext, ksgpu_lib],
    include_directories: ksgpu_incdir,
    cuda_args: cuda_args,
    link_args: link_args,
    link_language : 'cuda',
    link_with: gpu_mm_lib,  # link libgpu_mm.so to gpu_mm_pybind11.so
    override_options: ['cpp_rtti=true'],
    subdir: 'gpu_mm',
    install: true
)

py3.install_sources(
    python_files_toplevel,
    subdir: 'gpu_mm'
)

py3.install_sources(
    header_files_toplevel,
    subdir: 'gpu_mm/include'
)

# foreach x: local_executables
#     executable(
#         x,
#         'src_bin/' + x + '.cu',
# 	  dependencies: [ksgpu_lib],
# 	  include_directories: ksgpu_incdir,
#         cuda_args: cuda_args,
#	  link_args: link_args,
# 	  link_with: gpu_mm_lib
#     )
# endforeach
