PYTHON := python3
SHELL := /bin/bash
NVCC = nvcc -std=c++17 $(ARCH) -m64 -O3 --compiler-options -Wall,-fPIC -I$(KSGPU_DIR)/include
NVCC_PYEXT = $(NVCC) $(PYBIND11_INC) $(NUMPY_INC)
CULIBS = -lcufft -lcublas

ARCH = -gencode arch=compute_80,code=sm_80
ARCH += -gencode arch=compute_86,code=sm_86
ARCH += -gencode arch=compute_89,code=sm_89
# ARCH += -gencode arch=compute_90,code=sm_90

.DEFAULT_GOAL: all
.PHONY: all clean install .FORCE


PY_INSTALL_DIR := $(shell $(PYTHON) choose_python_install_dir.py)
ifneq ($(.SHELLSTATUS),0)
  $(error choose_python_install_dir.py failed!)
endif

# Note: 'pybind11 --includes' includes the base python dir (e.g. -I/xxx/include/python3.12) in addition to pybind11
PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)
ifneq ($(.SHELLSTATUS),0)
  $(error 'pybind11 --includes' failed. Maybe pybind11 is not installed?)
endif

NUMPY_INC := -I$(shell $(PYTHON) -c 'import numpy; print(numpy.get_include())')
ifneq ($(.SHELLSTATUS),0)
  $(error 'numpy.get_include() failed'. Maybe numpy is not installed?)
endif

KSGPU_PYEXT := $(shell $(PYTHON) -c 'import ksgpu.ksgpu_pybind11 as kp; print(kp.__file__)')
ifneq ($(.SHELLSTATUS),0)
  $(error 'import ksgpu.ksgpu_pybind11 failed'. Maybe ksgpu is not installed?)
endif

KSGPU_DIR := $(shell dirname $(KSGPU_PYEXT))
ifneq ($(.SHELLSTATUS),0)
  $(error dirname $(KSGPU_PYEXT) failed)
endif


HFILES = \
  include/gpu_mm.hpp \
  include/gpu_mm_internals.hpp \
  include/plan_iterator.hpp

OFILES = \
  src_lib/LocalPixelization.o \
  src_lib/PointingPlan.o \
  src_lib/PointingPlanTester.o \
  src_lib/PointingPrePlan.o \
  src_lib/ToyPointing.o \
  src_lib/check_arguments.o \
  src_lib/cuts.o \
  src_lib/expand_dynamic_map.o \
  src_lib/gpu_point.o \
  src_lib/gpu_utils.o \
  src_lib/local_map_to_global.o \
  src_lib/map2tod.o \
  src_lib/map2tod_reference.o \
  src_lib/map2tod_unplanned.o \
  src_lib/misc.o \
  src_lib/pycufft.o \
  src_lib/test_plan_iterator.o \
  src_lib/tod2map.o \
  src_lib/tod2map_reference.o \
  src_lib/tod2map_unplanned.o

# FIXME instead of gpu_mm_pybind11.o, should I be using a name like gpu_mm_pybind11.cpython-312-x86_64-linux-gnu.so?
PYOFILES = \
  src_pybind11/gpu_mm_pybind11.o

LIBFILES = \
  lib/libgpu_mm.so \
  lib/libgpu_mm.a

PYEXTFILES = \
  src_python/gpu_mm/gpu_mm_pybind11.so \
  src_python/gpu_mm/libgpu_mm.so   # just a copy of lib/libgpu_mm.so

PYFILES = \
  src_python/gpu_mm/__init__.py \
  src_python/gpu_mm/__main__.py \
  src_python/gpu_mm/gpu_mm.py \
  src_python/gpu_mm/gpu_pointing.py \
  src_python/gpu_mm/gpu_utils.py \
  src_python/gpu_mm/pycufft.py \
  src_python/gpu_mm/tests.py \
  src_python/gpu_mm/tests_mpi.py

SRCDIRS = \
  include \
  src_lib \
  src_pybind11 \
  src_python \
  src_python/gpu_mm \

all: $(LIBFILES) $(PYOFILES) $(PYEXTFILES)

%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

src_pybind11/%.o: src_pybind11/%.cu $(HFILES)
	$(NVCC_PYEXT) -c -o $@ $<

bin/%: src_bin/%.o lib/libgpu_mm.so
	mkdir -p bin && $(NVCC) -o $@ $^

lib/libgpu_mm.so: $(OFILES)
	@mkdir -p lib
	rm -f $@
	$(NVCC) -shared -o $@ $^ $(CULIBS)

lib/libgpu_mm.a: $(OFILES)
	@mkdir -p lib
	rm -f $@
	ar rcs $@ $^

src_python/gpu_mm/libgpu_mm.so: lib/libgpu_mm.so
	cp -f $< $@ 

# Check out the obnoxious level of quoting needed around $ORIGIN!
src_python/gpu_mm/gpu_mm_pybind11.so: $(PYOFILES) src_python/gpu_mm/libgpu_mm.so
	cd src_python/gpu_mm && $(NVCC) '-Xcompiler=-Wl\,-rpath\,'"'"'$$ORIGIN/'"'"'' -shared -o gpu_mm_pybind11.so $(addprefix ../../,$(PYOFILES)) libgpu_mm.so $(KSGPU_PYEXT) $(KSGPU_DIR)/libksgpu.so $(CULIBS)

install: src_python/gpu_mm/gpu_mm_pybind11.so src_python/gpu_mm/libgpu_mm.so
	@mkdir -p $(PY_INSTALL_DIR)/gpu_mm
	cp -f $^ $(PYFILES) $(PY_INSTALL_DIR)/gpu_mm

# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh $$d/*.py; do echo $$f; done; done >$@

clean:
	rm -f $(LIBFILES) $(PYEXTFILES) source_files.txt *~
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o; done

# INSTALL_DIR ?= /usr/local
#
# install: $(LIBFILES)
# 	mkdir -p $(INSTALL_DIR)/include
#	mkdir -p $(INSTALL_DIR)/lib
#	cp -rv lib $(INSTALL_DIR)/
#	cp -rv include $(INSTALL_DIR)/
