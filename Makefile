ARCH =
ARCH += -gencode arch=compute_80,code=sm_80
ARCH += -gencode arch=compute_86,code=sm_86
# ARCH += -gencode arch=compute_89,code=sm_89
# ARCH += -gencode arch=compute_90,code=sm_90

GPUTILS_INCDIR=../gputils/include
GPUTILS_LIBDIR=../gputils/lib

NVCC = nvcc -std=c++17 $(ARCH) -m64 -O3 -I$(GPUTILS_INCDIR) --compiler-options -Wall,-fPIC
SHELL := /bin/bash

.DEFAULT_GOAL: all
.PHONY: all clean .FORCE

HFILES = \
  include/gpu_mm.hpp

OFILES = \
  src_lib/ActPointing.o \
  src_lib/map2tod.o \
  src_lib/tod2map.o \
  src_lib/cnpy.o

XFILES = \
  bin/test-map2tod \
  bin/time-map2tod \
  bin/test-tod2map \
  bin/scratch

LIBFILES = \
  lib/libgpu_mm.a \
  lib/libgpu_mm.so

SRCDIRS = \
  include \
  src_bin \
  src_lib

all: $(LIBFILES) $(XFILES)

# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh; do echo $$f; done; done >$@

clean:
	rm -f $(LIBFILES) source_files.txt *~
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o; done

%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

bin/%: src_bin/%.o lib/libgpu_mm.a
	mkdir -p bin && $(NVCC) -o $@ $^ $(GPUTILS_LIBDIR)/libgputils.a -lz

lib/libgpu_mm.so: $(OFILES)
	@mkdir -p lib
	rm -f $@
	$(NVCC) -shared -o $@ $^

lib/libgpu_mm.a: $(OFILES)
	@mkdir -p lib
	rm -f $@
	ar rcs $@ $^
