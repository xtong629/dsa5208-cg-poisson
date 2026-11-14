# Makefile -- serial / mpi / hybrid / cuda builds share the headers in src/common.
# Local (Mac) and cluster settings are overridable on the command line, e.g.
#   make serial CXX=g++            make mpi MPICXX=mpicxx
CXX      ?= c++
MPICXX   ?= mpicxx
OPT      ?= -O3
CXXFLAGS ?= $(OPT) -std=c++17 -Wall -Wextra
BUILD    := build

# macOS only: the Command Line Tools on some machines ship a stale libc++ header dir that is
# searched before the SDK's, so <cmath> is not found. Put the SDK's libc++ headers first.
ifeq ($(shell uname -s),Darwin)
SDKROOT_PATH := $(shell xcrun --show-sdk-path 2>/dev/null)
CXXFLAGS += -isystem $(SDKROOT_PATH)/usr/include/c++/v1
endif

# OpenMP flags for the hybrid (MPI+OpenMP) build. Override with OMPFLAGS=... if your compiler
# differs (e.g. nvc++: OMPFLAGS=-mp). On macOS Apple clang has no built-in OpenMP: use Homebrew's
# libomp (brew install libomp).
ifeq ($(shell uname -s),Darwin)
LIBOMP   ?= $(shell brew --prefix libomp 2>/dev/null)
OMPFLAGS ?= -Xpreprocessor -fopenmp -I$(LIBOMP)/include -L$(LIBOMP)/lib -lomp -Wl,-rpath,$(LIBOMP)/lib
else
OMPFLAGS ?= -fopenmp
endif

.PHONY: all serial mpi hybrid cuda cuda-alt cuda-check test clean
all: serial mpi hybrid

serial: $(BUILD)/serial
$(BUILD)/serial: src/serial_main.cpp $(wildcard src/common/*.hpp) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $<

mpi: $(BUILD)/mpi
$(BUILD)/mpi: src/mpi_main.cpp $(wildcard src/common/*.hpp src/mpi/*.hpp) | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI -Isrc -o $@ $<

hybrid: $(BUILD)/hybrid
$(BUILD)/hybrid: src/mpi_main.cpp $(wildcard src/common/*.hpp src/mpi/*.hpp) | $(BUILD)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -DUSE_MPI -DUSE_OPENMP -Isrc -o $@ $<

# ---- CUDA (build on Vanda inside an interactive job after loading the modules) ----
# nvcc uses the MPI compiler wrapper as host compiler, so MPI include/link flags come for free.
#   make cuda                       (A40 = sm_86)
#   make cuda CUDA_ARCH=sm_80       other GPU;  make cuda NVCC=/path/to/nvcc
#   make cuda CUDA_EXTRA=-DHOST_STAGING   default to staging through pinned host buffers
NVCC       ?= nvcc
CUDA_ARCH  ?= sm_86
CUDA_EXTRA ?=
CUDA_HOSTCXX ?= $(MPICXX)
CUDA_SRC := $(wildcard src/cuda/*.hpp src/cuda/*.cuh) $(wildcard src/common/*.hpp src/mpi/*.hpp)

cuda: $(BUILD)/cuda
$(BUILD)/cuda: src/cuda_main.cu $(CUDA_SRC) | $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -ccbin $(CUDA_HOSTCXX) -DUSE_MPI -DUSE_CUDA $(CUDA_EXTRA) -Isrc -o $@ $<

# Fallback if `nvcc -ccbin mpicxx` fails (e.g. host-compiler version check, wrapper flags nvcc
# rejects): compile the .cu with nvcc using the SYSTEM g++ as host compiler and only the -I flags
# of the MPI wrapper, then link with the MPI wrapper plus the CUDA runtime. Produces the same
# build/cuda. Overrides:
#   make cuda-alt MPI_INC="-I/path/to/mpi/include"      (NVHPC / MPICH: see `mpicxx -show`)
#   make cuda-alt CUDA_HOME=/path/to/cuda  CUDA_ALT_HOSTCXX=g++-12
CUDA_ALT_HOSTCXX ?= g++
MPI_INC          ?= $(filter -I%,$(shell $(MPICXX) --showme:compile 2>/dev/null))
CUDA_HOME        ?= $(patsubst %/bin/nvcc,%,$(shell command -v $(NVCC) 2>/dev/null))
CUDA_ALT_LDFLAGS ?= -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/x86_64-linux/lib -lcudart \
                    -Wl,-rpath,$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/targets/x86_64-linux/lib
cuda-alt: | $(BUILD)
	@test -n "$(MPI_INC)" || echo "note: no MPI include flags found; pass MPI_INC=-I/path/to/mpi/include if the compile fails"
	$(NVCC) -c -O3 -std=c++17 -arch=$(CUDA_ARCH) -ccbin $(CUDA_ALT_HOSTCXX) $(MPI_INC) -DUSE_MPI -DUSE_CUDA $(CUDA_EXTRA) -Isrc -o $(BUILD)/cuda_main.o src/cuda_main.cu
	$(MPICXX) -o $(BUILD)/cuda $(BUILD)/cuda_main.o $(CUDA_ALT_LDFLAGS)

# Type-check the CUDA sources WITHOUT CUDA (uses src/cuda/stub/cuda_runtime.h): catches typos and
# wrong API usage on a laptop; it does not run or test anything.
cuda-check:
	$(MPICXX) $(CXXFLAGS) -x c++ -fsyntax-only -DUSE_MPI -DUSE_CUDA -DCUDA_STUB -Isrc src/cuda_main.cu

$(BUILD):
	mkdir -p $(BUILD)

test: serial mpi hybrid
	python3 scripts/validate.py all

clean:
	rm -rf $(BUILD)
