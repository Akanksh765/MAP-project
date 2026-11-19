# make            -> gen, graph_omp, graph_mpi (CPU)   |  make cuda ARCH=sm_80  -> graph_cuda (needs nvcc + GPU)
ARCH ?= sm_70
CXXFLAGS = -O3 -march=native -std=c++17
all: gen graph_omp graph_mpi
gen: src/gen.cpp;               g++ $(CXXFLAGS) -o $@ $<
graph_omp: src/graph_omp.cpp src/common.h;  g++ $(CXXFLAGS) -fopenmp -o $@ $<
graph_mpi: src/graph_mpi.cpp src/common.h;  mpicxx $(CXXFLAGS) -o $@ $<
cuda: src/graph_cuda.cu src/common.h;       nvcc -O3 -std=c++17 -arch=$(ARCH) -o graph_cuda $<
