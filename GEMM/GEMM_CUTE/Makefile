CC_FILES=$(shell find ./ -name "*.cu")
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	/usr/local/cuda-13.0/bin/nvcc -o $@ $< -O2 -arch=sm_89 -std=c++17 -I../../third-party/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas

clean:
	rm -rf $(EXE_FILES)