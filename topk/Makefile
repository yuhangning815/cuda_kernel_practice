# CUDA Top-K and Histogram Makefile

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3 -arch=sm_75 -std=c++11

# Target executables
TARGETS = histogram radix_select

# Default target
all: $(TARGETS)

# Build the histogram program
histogram: histogram.cu
	$(NVCC) $(NVCCFLAGS) -o histogram histogram.cu
	@echo "Build complete: histogram"

# Build the radix_select program
radix_select: radix_select.cu
	$(NVCC) $(NVCCFLAGS) -o radix_select radix_select.cu
	@echo "Build complete: radix_select"

# Run histogram
run-histogram: histogram
	./histogram

# Run radix_select
run-radix: radix_select
	./radix_select

# Run both programs
run: $(TARGETS)
	@echo "==================== Running Histogram ===================="
	./histogram
	@echo ""
	@echo "==================== Running Radix Select ===================="
	./radix_select

# Clean build artifacts
clean:
	rm -f $(TARGETS)
	@echo "Cleaned build artifacts"

# Rebuild (clean + build)
rebuild: clean all

# Help target
help:
	@echo "Available targets:"
	@echo "  make               - Build all programs"
	@echo "  make all           - Build all programs"
	@echo "  make histogram     - Build only histogram"
	@echo "  make radix_select  - Build only radix_select"
	@echo "  make run           - Build and run all programs"
	@echo "  make run-histogram - Build and run histogram"
	@echo "  make run-radix     - Build and run radix_select"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make rebuild       - Clean and rebuild"
	@echo "  make help          - Show this help message"

.PHONY: all run run-histogram run-radix clean rebuild help
