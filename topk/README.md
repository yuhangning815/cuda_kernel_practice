# Top-K Algorithms - CUDA Implementation

This directory contains CUDA implementations of histogram computation and top-k radix select algorithms, with detailed optimizations and explanations.

## 📁 Contents

### Programs

1. **`histogram.cu`** - Histogram computation with 3 implementations:
   - CPU baseline
   - GPU naive (direct global atomics)
   - GPU shared memory (optimized)
   - GPU privatized (highly optimized with grid-stride loop)

2. **`radix_select.cu`** - Top-K selection using radix select with bit comparison:
   - CPU baseline (std::sort)
   - GPU naive (direct partition)
   - GPU optimized (shared memory buffering)

### Documentation

- **`RADIX_SELECT_EXPLAINED.md`** - Comprehensive explanation of radix select algorithm
  - Algorithm details
  - Bit manipulation techniques
  - Optimization strategies
  - Performance analysis

### Build System

- **`Makefile`** - Build and run all programs

## 🚀 Quick Start

### Build All Programs
```bash
make
```

### Build Individual Programs
```bash
make histogram      # Build histogram only
make radix_select   # Build radix_select only
```

### Run Programs
```bash
make run            # Run all programs
make run-histogram  # Run histogram only
make run-radix      # Run radix_select only
```

### Clean Build Artifacts
```bash
make clean
```

## 📊 Performance Results

### Histogram (100M elements, 256 bins)

| Implementation | Time (ms) | Speedup vs CPU |
|----------------|-----------|----------------|
| CPU | 26.7 | 1x |
| GPU Naive | 25.6 | 1.04x |
| GPU Shared Memory | 4.1 | **6.5x** |
| GPU Privatized | 0.97 | **27x** |

**Key Insight:** Shared memory reduces global atomic contention by 21x!

### Top-K Radix Select (10M elements, Top-1000)

| Implementation | Time (ms) | Speedup vs CPU |
|----------------|-----------|----------------|
| CPU (std::sort) | 700 | 1x |
| GPU Naive | 6.1 | **114x** |
| GPU Optimized | 6.2 | **112x** |

**Key Insight:** Radix select is O(N × bits) instead of O(N log N), and massively parallel!

## 🎓 Learning Resources

### Core Concepts

Both implementations demonstrate fundamental CUDA optimization patterns:

#### 1. **Two-Phase Approach**
```
Phase 1: Fast local operations (shared memory)
Phase 2: Infrequent global synchronization
```

#### 2. **Memory Hierarchy**
```
Registers:      ~1 cycle    ⚡⚡⚡⚡⚡
Shared Memory:  ~20 cycles  ⚡⚡⚡⚡
Global Memory:  ~500 cycles ⚡
```

#### 3. **Atomic Operation Reduction**
- Naive: N atomic operations to global memory
- Optimized: ~N/BLOCK_SIZE atomic operations to global memory
- Speedup: 100-1000x on atomic-heavy workloads

#### 4. **Grid-Stride Loop Pattern**
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;

for (int i = idx; i < n; i += stride) {
    process(data[i]);
}
```
Benefits:
- Works with any data size
- Better cache utilization
- Enables privatization optimization

## 🔧 Customization

### Adjust Problem Size

**Histogram:**
```cpp
// In histogram.cu, line ~145
const int N = 100 * 1024 * 1024;  // Change data size
#define NUM_BINS 256               // Change number of bins
```

**Radix Select:**
```cpp
// In radix_select.cu, line ~290
const int N = 10 * 1024 * 1024;   // Change data size
const int K = 1000;                 // Change K
```

### Tune Block Size

```cpp
// In both files
#define BLOCK_SIZE 256  // Try: 128, 256, 512
```

### Adjust GPU Architecture

```makefile
# In Makefile
NVCCFLAGS = -O3 -arch=sm_75 -std=c++11

# Common architectures:
# sm_75 - RTX 20 series, T4
# sm_80 - A100
# sm_86 - RTX 30 series
# sm_89 - RTX 40 series
```

## 🧪 Experiments to Try

### 1. Histogram: Vary Number of Bins
```cpp
#define NUM_BINS 64    // Low contention
#define NUM_BINS 256   // Medium
#define NUM_BINS 1024  // High contention
```
**Question:** How does performance scale with contention?

### 2. Histogram: Data Distribution
```cpp
// Uniform (current)
data[i] = rand() * 19 % NUM_BINS;

// Skewed (worst case)
data[i] = rand() % 32;  // Only first 32 bins

// Sequential (best locality)
data[i] = i % NUM_BINS;
```
**Question:** Which distribution is hardest for atomics?

### 3. Radix Select: Different K Values
```cpp
const int K = 10;      // Very selective
const int K = 1000;    // Moderate  
const int K = 100000;  // Large K
```
**Question:** How many bits are examined on average?

### 4. Privatization: Vary Number of Blocks
```cpp
// In histogram.cu, line ~245
int num_blocks = 32;    // Underutilized
int num_blocks = 128;   // Balanced
int num_blocks = 1024;  // Over-provisioned
```
**Question:** What's the sweet spot for your GPU?

## 📚 Related Patterns

These optimization techniques apply to many parallel algorithms:

- **Reduction** (sum, max, min)
- **Prefix Sum** (scan operations)
- **Matrix Multiplication** (tiling)
- **Sorting** (radix sort, bucket sort)
- **Graph Algorithms** (BFS, connected components)

**Universal Pattern:**
> Minimize slow global operations by buffering in fast local memory!

## 🐛 Troubleshooting

### Compilation Error: Unsupported Architecture
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Update Makefile with correct sm_XX
```

### Incorrect Results
- Check that data size is compatible with block size
- Verify atomic operations are used correctly
- Ensure synchronization (__syncthreads) is in correct places

### Performance Issues
- Try different block sizes (must be multiple of 32)
- Adjust number of blocks for privatization
- Profile with `nvprof` or `nsys` to find bottlenecks

## 📖 Further Reading

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Radix Select: [Quick Select on GPU](https://research.nvidia.com/publication/2008-08_efficient-selection-algorithms-gpu)
- [GPU Gems 3 - Chapter 39: Parallel Prefix Sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

## 🤝 Contributing

Feel free to:
- Add more optimizations
- Try different algorithms (heap select, quickselect)
- Implement for different data types
- Add more detailed profiling

## 📝 License

This code is for educational purposes.

---

**Happy GPU Programming! 🚀**

