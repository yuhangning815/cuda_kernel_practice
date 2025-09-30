# Block Reduce Sum CUDA Kernels

## Overview

This project implements optimized CUDA kernels for block-wise reduction sum operations using warp-level primitives and shared memory. The kernels efficiently compute the sum of all elements in a tensor using hierarchical reduction patterns.

The implementation uses a two-level reduction strategy:
1. **Warp-level reduction**: Uses `__shfl_xor_sync()` for efficient intra-warp communication
2. **Block-level reduction**: Uses shared memory to aggregate results from multiple warps
3. **Global reduction**: Uses atomic operations to combine results from multiple blocks

## Hardware Configuration

- **GPU**: NVIDIA GeForce RTX 4060 (Ada Lovelace Architecture)
- **Compute Capability**: 8.9
- **Memory**: 8GB GDDR6
- **Warp Size**: 32 threads
- **Max Threads per Block**: 1024

## Algorithm Overview

### Reduction Strategy
```
Block Structure: 256 threads = 8 warps × 32 threads
│
├─ Warp 0: Reduce 32 elements → 1 value
├─ Warp 1: Reduce 32 elements → 1 value  
├─ ...
└─ Warp 7: Reduce 32 elements → 1 value
   │
   └─ Shared Memory: Combine 8 warp results → 1 block result
      │
      └─ Atomic Add: Combine all block results → Final sum
```

## Kernel Implementations

### 1. Standard FP32 Kernel (`block_all_reduce_sum_f32_f32_kernel`)
- **Description**: Basic single-element per thread reduction
- **Features**:
  - Each thread processes 1 float element
  - Warp-level reduction using shuffle instructions
  - Shared memory for inter-warp communication
  - Atomic addition for final result
- **Grid/Block Configuration**: `grid(N/256), block(256)`
- **Memory Pattern**: Coalesced access, 1 element per thread
- **Performance of S=4096 and K=4096**: 0.469 ms 

### 2. Vectorized FP32x4 Kernel (`block_all_reduce_sum_f32x4_f32_kernel`)
- **Description**: Vectorized kernel processing 4 elements per thread
- **Features**:
  - Uses `float4` vectorized loads for memory efficiency
  - Each thread pre-reduces 4 elements before warp reduction
  - 4x memory throughput improvement per thread
  - Same warp/block reduction pattern as standard kernel
- **Grid/Block Configuration**: `grid(N/256), block(64)` (256/4 threads)
- **Memory Pattern**: 128-bit coalesced loads, 4 elements per thread
- **Performance of S=4096 and K=4096 **: 0.273 ms 

## Warp-Level Primitives

### Shuffle-based Reduction
```cpp
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}
```

**Reduction Pattern**: Butterfly network with XOR shuffle
- Step 1: Thread pairs (0,16), (1,17), ..., (15,31) exchange values
- Step 2: Thread pairs (0,8), (1,9), ..., (7,15) exchange values  
- Step 3: Thread pairs (0,4), (1,5), ..., (3,7) exchange values
- Step 4: Thread pairs (0,2), (1,3) exchange values
- Step 5: Thread pair (0,1) exchanges values
- Result: Thread 0 holds the sum of all 32 values

## Benchmarking

The `block_reduce.py` script provides performance comparison between:
- Custom CUDA kernels (`block_all_reduce_sum_f32_f32`, `block_all_reduce_sum_f32x4_f32`)
- PyTorch native implementation (`torch.sum`)

## Key Optimizations

### Memory Access Patterns
1. **Coalesced Access**: All threads in a warp access consecutive memory locations
2. **Vectorized Loads**: `float4` loads maximize memory bandwidth utilization
3. **Shared Memory**: Efficient inter-warp communication with low latency

### Computational Efficiency  
1. **Warp Primitives**: `__shfl_xor_sync()` eliminates need for shared memory in warp reduction
2. **Loop Unrolling**: `#pragma unroll` ensures compile-time loop optimization
3. **Atomic Operations**: Final accumulation using hardware atomic adds

### Thread Utilization
1. **Full Warp Utilization**: 256 threads per block = 8 full warps
2. **Load Balancing**: Even work distribution across all threads
3. **Occupancy**: Block size optimized for maximum SM occupancy

## Usage

```bash
# Compile and run the benchmark
python block_reduce.py
```

The script automatically compiles the CUDA kernels using PyTorch's JIT compilation and runs performance comparisons across different tensor configurations.

## Implementation Notes

- **Numerical Stability**: Uses float32 precision throughout to maintain accuracy
- **Boundary Handling**: Proper bounds checking for non-power-of-2 tensor sizes  
- **Atomic Consistency**: Global atomic operations ensure race-condition-free final results
- **Template Flexibility**: Kernels are templated for different block sizes and data types
- **Memory Bandwidth**: Vectorized operations maximize memory subsystem utilization


Command line:
```bash
TORCH_CUDA_ARCH_LIST="8.9+PTX" python block_reduce.py