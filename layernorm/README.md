# Layer Normalization CUDA Kernels

## Overview

This project implements optimized CUDA kernels for Layer Normalization, a fundamental normalization technique in deep learning. Layer Normalization normalizes inputs across the feature dimension for each sample independently, providing stable training dynamics and improved convergence.

The mathematical formulation is:
```
LayerNorm(x) = γ * (x - μ) / σ + β

Where:
- μ = mean(x) = (1/K) * Σx_i        (mean across feature dimension)
- σ = std(x) = sqrt((1/K) * Σ(x_i - μ)²)  (standard deviation)
- γ = scale parameter (learnable)
- β = bias parameter (learnable)
```

## Hardware Configuration

- **GPU**: NVIDIA GeForce RTX 4060 (Ada Lovelace Architecture)
- **Compute Capability**: 8.9
- **Memory**: 8GB GDDR6
- **Tensor Cores**: 3rd Gen RT Cores, 3rd Gen Tensor Cores
- **Warp Size**: 32 threads

## Algorithm Implementation

### Two-Pass Approach
Our kernels use a numerically stable two-pass algorithm:

1. **First Pass**: Compute mean (μ) across feature dimension
2. **Second Pass**: Compute variance and apply normalization

### Block-Level Reduction
- **Warp Reduction**: Uses `__shfl_xor_sync()` for intra-warp communication
- **Block Reduction**: Shared memory aggregation across warps
- **Numerical Stability**: epsilon = 1e-5 added to variance for stability

## Kernel Implementations

### 1. Standard FP32 Kernel (`layer_norm_f32_kernel`)
- **Description**: Basic single precision layer normalization
- **Features**:
  - Single element per thread processing
  - Two-stage reduction using warp primitives
  - Shared memory for mean and variance storage
  - Numerically stable computation
- **Grid/Block Configuration**: `grid(N), block(K)` where N=batch_size, K=hidden_size
- **Memory Pattern**: Coalesced access within each sequence
- **Performance**: _______ ms (to be filled)

### 2. Vectorized FP32x4 Kernel (`layer_norm_f32x4_kernel`)  
- **Description**: Vectorized kernel processing 4 elements per thread
- **Features**:
  - Uses `float4` vectorized loads/stores
  - 4x computational throughput per thread
  - Pre-reduction of 4 elements before block reduction
  - Optimized memory bandwidth utilization
- **Grid/Block Configuration**: `grid(N), block(K/4)` 
- **Memory Pattern**: 128-bit vectorized loads, 4 elements per thread
- **Performance**: _______ ms (to be filled)

### 3. Half Precision FP16 Kernel (`layer_norm_f16_f16_kernel`)
- **Description**: Half precision implementation with native half operations
- **Features**:
  - All computations in FP16 for memory efficiency
  - Uses `hrsqrt()` and `__hfma()` for hardware acceleration
  - 2x memory bandwidth efficiency vs FP32
  - Reduced precision but sufficient accuracy for most use cases
- **Grid/Block Configuration**: `grid(N), block(K)`
- **Memory Pattern**: Coalesced half precision access
- **Performance**: _______ ms (to be filled)

## Reduction Strategy

### Hierarchical Block Reduction
```cpp
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  // Step 1: Warp-level reduction using shuffle
  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  
  // Step 2: Warp leaders store to shared memory  
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  
  // Step 3: First warp reduces across warp results
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  
  return val; // Thread 0 has the final result
}
```

### Warp-Level Primitives
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

## Benchmarking

The `layernorm.py` script provides comprehensive performance evaluation comparing:
- Custom CUDA kernel implementations
- PyTorch native layer normalization

### Test Configuration
- **Tensor Shapes**: N × K matrices where:
  - N = 4096 (batch size × sequence length)  
  - K = [512, 1024, 2048, 4096, 8192] (hidden dimensions)
- **Data Types**: FP32 (Float32) and FP16 (Half)
- **Parameters**: γ=1.0 (scale), β=0.0 (bias)
- **Iterations**: 1000 (with 10 warmup runs)

### Sample Results Template

#### FP32 Results (N=4096, K=512)
```
-------------------------------------------------------------------------------------
                                        N=4096, K=512
            out_f32: [_______, _______, _______], time:_______ms
          out_f32x4: [_______, _______, _______], time:_______ms
          out_f32_th: [_______, _______, _______], time:_______ms
-------------------------------------------------------------------------------------
```

#### FP16 Results (N=4096, K=512)
```
-------------------------------------------------------------------------------------
         out_f16f16: [_______, _______, _______], time:_______ms
          out_f16_th: [_______, _______, _______], time:_______ms
-------------------------------------------------------------------------------------
```

## Key Optimizations

### Memory Access Patterns
1. **Coalesced Access**: All threads in a warp access consecutive memory locations
2. **Vectorized Loads**: `float4` operations maximize memory bandwidth utilization  
3. **Shared Memory**: Efficient cross-warp communication for reduction operations

### Computational Efficiency
1. **Warp Primitives**: `__shfl_xor_sync()` eliminates shared memory for intra-warp reductions
2. **Fused Operations**: `__hfma()` for fused multiply-add in FP16
3. **Numerical Stability**: Carefully ordered operations to prevent overflow/underflow

### Precision vs Performance
1. **FP32**: Maximum numerical precision with standard performance
2. **FP32x4**: Same precision with 4x vectorized throughput  
3. **FP16**: Reduced precision with 2x memory efficiency and faster computation

## Implementation Notes

### Supported Hidden Dimensions
- **FP32 Kernels**: K ∈ {64, 128, 256, 512, 1024}
- **FP32x4 Kernels**: K ∈ {64, 128, 256, 512, 1024, 2048, 4096} (must be divisible by 4)
- **FP16 Kernels**: K ∈ {64, 128, 256, 512, 1024}

### Numerical Considerations
- **Epsilon**: 1e-5 added to variance to prevent division by zero
- **Precision**: FP16 sufficient for most transformer applications
- **Overflow Protection**: Input ranges handled safely by all implementations

### Memory Requirements
- **FP32**: 8 bytes per element (input + output)
- **FP16**: 4 bytes per element (input + output) 
- **Shared Memory**: ~1KB per block for reduction operations

## Usage

```bash
# Compile and run the benchmark
python layernorm.py
```

The script automatically compiles the CUDA kernels using PyTorch's JIT compilation and runs comprehensive performance comparisons across different tensor configurations and precision levels.

## Technical Details

### Block Configuration Strategy
- **Grid Dimension**: One block per sequence (N blocks total)
- **Block Dimension**: One thread per feature element (K threads per block)
- **Occupancy**: Optimized for maximum SM utilization on Ada Lovelace

### Synchronization Points
1. **After Mean Computation**: `__syncthreads()` ensures mean is available to all threads
2. **After Variance Computation**: `__syncthreads()` ensures variance is available
3. **Warp Synchronous**: Shuffle operations are naturally synchronous within warps

### Performance Characteristics
- **Memory Bound**: Performance primarily limited by memory bandwidth
- **Compute Intensity**: Low arithmetic intensity (O(K) operations per element)
- **Scalability**: Linear scaling with hidden dimension size within block limits
