# GELU Kernel Implementation

## Overview

This project implements optimized CUDA kernels for the Gaussian Error Linear Unit (GELU) activation function. The GELU function is defined as:

```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

The implementation uses the tanh approximation variant which is commonly used in deep learning frameworks for computational efficiency.

## Hardware Configuration

- **GPU**: NVIDIA GeForce RTX 4060 (Ada Lovelace Architecture)
- **Compute Capability**: 8.9
- **Memory**: 8GB GDDR6

## Kernel Implementations

### 1. Standard FP32 Kernel (`gelu_f32_kernel`)
- **Description**: Basic single-element processing kernel
- **Features**: 
  - Single-threaded element processing
  - Input clamping to prevent numerical overflow
  - Tanh-approximated GELU computation
- **Grid/Block Configuration**: `grid(N/256), block(256)`
- **Performance**: _______ ms (to be filled)

### 2. Vectorized FP32x4 Kernel (`gelu_f32x4_kernel`) 
- **Description**: Vectorized kernel processing 4 elements per thread
- **Features**:
  - Uses `float4` vectorized loads/stores
  - 4x throughput improvement per thread
  - Coalesced memory access patterns
- **Grid/Block Configuration**: `grid(N/256), block(64)` (256/4 threads)  
- **Performance**: _______ ms (to be filled)

## Benchmarking

The `gelu.py` script provides comprehensive benchmarking comparing:
- Custom CUDA kernel implementations (`gelu_f32`, `gelu_f32x4`)
- PyTorch native GELU implementation (`torch.gelu`)

### Test Matrix
- **Tensor Sizes**: [1024, 2048, 4096] × [1024, 2048, 4096]
- **Data Type**: FP32 (Float32)
- **Iterations**: 1000 (with 10 warmup runs)

### Sample Results Template
```
-------------------------------------------------------------------------------------
                                        S=1024, K=1024
           out_f32: [_______, _______], time:_______ms
         out_f32x4: [_______, _______], time:_______ms
         out_f32_th: [_______, _______], time:_______ms
-------------------------------------------------------------------------------------
```

## Key Optimizations

1. **Memory Coalescing**: Vectorized memory operations using `float4`
2. **Numerical Stability**: Input clamping to prevent exp() overflow
3. **Mathematical Approximation**: Tanh-based approximation for computational efficiency
4. **Thread Utilization**: Optimized block sizes for maximum occupancy

## Usage

```bash
# Compile and run the benchmark
python gelu.py
```

The script will automatically compile the CUDA kernels using PyTorch's JIT compilation and run performance comparisons across different tensor sizes.

## Implementation Notes

- The kernel uses tanh approximation which provides good accuracy while being computationally efficient
- Input values are clamped to the range [-88.376, 88.376] to prevent numerical overflow in the exp() function
- The vectorized version processes 4 elements per thread using SIMD operations for improved throughput
- Memory access patterns are optimized for coalescing to maximize memory bandwidth utilization
