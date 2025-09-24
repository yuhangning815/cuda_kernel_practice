# Sigmoid Activation Function CUDA Kernels

## Overview

This project implements optimized CUDA kernels for the Sigmoid activation function. The Sigmoid function is a widely-used activation function in neural networks, defined as:

```
σ(x) = 1 / (1 + e^(-x))
```

The implementation provides both FP32 and FP16 precision variants with vectorized optimizations for improved memory bandwidth utilization.

## Hardware Configuration

- **GPU**: NVIDIA GeForce RTX 4060 (Ada Lovelace Architecture)
- **Compute Capability**: 8.9
- **Memory**: 8GB GDDR6
- **Tensor Cores**: 3rd Gen RT Cores, 3rd Gen Tensor Cores

## Mathematical Implementation

### Numerical Stability
The kernels implement input clamping to prevent numerical overflow in the exponential function:
- **FP32**: Input range clamped to [-88.376, 88.376] to prevent `exp()` overflow
- **FP16**: Input range clamped to [-9.704, 11.089] for half precision stability

### Sigmoid Computation
```cpp
// FP32 Implementation
y[i] = 1.0f / (1.0f + expf(-x[i]));

// FP16 Implementation  
y[i] = 1.0h / (1.0h + hexp(-x[i]));
```

## Kernel Implementations

### 1. Standard FP32 Kernel (`sigmoid_f32_kernel`)
- **Description**: Basic single-element processing kernel
- **Features**:
  - Single-threaded element processing
  - Input clamping to prevent numerical overflow
  - Standard sigmoid computation using `expf()`
- **Grid/Block Configuration**: `grid(N/256), block(256)`
- **Memory Pattern**: Coalesced access, 1 element per thread
- **Performance**: _______ ms (to be filled)

### 2. Vectorized FP32x4 Kernel (`sigmoid_f32x4_kernel`)
- **Description**: Vectorized kernel processing 4 elements per thread
- **Features**:
  - Uses `float4` vectorized loads/stores
  - 4x throughput improvement per thread
  - Reduces address calculation overhead
  - Coalesced memory access patterns
- **Grid/Block Configuration**: `grid(N/256), block(64)` (256/4 threads)
- **Memory Pattern**: 128-bit vectorized loads, 4 elements per thread
- **Performance**: _______ ms (to be filled)

### 3. Standard FP16 Kernel (`sigmoid_f16_kernel`)
- **Description**: Half precision single-element processing
- **Features**:
  - Half precision computation using `hexp()`
  - 2x memory bandwidth efficiency vs FP32
  - Input clamping optimized for FP16 range
  - Maintains computational accuracy for most use cases
- **Grid/Block Configuration**: `grid(N/256), block(256)`
- **Memory Pattern**: Coalesced access, 1 half per thread
- **Performance**: _______ ms (to be filled)

## Key Optimizations

### Memory Access Patterns
1. **Coalesced Access**: All kernels ensure consecutive memory access patterns
2. **Vectorized Operations**: FP32x4 uses SIMD loads for maximum bandwidth
3. **Memory Bandwidth**: FP16 achieves 2x bandwidth efficiency over FP32

### Computational Efficiency
1. **Input Clamping**: Prevents expensive overflow/underflow handling
2. **Native Functions**: Uses hardware-optimized `expf()` and `hexp()` functions
3. **Register Optimization**: Minimizes memory transactions through register usage

### Precision vs Performance Trade-offs
1. **FP32**: Maximum precision with standard performance
2. **FP32x4**: Same precision with 4x vectorized throughput
3. **FP16**: Reduced precision with 2x memory efficiency and faster computation

## Benchmarking

The `sigmoid.py` script provides comprehensive benchmarking comparing:
- Custom CUDA kernel implementations
- PyTorch native sigmoid implementation (`torch.sigmoid`)

### Test Matrix
- **Tensor Sizes**: [1024, 2048, 4096] × [1024, 2048, 4096]
- **Data Types**: FP32 (Float32) and FP16 (Half)
- **Iterations**: 1000 (with 10 warmup runs)

### Sample Results Template

#### FP32 Results
```
-------------------------------------------------------------------------------------
                                        S=1024, K=1024
            out_f32: [_______, _______], time:_______ms
          out_f32x4: [_______, _______], time:_______ms
          out_f32_th: [_______, _______], time:_______ms
-------------------------------------------------------------------------------------
```

#### FP16 Results  
```
-------------------------------------------------------------------------------------
            out_f16: [_______, _______], time:_______ms
           out_f16_th: [_______, _______], time:_______ms
-------------------------------------------------------------------------------------
```

## Implementation Notes

### FP32 Kernels
- **Range Safety**: Input values clamped to [-88.376, 88.376] to prevent `exp()` overflow
- **Vectorization**: FP32x4 processes 4 elements per thread using `float4` data types
- **Memory Alignment**: Assumes input tensors are properly aligned for vectorized access

### FP16 Kernels  
- **Range Safety**: Input values clamped to [-9.704, 11.089] for half precision stability
- **Native Half Operations**: Uses `hexp()` for hardware-accelerated half precision exponentials
- **Precision Trade-off**: Maintains sufficient accuracy for most neural network applications

### Boundary Handling
- **Vector Kernels**: Proper bounds checking for non-aligned tensor sizes
- **Memory Safety**: All kernels include proper bounds checking to prevent out-of-bounds access
- **Tail Handling**: Vectorized kernels handle non-multiple sizes gracefully

## Usage

```bash
# Compile and run the benchmark
python sigmoid.py
```

The script automatically compiles the CUDA kernels using PyTorch's JIT compilation and runs performance comparisons across different tensor sizes and precision levels.

## Technical Details

### Kernel Launch Configuration
- **Block Size**: 256 threads (FP32), 64 threads (FP32x4), 256 threads (FP16)
- **Grid Size**: Calculated based on total elements and threads per block
- **Occupancy**: Optimized for maximum SM utilization on Ada Lovelace

### Memory Requirements
- **FP32**: 4 bytes per element (input + output = 8 bytes per element)
- **FP16**: 2 bytes per element (input + output = 4 bytes per element)
- **Vectorized**: Reduced memory transaction overhead through coalesced access

### Numerical Accuracy
- **FP32**: IEEE 754 single precision (≈7 decimal digits)
- **FP16**: IEEE 754 half precision (≈3-4 decimal digits)
- **Range**: Sigmoid naturally maps to (0,1) range, well within both precisions
