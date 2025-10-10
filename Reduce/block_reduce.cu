#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])




// FP32
// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int offset = kWarpSize >> 1; offset >= 1; offset >>= 1) {
    // 0xffffffff is the active lane
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}





// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
  constexpr int NUM_WARPS = (NUM_THREADS + 32 - 1) / 32;
  __shared__ float reduce_smem[NUM_WARPS];         // Each warp does a reduce and store the result to shared memory !!!
  
  int gemm_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  float val = (gemm_idx < N) ? a[gemm_idx] : 0.0f;
  val = warp_reduce_sum_f32<32>(val);

  if (threadIdx.x % 32 == 0) {
    reduce_smem[threadIdx.x / 32] = val;
  }
  __syncthreads();

  if (threadIdx.x < NUM_WARPS) {  // // 只用前 NUM_WARPS 个 thread ( < 32) 
    float val_sum = reduce_smem[threadIdx.x];
    val_sum = warp_reduce_sum_f32<NUM_WARPS>(val_sum);
    if (threadIdx.x == 0) {
      atomicAdd(y, val_sum);  // // 所有block 同时加到 global mem 要atomic add
    }
  }
}





// Block All Reduce Sum + float4
// grid(N/256), block(256/4)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *a, float *y,
                                                      int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  // Use 4x less threads and shared memory -> do the sum while loading from the global memory 
  float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;      // 在load 进来的时候就先做了一次加法。一个block 可以处理 1024 * 4096 个元素
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}









#define STRINGFY(str) #str

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define LANUCH_REDUCE_KERNEL(NumThreads, packed_type, acc_type, element_type,          \
                             out_type)                                         \
  block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NumThreads)>               \
      <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),        \
                        reinterpret_cast<out_type *>(y.data_ptr()), N);

#define DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
  const int NumThreads = (K) / (n_elements);                                           \
  dim3 block(NumThreads);                                                              \
  dim3 grid((S));                                                              \
  switch (NumThreads) {                                                                \
  case 32:                                                                     \
    LANUCH_REDUCE_KERNEL(32, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_REDUCE_KERNEL(64, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_REDUCE_KERNEL(128, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_REDUCE_KERNEL(256, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_REDUCE_KERNEL(512, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_REDUCE_KERNEL(1024, packed_type, acc_type, element_type, out_type)  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }



#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type,     \
                             n_elements, out_type)                             \
  torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(               \
      torch::Tensor x) {                                                       \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto y_th_type =                                                           \
        (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;           \
    auto options =                                                             \
        torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);       \
    auto y = torch::zeros({1}, options);                                       \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \                                       
      dim3 grid((N + 1024 - 1) / 1024);                                        \
      dim3 block(1024 / (n_elements));                                         \
      block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /          \
                                                               (n_elements)>   \
          <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),    \
                            reinterpret_cast<out_type *>(y.data_ptr()), N);    \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        /* each row fits in one block, so we have one block per row*/                                                                       \
        DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 grid((N + 1024 - 1) / 1024);                                      \
        dim3 block(1024 / (n_elements));                                       \
        block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /        \
                                                                 (n_elements)> \
            <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),  \
                              reinterpret_cast<out_type *>(y.data_ptr()), N);  \
      }                                                                        \
    }                                                                          \
    return y;                                                                  \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32, f32, torch::kFloat32, float, 1, float)      // -> Copy pasted as an actual C++ function: block_all_reduce_sum_f32_f32
TORCH_BINDING_REDUCE(f32x4, f32, torch::kFloat32, float, 4, float)     // -> Copy pasted as an actual C++ function: block_all_reduce_sum_f32x4_f32



#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
}