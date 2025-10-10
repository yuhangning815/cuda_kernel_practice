#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

template <typename T>
void gen_random_data(T *data, int n);

// Native CUDA GEMM kernel (no CuTe)
// template <typename T>
// __global__ void gemm_native_cuda(T *C, const T *A, const T *B, int M, int N, int K) {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (row < M && col < N) {
//         T sum = T(0.0f);
//         for (int k = 0; k < K; k++) {
//             // A is M x K, B is N x K (so B^T is K x N)
//             // C = A * B^T, so C[row][col] = sum(A[row][k] * B[col][k])
//             sum += A[row * K + k] * B[col * K + k];
//         }
//         C[row * N + col] = sum;
//     }
// }



#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define NumBuffer 2

#define OFFSET(row, col, ld) ((row) * (ld) + (col))


// dim3 grid_native((n + BN - 1) / BN, (m + BM - 1) / BM);
// dim3 block_native(BN / TN, BM / TM);
// A is M x K in row major, B is N x K in row major 
// Native CUDA GEMM kernel (no CuTe)
template <typename T>
__global__ void gemm_native_cuda(T *C, const T *A, const T *B, int M, int N, int K) {

  // 1. avoid bank conflicts
  // 2. Double buffer

  __shared__ T shmem_A[NumBuffer][BM][BK + 1];
  __shared__ T shmem_B[NumBuffer][BN][BK + 1];


  // each thread is also responsible for a block 
  float t_c[TM][TN] = {0.0};

  int gmem_start_m = blockIdx.y * BM;
  int gmem_start_n = blockIdx.x * BN;


  // k_idx = 0 - first load
  int buffer_idx = 0;
  for (int j = 0; j < BM * BK; j += blockDim.x * blockDim.y) {
    int row = (j + threadIdx.y * blockDim.x + threadIdx.x) / BK;
    int col = (j + threadIdx.y * blockDim.x + threadIdx.x) % BK;
    shmem_A[buffer_idx][row][col] = A[OFFSET(gmem_start_m + row,col, K)];
  }

  for (int j = 0; j < BN * BK; j += blockDim.x * blockDim.y) {
      int row = (j + threadIdx.y * blockDim.x + threadIdx.x) / BK;
      int col = (j + threadIdx.y * blockDim.x + threadIdx.x) % BK;
      shmem_B[buffer_idx][row][col] = B[OFFSET(gmem_start_n + row,col, K)];
  }

  // start from k_idx = 1
  for (int k_idx = 1; k_idx < (K + BK - 1) / BK; k_idx ++) {
    int gemm_start_k_next = k_idx * BK;

    int buffer_idx_next = (buffer_idx + 1) % NumBuffer;
    for (int j = 0; j < BM * BK; j += blockDim.x * blockDim.y) {
        int row = (j + threadIdx.y * blockDim.x + threadIdx.x) / BK;
        int col = (j + threadIdx.y * blockDim.x + threadIdx.x) % BK;
        shmem_A[buffer_idx_next][row][col] = A[OFFSET(gmem_start_m + row, gemm_start_k_next + col, K)];
    }
    
    for (int j = 0; j < BN * BK; j += blockDim.x * blockDim.y) {
        int row = (j + threadIdx.y * blockDim.x + threadIdx.x) / BK;
        int col = (j + threadIdx.y * blockDim.x + threadIdx.x) % BK;
        shmem_B[buffer_idx_next][row][col] = B[OFFSET(gmem_start_n + row, gemm_start_k_next + col, K)];
    }

    // [128, 8] x [128, 8] = [128, 128]
    // each thread traverses the K dimension to compute!
    #pragma unroll
    for (int k = 0; k < BK; k++ ){
      #pragma unroll
      for (int i = 0; i < TM; i++ ){
        #pragma unroll
        for (int j = 0; j < TN; j++ ){
            int output_row = threadIdx.y * TM + i;
            int output_col = threadIdx.x * TN + j;
            t_c[i][j] += shmem_A[buffer_idx][output_row][k] * shmem_B[buffer_idx][output_col][k];
        }
      }
    }
    buffer_idx = buffer_idx_next;
    __syncthreads();
  } // for k_idx

  // last compute
  #pragma unroll
  for (int k = 0; k < BK; k++ ){
    #pragma unroll
    for (int i = 0; i < TM; i++ ){
      #pragma unroll
      for (int j = 0; j < TN; j++ ){
          int output_row = threadIdx.y * TM + i;
          int output_col = threadIdx.x * TN + j;
          t_c[i][j] += shmem_A[buffer_idx][output_row][k] * shmem_B[buffer_idx][output_col][k];
      }
    }
  }

  // write back to global memory
  #pragma unroll
  for (int i = 0; i < TM; i++ ){
    #pragma unroll
    for (int j = 0; j < TN; j++ ){
        int output_row = threadIdx.y * TM + i;
        int output_col = threadIdx.x * TN + j;
        C[OFFSET(gmem_start_m + output_row, gmem_start_n + output_col, N)] = t_c[i][j];
    }
  }
}







template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  // 1. Partition the Global Maqtrix into Tiles 
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  // register represetation of the tile 
  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  clear(tCrC);
  
  int num_tile_k = size<2>(gA);

  #pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}







int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 81920;
  int n = 256;
  int k = 256;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*) malloc(sizeof(T) * m * k);
  Bptr_host = (T*) malloc(sizeof(T) * n * k);
  gen_random_data(Aptr_host, m * k);
  gen_random_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);


  // A is row major, B is col major 
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;


  // <2,2,1> -> there will be 2 warps horizontal, and 2 warps vertical -> 128 threads per block 
  // <1,2,1> -> each warp takes care of 2 blocks along N. 
  // Data output = <32, 32> , k = 16 
  using MMA = decltype(make_tiled_mma(mma_atom{},   // define the tiled_mma 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));

  constexpr int kTileM = 128; 
  constexpr int kTileN = 128; 
  constexpr int kTileK = 32; 

  // Test CuTe implementation
  dim3 block(size(MMA{}));
  dim3 grid(n / kTileN, m / kTileM);
  
  cudaEvent_t start_cute, stop_cute;
  cudaEventCreate(&start_cute);
  cudaEventCreate(&stop_cute);
  
  cudaEventRecord(start_cute);
  for (int i = 0; i < 10; ++i) {
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaEventRecord(stop_cute);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  printf("CuTe err = %d, str = %s\n", err, cudaGetErrorString(err));
  
  float milliseconds_cute = 0;
  cudaEventElapsedTime(&milliseconds_cute, start_cute, stop_cute);
  printf("CuTe GEMM time: %.3f ms (average per iteration: %.3f ms)\n", 
         milliseconds_cute, milliseconds_cute / 10.0f);

  // ---------------------- Test Native CUDA implementation ----------------------
  T *Cptr_native;
  cudaMalloc(&Cptr_native, sizeof(T) * m * n);
  
  // dim3 block_native(16, 16);
  // dim3 grid_native((n + 15) / 16, (m + 15) / 16);

  dim3 grid_native((n + BN - 1) / BN, (m + BM - 1) / BM);
  dim3 block_native(BN / TN, BM / TM);

  cudaEvent_t start_native, stop_native;
  cudaEventCreate(&start_native);
  cudaEventCreate(&stop_native);
  
  cudaEventRecord(start_native);
  for (int i = 0; i < 10; ++i) {
    gemm_native_cuda<T><<<grid_native, block_native>>>(Cptr_native, Aptr, Bptr, m, n, k);
  }
  cudaEventRecord(stop_native);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("Native CUDA err = %d, str = %s\n", err, cudaGetErrorString(err));
  
  float milliseconds_native = 0;
  cudaEventElapsedTime(&milliseconds_native, start_native, stop_native);
  printf("Native CUDA GEMM time: %.3f ms (average per iteration: %.3f ms)\n", 
         milliseconds_native, milliseconds_native / 10.0f);

  // ---------------------- cublas ----------------------------
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);
  
  cudaEvent_t start_cublas, stop_cublas;
  cudaEventCreate(&start_cublas);
  cudaEventCreate(&stop_cublas);
  
  cudaEventRecord(start_cublas);
  for (int i = 0; i < 100; ++i) {
    // T = Transpose, N = Normal 
    // in cublas, Ret^T = B^T * A^T (cublas normally return the col major matrix )
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  n, m, k,
          	  &alpha,
          	  (half *)Bptr, k,
          	  (half *)Aptr, k,
          	  &beta,
          	  (half *)Cptr_cublas, n);       // Cptr_cublas is the result pointer
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }
  cudaEventRecord(stop_cublas);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("cuBLAS err = %d, str = %s\n", err, cudaGetErrorString(err));
  
  float milliseconds_cublas = 0;
  cudaEventElapsedTime(&milliseconds_cublas, start_cublas, stop_cublas);
  printf("cuBLAS GEMM time: %.3f ms (average per iteration: %.3f ms)\n", 
         milliseconds_cublas, milliseconds_cublas / 100.0f);

  T *Cptr_host;
  T *Cptr_native_host;
  T *Cptr_cublas_host;

  Cptr_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_native_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // compare all three implementations
  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_native_host, Cptr_native, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.2;
  int cute_cublas_diff = 0, native_cublas_diff = 0;
  for (int i = 0; i < m * n; ++i) {
    float cute_val = Cptr_host[i];
    float native_val = Cptr_native_host[i];
    float cublas_val = Cptr_cublas_host[i];
    
    if (fabs(cublas_val - cute_val) > threshold) {
      printf("CuTe vs cuBLAS diff: cute=%f, cublas=%f\n", cute_val, cublas_val);
      cute_cublas_diff++;
    }
    if (fabs(cublas_val - native_val) > threshold) {
      printf("Native vs cuBLAS diff: native=%f, cublas=%f\n", native_val, cublas_val);
      native_cublas_diff++;
    }
  }
  
  printf("Differences found: CuTe-cuBLAS: %d, Native-cuBLAS: %d\n", cute_cublas_diff, native_cublas_diff);

  // Performance Summary
  printf("\n==================== Performance Summary ====================\n");
  printf("Matrix dimensions: M=%d, N=%d, K=%d\n", m, n, k);
  printf("CuTe GEMM:        %.3f ms/iter\n", milliseconds_cute / 10.0f);
  printf("Native CUDA GEMM: %.3f ms/iter\n", milliseconds_native / 10.0f);
  printf("cuBLAS GEMM:      %.3f ms/iter\n", milliseconds_cublas / 100.0f);
  printf("=============================================================\n\n");

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_native = make_tensor(Cptr_native_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_cublas_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(4, 4);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_native = local_tile(tensor_C_native, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  printf("CuTe result (first 4x4):\n");
  print_tensor(tc1);
  printf("Native CUDA result (first 4x4):\n");
  print_tensor(tc1_native);
  printf("cuBLAS result (first 4x4):\n");
  print_tensor(tc1_cublas);

  // Cleanup
  cudaEventDestroy(start_cute);
  cudaEventDestroy(stop_cute);
  cudaEventDestroy(start_native);
  cudaEventDestroy(stop_native);
  cudaEventDestroy(start_cublas);
  cudaEventDestroy(stop_cublas);
  cublasDestroy(handle);
  
  cudaFree(Cptr);
  cudaFree(Aptr);
  cudaFree(Bptr);
  cudaFree(Cptr_native);
  cudaFree(Cptr_cublas);
  
  free(Aptr_host);
  free(Bptr_host);
  free(Cptr_host);
  free(Cptr_native_host);
  free(Cptr_cublas_host);
}

template <typename T>
void gen_random_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}