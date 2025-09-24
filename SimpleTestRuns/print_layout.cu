#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
using namespace cute;

constexpr int BLK_M = 4;   // 4x4 tile
constexpr int BLK_N = 4;



template <class ThrCoordTensor, class ThrDataTensor>
__device__ void print_thread_once_serialized(const ThrCoordTensor& tCoord,
                                             const ThrDataTensor& tData,
                                             int tx) {
  // gather up to 2 elements (modify as needed)
  int r0 = int(get<0>(tCoord(0,0))), c0 = int(get<1>(tCoord(0,0))), v0 = int(tData(0,0));
  int r1 = int(get<0>(tCoord(0,1))), c1 = int(get<1>(tCoord(0,1))), v1 = int(tData(0,1));
  for (int who = 0; who < blockDim.x; ++who) {
    __syncthreads();
    if (tx == who) {
      printf("tx=%d shape=(%d,%d) -> (%d,%d)=%d  (%d,%d)=%d\n",
             tx, int(size<0>(tCoord)), int(size<1>(tCoord)), r0,c0,v0, r1,c1,v1);
    }
  }
  __syncthreads();
}

__device__ auto make_coord_tile() {
  return make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_N>{}));
}

__global__ void k_partition_debug(const int* __restrict__ A,
                                  int* __restrict__ owner,
                                  int M, int N)
{
  // Build row-major tensors on the 4x4 tile at (0,0)
  auto mA = make_tensor(make_gmem_ptr(A),
                        make_layout(make_shape(M, N), GenRowMajor{}));  // (M,N)
  auto mO = make_tensor(make_gmem_ptr(owner),
                        make_layout(make_shape(M, N), GenRowMajor{}));  // (M,N)
  auto gA = local_tile(mA, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(0,0));
  auto gO = local_tile(mO, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(0,0));

  if (threadIdx.x == 0){
    auto cA = make_coord_tile();  // coordinate tensor
    auto c00 = cA(make_coord(0,0));
    auto c01 = cA(make_coord(0,1));

    // Extract tuple components to print
    printf("c00=(%d,%d)  c01=(%d,%d)\n",
          (int)get<0>(c00), (int)get<1>(c00),
          (int)get<0>(c01), (int)get<1>(c01));
  }


  // Outer thread map on the 2D tile:
  auto thread_map = make_layout(make_shape(Int<1>{}, Int<2>{}), GenRowMajor{});

  __syncthreads();

  int tx = threadIdx.x;
  auto tO  = local_tile(gO, thread_map, tx);

  tO(make_coord(0,0)) = tx;
  tO(make_coord(0,1)) = tx;
}



static inline void check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main() {
  const int M = BLK_M, N = BLK_N;
  const int elems = M * N;

  // Host init: A[r,c] = 10*r + c
  std::vector<int> hA(elems), hOwner(elems, -1);
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c)
      hA[r*N + c] = 10*r + c;

  int *dA = nullptr, *dOwner = nullptr;
  check(cudaMalloc(&dA, elems * sizeof(int)), "cudaMalloc dA");
  check(cudaMalloc(&dOwner, elems * sizeof(int)), "cudaMalloc dOwner");
  check(cudaMemcpy(dA, hA.data(), elems * sizeof(int), cudaMemcpyHostToDevice), "cpy A");
  check(cudaMemset(dOwner, 0xFF, elems * sizeof(int)), "memset owner");  // -1 pattern

  printf("=== DEBUG partition (4x4, 8 threads, 1x2 per thread) ===\n");
  k_partition_debug<<<1, 8>>>(dA, dOwner, M, N);
  check(cudaGetLastError(), "kernel launch");
  check(cudaDeviceSynchronize(), "sync");

  check(cudaMemcpy(hOwner.data(), dOwner, elems * sizeof(int), cudaMemcpyDeviceToHost), "cpy owner");
  printf("\nOwner map (element -> thread id):\n");
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      printf("%2d ", hOwner[r*N + c]);
    }
    printf("\n");
  }

  cudaFree(dA);
  cudaFree(dOwner);
  return 0;
}
