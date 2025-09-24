/**
 * nvcc -std=c++17 -O2 -arch=sm_80 cute_copy_4x4.cu -o cute_copy_4x4
 *
 * Requires CUTE headers on your include path (CUTLASS 3+).
 */
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cstdio>

using namespace cute;

using _0 = Int<0>;
using _1 = Int<1>;
using _2 = Int<2>;
using _4 = Int<4>;

// Parent 4x4 tile in row-major (N contiguous)
using TileMN    = Layout<Shape<_4,_4>, Stride<_4,_1>>;

// 8 threads arranged as a 4×2 grid over the 4×4 tile.
// Linear thread id: tid = t_m*2 + t_n2  (Stride<_2,_1>)
using ThrLayout = Layout<Shape<_4,_2>, Stride<_2,_1>>;

// Each thread handles a 1×2 stripe along N (two consecutive columns).
// Inside the per-thread tile, stepping N (+j) advances by 1 element in memory.
using ValLayout = Layout<Shape<_1,_2>, Stride<_0,_1>>;

// A symmetric tiled-copy: same per-thread value shape for src and dst.
template <class T>
using TiledCopy1x2N = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<T>, T>{},   // elementwise copy atom
    ThrLayout{},                        // thread layout (Lt)
    ValLayout{},                        // src value layout (Lv_src)
    ValLayout{}));                      // dst value layout (Lv_dst)

// Kernel: copy 4x4 using 8 threads, each copying 2 consecutive along N.
__global__ void copy_4x4_8t_1x2N(const float* __restrict__ src,
                                 float* __restrict__ dst)
{
  // Build parent tensors over global memory
  Tensor gSrc = make_tensor(make_gmem_ptr(src), TileMN{});
  Tensor gDst = make_tensor(make_gmem_ptr(dst), TileMN{});

  // Partition per-thread using the tiled-copy descriptor
  auto tSrc = local_partition(gSrc, TiledCopy1x2N<float>{}, threadIdx.x);
  auto tDst = local_partition(gDst, TiledCopy1x2N<float>{}, threadIdx.x);

  // tSrc/tDst have shape (1,2). Load, print, and store.
  float v0 = tSrc(0,0);
  float v1 = tSrc(0,1);
  tDst(0,0) = v0;
  tDst(0,1) = v1;

  // Decode coordinates from the payload (val = 10*m + n)
  int m0 = int(v0) / 10, n0 = int(v0) % 10;
  int m1 = int(v1) / 10, n1 = int(v1) % 10;

  // (Optional) also show the 2D thread coordinate from ThrLayout
  auto tcoord = right_inverse(ThrLayout{})(threadIdx.x);
  int tm = int(get<0>(tcoord));
  int tn2 = int(get<1>(tcoord));

  printf("tx=%d thr=(%d,%d) shape=(%d,%d) -> (%d,%d)=%d  (%d,%d)=%d\n",
         int(threadIdx.x), tm, tn2,
         int(size<0>(shape(tSrc))), int(size<1>(shape(tSrc))),
         m0, n0, int(v0), m1, n1, int(v1));
}

int main() {
  constexpr int M = 4, N = 4, E = M*N;
  float h_src[E], h_dst[E];

  // Fill src with val = 10*m + n (easy to read coordinates)
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      h_src[m*N + n] = float(10*m + n);

  float *d_src = nullptr, *d_dst = nullptr;
  cudaMalloc(&d_src, E*sizeof(float));
  cudaMalloc(&d_dst, E*sizeof(float));
  cudaMemcpy(d_src, h_src, E*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_dst, 0, E*sizeof(float));

  // One block of 8 threads
  copy_4x4_8t_1x2N<<<1, 8>>>(d_src, d_dst);
  cudaDeviceSynchronize();

  // Fetch the result to prove the copy worked
  cudaMemcpy(h_dst, d_dst, E*sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nResult matrix (row-major):\n");
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      printf("%3d ", int(h_dst[m*N + n]));
    }
    printf("\n");
  }

  cudaFree(d_src);
  cudaFree(d_dst);
  return 0;
}


