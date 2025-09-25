// Tensor Core implementation - WMMA



// 对 d 也进行了切片
struct __align__(8) MD_Struct
{
    float m; // max val
    float d; // exp sum
};

struct MDStructOp
{
    __device__ __forceinline__ MD_Struct operator()(MD_Struct &a, MD_Struct &b)
    {
        MD_Struct ret;
        ret.m = max(a.m, b.m);
        ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
        return ret;
    }
};

__device__ __inline__ MD_Struct warpAllReduce(MD_Struct val)
{
    float tmp_m;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        tmp_m = max(val.m, __shfl_xor_sync(0xffffffff, val.m, mask, 32));
        val.d = val.d * __expf(val.m - tmp_m) + __shfl_xor_sync(0xffffffff, val.d, mask, 32) * __expf(__shfl_xor_sync(0xffffffff, val.m, mask, 32) - tmp_m);
        val.m = tmp_m;
    }
    return val;
}




void launchFlashAttentionKernel_wmma(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                    float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream)
{
    printf("in fa v3 launchFlashAttentionKernel_wmma\n");
    constexpr int Bc = 32;
    constexpr int Br = 64;
    constexpr int Wr = 32;
    constexpr int Wc = 16;
    constexpr int Bd = 32;
    // 让 Bd 等于 Bc 从而使得 QK 矩阵分片[Br, Bc] 与 QKV 矩阵分片[Br, Bd] 形状相同，方便排布
    assert(M % Bc == 0 && N % Br == 0 && d % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    /**
        __shared__ half s_Q_half[Br * Bd];
        __shared__ half s_K_half[Bc * Bd];
        __shared__ half s_V_half[Bc * Bd];
        __shared__ float s_S[Br * Bc];
        __shared__ half s_S_half[Br * Bc];
        __shared__ float s_O[Br * Bd];

        // 前一个 Bc 组的 l 和 m
        __shared__ MD_Struct row_ml_prev[Br];
        __shared__ MD_Struct row_ml[Br];
        __shared__ MD_Struct row_ml_new[Br];
        */

    const int sram_size = (Br * Bc + Br * Bd) * sizeof(float) + (Br * Bd + 2 * Bc * Bd + Br * Bc) * sizeof(half) + 8 * 3 * Br;
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);

    dim3 grid_dim(num_head, batch_size);
    dim3 block_dim(Bc * Br / (Wr * Wc) * 32);
    flashAttentionKernel_v3<Bc, Br, Wc, Wr><<<grid_dim, block_dim, 0, stream>>>(Q, K, V, O, l, m, N, M, d, softmax_scale);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


//Bc、Br 和 Bd 表示单个 block 每次处理的矩阵大小，而 Wc、Wr 则表示单个 warp 处理的矩阵大小

// s_Q_half[Br * Bd]：加载 Q 矩阵分片
// s_K_half[Bc * Bd]：加载 K 矩阵分片
// s_V_half[Bc * Bd]：加载 V 矩阵分片
// s_S[Br * Bc]：存储 Q*K^T 结果
// s_S_half[Br * Bc]：存储 softmax结果并作为左矩阵参与第二个矩阵乘法
// s_O[Br * Bd]：存储矩阵乘法  softmax(S)*V 的结果
// row_ml_prev[Br]：存储截至前一个分片的所有行的 m 和 l
// row_ml[Br]：存储当前分片处理的所有行的 m 和 l
// row_ml_new[Br]：存储截至当前分片的所有行的 m 和 l


// warp_col：当前 warp 负责计算的 gemm 矩阵分片 [Wr, Wc] 在 block 负责的矩阵分片 [Br, Bc] 中的列索引
// warp_row：当前 warp 负责计算的 gemm 矩阵分片 [Wr, Wc] 在 block 负责的矩阵分片 [Br, Bc] 中的行索引
// warp_id：当前 warp 在 block 中的索引
// lane_id：当前 thread 在 warp 中的索引
// WMITERS：由于 WMMA API 限制，一个 warp 单次不能完成 [Wr, Wc] 的 gemm 运算，需要进行循环处理，在行上的循环次数
// WNITERS：同上，在列上的循环次数
// WKITERS：同上，在 Bd 维度上的循环次数




template <int Br, int Bc, int Bd>
__device__ void loadQKFromGmemAndConvertToHalf(const float *Q, const float *K, const int d,
                                                half *s_Q, half *s_K, const int offset_q, const int offset_kv)
{
    int row_a, col_a;
    float4 tmp4;

#pragma unroll
    for (int i = (threadIdx.x << 2); i < Br * Bd; i += (blockDim.x << 2))
    {
        row_a = i / Bd;  // load Rectangular block 
        col_a = i % Bd;
        // Coalescing：就是DRAM 会被所有的请求集合在一起 然后 读取数据。 所以只要 读取的地址是连续的 就可以coalesce
        tmp4 = reinterpret_cast<const float4 *>(Q + offset_q + row_a * d + col_a)[0];  // float4 是4个float -> 16bytes; On the fly conversion is good !!!
        s_Q[row_a * Bd + col_a] = __float2half(tmp4.x);
        s_Q[row_a * Bd + col_a + 1] = __float2half(tmp4.y);
        s_Q[row_a * Bd + col_a + 2] = __float2half(tmp4.z);  
        s_Q[row_a * Bd + col_a + 3] = __float2half(tmp4.w);
    }

#pragma unroll
    for (int i = (threadIdx.x << 2); i < Bc * Bd; i += (blockDim.x << 2))
    {
        row_a = i / Bd;
        col_a = i % Bd;
        tmp4 = reinterpret_cast<const float4 *>(K + offset_kv + row_a * d + col_a)[0];
        s_K[row_a * Bd + col_a] = __float2half(tmp4.x);
        s_K[row_a * Bd + col_a + 1] = __float2half(tmp4.y);
        s_K[row_a * Bd + col_a + 2] = __float2half(tmp4.z);
        s_K[row_a * Bd + col_a + 3] = __float2half(tmp4.w);
    }
}


template <int Bd, int Wc, int Wr, typename T1, typename T2, typename T3>
__device__ void gemmFromSmemByWMMA(const half *__restrict__ s_Q, const half *__restrict__ s_K,
                                    T1 *q_frag, T2 *k_frag, T3 *acc_frag, const int warp_row, const int warp_col,
                                    const int WMITERS, const int WNITERS, const int WKITERS)
{
    using namespace nvcuda;

    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
        for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
        {
            int shm_offset = warp_row * Wr * Bd + wmidx * 16 * Bd + wkidx * 16;
            wmma::load_matrix_sync(q_frag[wmidx * WKITERS + wkidx], s_Q + shm_offset, Bd);
        }
    }

    for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
    {
        for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
        {
            int shm_offset = warp_col * Wc * Bd + wnidx * 16 * Bd + wkidx * 16;
            wmma::load_matrix_sync(k_frag[wnidx * WNITERS + wkidx], s_K + shm_offset, Bd);
        }
    }

    for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
    {
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
            {
                wmma::mma_sync(acc_frag[wmidx * WNITERS + wnidx], q_frag[wmidx * WKITERS + wkidx],
                                k_frag[wnidx * WKITERS + wkidx], acc_frag[wmidx * WNITERS + wnidx]);
            }
        }
    }
}


