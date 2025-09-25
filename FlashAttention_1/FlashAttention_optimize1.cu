/**
    * grid( num_head, batch_size )
    * block( BLOCK_SIZE )
    * Q\O: [batch_size, num_head, N, d]
    * K\V: [batch_size, num_head, M, d]
    * l: [batch_size, num_head, N, 1]
    * m: [batch_size, num_head, N, 1]
    */



    
template <int Bc>
__global__ void flashAttentionKernel_v1(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                        float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                        const int N, const int M, const int d, const float softmax_scale)
{
    const int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
    const int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * M * d;
    const int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

    extern __shared__ float s_ptr[];
    float *s_Q = s_ptr;        // [1, d]
    float *s_K = s_Q + d;      // [Bc, d]
    float *s_V = s_K + Bc * d; // [Bc, d]
    float *s_S = s_V + Bc * d; // [1, Bc]

    __shared__ MD_F row_ml_prev;

    // 对 K|V 在 M 维度分组，每组长度为 Bc，共分为 Tc 组. 一个thread负责一行
    for (int i = 0; i < M; i += Bc)
    {
        // 加载 [Bc, d] 数据到 s_K 和 s_Vs_V； 
        // 优化1：这里是 coalesced access on GMEM！！！
        for (int j = threadIdx.x; j < Bc * d; j += blockDim.x)
        {
            s_K[j] = K[kv_offset + i * d + j];
            s_V[j] = V[kv_offset + i * d + j];
        }
        __syncthreads();

        // 遍历 Q 的 N 列，每次处理一列
        for (int j = 0; j < N; ++j)
        {
            // 加载 1 列数据到 s_Q
            for (int k = threadIdx.x; k < d; k += blockDim.x)
            {
                s_Q[k] = Q[qo_offset + j * d + k];
            }
            // 上一个 Bc 组结束时每行的 m 和 l
            if (threadIdx.x == 0)
            {
                row_ml_prev = {m[lm_offset + j], l[lm_offset + j]};
            }
            __syncthreads();

            // 存储当前第 j 行的 l 和 m
            MD_F row_ml = {-1e20f, 0.0f};
            // 遍历 K^T 的 Bc 列
            for (int k = 0; k < Bc; ++k)
            {
                MD_F tmp_ml = {0.0f, 1.0f};
                // 计算 QK^T
                for (int x = threadIdx.x; x < d; x += blockDim.x)
                {
                    tmp_ml.m += s_Q[x] * s_K[k * d + x];
                }
                tmp_ml.m *= softmax_scale;
                __syncthreads();

                // 存储第 j 行的 Q 向量与第 k 列的 s_K 向量的内积, QK^T 矩阵当前第 j 列的值
                tmp_ml.m = blockAllReduceSum<float>(tmp_ml.m);
                row_ml = MDFOp()(row_ml, tmp_ml);
                if (threadIdx.x == 0) { s_S[k] = tmp_ml.m; }
                __syncthreads();
            }
            __syncthreads();

            MD_F row_ml_new = MDFOp()(row_ml_prev, row_ml);

            // 遍历矩阵 O 的 d 维度，O = softmax(QK^T)V
            for (int k = threadIdx.x; k < d; k += blockDim.x)
            {
                float pv = 0.0f;
                for (int x = 0; x < Bc; ++x)
                {
                    pv += __expf(s_S[x] - row_ml.m) * s_V[x * d + k];   // 优化：memory coalescing + resolve bank conflict
                }
                // 更新 O 矩阵
                O[qo_offset + j * d + k] = 1.0f / row_ml_new.d * (row_ml_prev.d * __expf(row_ml_prev.m - row_ml_new.m) * O[qo_offset + j * d + k] + __expf(row_ml.m - row_ml_new.m) * pv);
            }

            // 写入当前 Bc 组的 l 和 m
            if (threadIdx.x == 0)
            {
                l[lm_offset + j] = row_ml_new.d;
                m[lm_offset + j] = row_ml_new.m;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

void launchFlashAttentionKernel_v1(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                    float *__restrict__ O, float *__restrict__ l, float *__restrict__ m,
                                    const int batch_size, const int num_head, const int N, const int M, const int d, cudaStream_t stream)
{
    constexpr int Bc = 4;
    assert(M % Bc == 0);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    const int sram_size = (d + 2 * Bc * d + Bc) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %g KB, requested shared memory: %g KB \n", max_sram_size / 1024.0f, sram_size / 1024.0f);

    constexpr int block_size = 128;
    dim3 grid_dim(num_head, batch_size);
    dim3 block_dim(block_size);
    flashAttentionKernel_v1<Bc><<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, l, m, N, M, d, softmax_scale);
}