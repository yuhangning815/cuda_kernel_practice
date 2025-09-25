#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>




// 问题：



__global__
void FlashAttention_base_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {

    // Q,K,V: [batch_size, num_head, N, d]
    // Scores =  Q · Kᵀ / √dhead                # (B, H, L, L)  
    // Z      =  Attn · V                                 # (B, H, L, dhead)

    // TODO: determine Bc, Br dynamically
    //  K，V的每个分块为 【Bc * d】，Q的每个分块为 【Br * d】
    // blockdim = Bc，一个thread负责一行

    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = num_head
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];


    // blockdim = Bc, Tc * Bc = N; 一个Block要每Blockdim个行的处理
    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM - 每个thread 一列一列的load；matrix dim = 【Bc, d】 
        // 问题1 ： L2 cache miss：每个thread 之间读的数据差的太远！要within 128/256 Bytes比较好！
        // 问题2： non-coalesced access for GMEM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];       
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }

            float row_m_prev = m[lm_offset + (Br * i) + tx];   // m = max 
            float row_l_prev = l[lm_offset + (Br * i) + tx];   // l = row_sum

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            // 做GEMM - 每个Q 行有个thread 要for loop乘 K 的每一行
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;     // S的size是 【Bc, Bc】，是正确的，不是partial sum

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);   // 问题3： Bank conflict again！if bc 是32的倍数
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM; 
            // 【Bc, Bc】 * 【Bc, d】 = 【Bc, d】
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}




torch::Tensor FlashAttention_base_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Q,K,V: [batch_size, num_head, N, d]

    // TODO: determine Bc, Br dynamically
    //  K，V的每个分块为 【Bc * d】，Q的每个分块为 【Br * d】
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    // 用于 streaming softmax -> 每一行维护一个variable
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads       -> 一个block处理一个batch的一个head
    dim3 block_dim(Bc);  // Bc threads per block            -> 1D block ！！！

    FlashAttention_base_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}