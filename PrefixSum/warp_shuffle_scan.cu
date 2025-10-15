/*
 * Prefix Sum (Exclusive Scan) Using Warp Shuffle Operations - FLOAT VERSION
 * 
 * This implementation uses CUDA warp shuffle intrinsics (__shfl_up_sync) to
 * perform efficient prefix sum computation on float arrays. The algorithm works
 * hierarchically:
 * 
 * 1. Warp-level scan: Each warp (32 threads) computes its local prefix sum
 *    using shuffle operations. This avoids shared memory and bank conflicts.
 * 
 * 2. Block-level scan: For blocks with multiple warps, we:
 *    - Compute local prefix sum within each warp
 *    - Store warp totals in shared memory
 *    - Use the first warp to compute prefix sum of warp totals
 *    - Add the warp offset to each element
 *    - Store the block level sum to block_sums and output it
 * 
 * 3. Grid-level scan: For large arrays with multiple blocks:
 *    - Compute prefix sum within each block
 *    - Recursively compute prefix sum of block totals
 *    - Add block offsets to all elements
 * 
 * Key Features:
 * - Uses warp shuffle for intra-warp communication (fast, no shared memory)
 * - Hierarchical approach scales to arbitrary array sizes
 * - Computes exclusive scan (output[i] = sum of input[0..i-1])
 * - Handles float arrays from size 1 to ~1 million elements
 * - For larger arrays, consider using double precision
 * 
 * Performance Notes:
 * - Warp shuffle operations are very fast (1-2 cycles)
 * - No shared memory bank conflicts for warp-level operations
 * - Recursive approach has O(log N) depth with respect to blocks
 * - Single precision floats have limited accuracy for very large accumulations
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>

#define WARP_SIZE 32
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            printf("CUDA Error: \n");                                         \
            printf("    File:       %s\n", __FILE__);                         \
            printf("    Line:       %d\n", __LINE__);                         \
            printf("    Error Code: %d\n", err);                              \
            printf("    Error Text: %s\n", cudaGetErrorString(err));          \
            exit(1);                                                          \
        }                                                                     \
    } while (0)


// Warp-level prefix sum using shuffle operations (inclusive scan)
__device__ __forceinline__ float warp_prefix_sum_inclusive(float val, int lane_id) {
    float inclusive = val;
    
    // Use warp shuffle to compute prefix sum
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float n = __shfl_up_sync(0xFFFFFFFF, inclusive, offset);
        if (lane_id >= offset) {
            inclusive += n;
        }
    }
    
    return inclusive;
}

// Warp-level prefix sum using shuffle operations (exclusive scan)
__device__ __forceinline__ float warp_prefix_sum_exclusive(float val, int lane_id) {
    // Perform inclusive scan first
    float inclusive = warp_prefix_sum_inclusive(val, lane_id);
    
    // Convert to exclusive scan by shifting
    float exclusive = __shfl_up_sync(0xFFFFFFFF, inclusive, 1);
    if (lane_id == 0) {
        exclusive = 0.0f;
    }
    
    return exclusive;
}



// Single warp kernel for arrays up to 32 elements
__global__ void warp_shuffle_scan_kernel_small(float *data, float *prefix_sum, int N) {
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    
    float val = (tid < N) ? data[tid] : 0.0f;
    float result = warp_prefix_sum_exclusive(val, lane_id);
    
    if (tid < N) {
        prefix_sum[tid] = result;
    }
}

// Multi-warp kernel for larger arrays
__global__ void warp_shuffle_scan_kernel(float *data, float *prefix_sum, int N) {
    extern __shared__ float shared_sums[];
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Step 1: Each warp computes its local prefix sum
    float val = (global_tid < N) ? data[global_tid] : 0.0f;
    float local_sum = warp_prefix_sum_exclusive(val, lane_id);
    
    // Get the total sum for this warp (last element of inclusive scan)
    float warp_total = warp_prefix_sum_inclusive(val, lane_id);
    warp_total = __shfl_sync(0xFFFFFFFF, warp_total, WARP_SIZE - 1);
    
    // Step 2: Last thread in each warp writes the warp total to shared memory
    if (lane_id == WARP_SIZE - 1) {
        shared_sums[warp_id] = warp_total;
    }
    __syncthreads();
    
    // Step 3: First warp computes prefix sum of warp totals
    float warp_offset = 0.0f;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    if (warp_id == 0) {
        float warp_sum_val = (tid < num_warps) ? shared_sums[tid] : 0.0f;
        float warp_sum_prefix = warp_prefix_sum_exclusive(warp_sum_val, lane_id);
        if (tid < num_warps) {
            shared_sums[tid] = warp_sum_prefix;
        }
    }
    __syncthreads();
    
    // Step 4: Add warp offset to local prefix sum
    warp_offset = shared_sums[warp_id];
    local_sum += warp_offset;
    
    // Write result
    if (global_tid < N) {
        prefix_sum[global_tid] = local_sum;
    }
}

// Multi-block kernel with block-level aggregation
__global__ void warp_shuffle_scan_large_kernel(float *data, float *prefix_sum, int N, float *block_sums) {
    extern __shared__ float shared_sums[];
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int global_tid = blockIdx.x * blockDim.x + tid;
    
    // Step 1: Each warp computes its local prefix sum
    float val = (global_tid < N) ? data[global_tid] : 0.0f;
    float local_sum = warp_prefix_sum_exclusive(val, lane_id);
    
    // Get the total sum for this warp
    float warp_total = warp_prefix_sum_inclusive(val, lane_id);
    warp_total = __shfl_sync(0xFFFFFFFF, warp_total, WARP_SIZE - 1);
    
    // Step 2: Last thread in each warp writes the warp total to shared memory
    if (lane_id == WARP_SIZE - 1) {
        shared_sums[warp_id] = warp_total;
    }
    __syncthreads();
    
    // Step 3: First warp computes prefix sum of warp totals
    float warp_offset = 0.0f;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    if (warp_id == 0) {
        float warp_sum_val = (tid < num_warps) ? shared_sums[tid] : 0.0f;
        float warp_sum_prefix = warp_prefix_sum_exclusive(warp_sum_val, lane_id);
        if (tid < num_warps) {
            shared_sums[tid] = warp_sum_prefix;
        }
    }
    __syncthreads();
    
    // Step 4: Add warp offset to local prefix sum
    warp_offset = shared_sums[warp_id];
    local_sum += warp_offset;
    
    // Step 5: Last thread in block saves block total, and output it
    if (block_sums != nullptr && tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = local_sum + ((global_tid < N) ? data[global_tid] : 0.0f);
    }
    
    // Write result
    if (global_tid < N) {
        prefix_sum[global_tid] = local_sum;
    }
}

// Add block offsets kernel - only used in large kernels 
__global__ void add_block_offsets_kernel(float *prefix_sum, float *block_offsets, int N) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid < N && blockIdx.x > 0) {
        prefix_sum[global_tid] += block_offsets[blockIdx.x];
    }
}

// Host function for small arrays (single block)
void warp_shuffle_scan_small(float *data, float *prefix_sum, int N) {
    float *d_data, *d_prefix_sum;
    size_t size = N * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, size));
    CUDA_CHECK(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
    
    int threads = (N + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    threads = (threads < 32) ? 32 : threads;
    
    if (threads <= WARP_SIZE) {
        warp_shuffle_scan_kernel_small<<<1, threads>>>(d_data, d_prefix_sum, N);
    } else {
        int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
        int shared_mem = num_warps * sizeof(float);
        warp_shuffle_scan_kernel<<<1, threads, shared_mem>>>(d_data, d_prefix_sum, N);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(prefix_sum, d_prefix_sum, size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_prefix_sum));
}

// Recursive helper for computing prefix sum on device
void recursive_prefix_sum(float *d_input, float *d_output, int N) {
    if (N <= WARP_SIZE) {
        warp_shuffle_scan_kernel_small<<<1, WARP_SIZE>>>(d_input, d_output, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else if (N <= 1024) {
        int threads = (N + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        int num_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
        int shared_mem = num_warps * sizeof(float);
        warp_shuffle_scan_kernel<<<1, threads, shared_mem>>>(d_input, d_output, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const int BLOCK_SIZE = 512;
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        float *d_block_sums, *d_block_offsets;
        CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_offsets, num_blocks * sizeof(float)));
        
        // Compute prefix sum within each block
        int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        int shared_mem = num_warps * sizeof(float);

        // do prefix sum on each block
        warp_shuffle_scan_large_kernel<<<num_blocks, BLOCK_SIZE, shared_mem>>>(
            d_input, d_output, N, d_block_sums);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // do prefix sum on block-level sum
        recursive_prefix_sum(d_block_sums, d_block_offsets, num_blocks);
        
        // Add block-level sum 
        add_block_offsets_kernel<<<num_blocks, BLOCK_SIZE>>>(d_output, d_block_offsets, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaFree(d_block_sums));
        CUDA_CHECK(cudaFree(d_block_offsets));
    }
}

// Host function for large arrays (multiple blocks)
void warp_shuffle_scan_large(float *data, float *prefix_sum, int N) {
    float *d_data, *d_prefix_sum;
    size_t size = N * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_prefix_sum, size));
    CUDA_CHECK(cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice));
    
    recursive_prefix_sum(d_data, d_prefix_sum, N);
    
    CUDA_CHECK(cudaMemcpy(prefix_sum, d_prefix_sum, size, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_prefix_sum));
}

// CPU reference implementation
void scan_cpu(float *data, float *prefix_sum, int N) {
    prefix_sum[0] = 0.0f;
    for (int i = 1; i < N; i++) {
        prefix_sum[i] = prefix_sum[i - 1] + data[i - 1];
    }
}

// Verification function with floating point tolerance
bool verify_result(float *cpu_result, float *gpu_result, int N) {
    // Scale epsilon with problem size due to accumulation of floating point errors
    const float BASE_EPSILON = 1e-4f;
    const float EPSILON = BASE_EPSILON * sqrtf((float)N / 1000.0f);
    
    int error_count = 0;
    const int MAX_ERRORS_TO_PRINT = 5;
    
    for (int i = 0; i < N; i++) {
        float diff = fabsf(cpu_result[i] - gpu_result[i]);
        float relative_error = (fabsf(cpu_result[i]) > BASE_EPSILON) ? diff / fabsf(cpu_result[i]) : diff;
        
        if (relative_error > EPSILON) {
            if (error_count < MAX_ERRORS_TO_PRINT) {
                printf("Mismatch at index %d: CPU = %.6f, GPU = %.6f (diff = %.6f, rel_err = %.6f, threshold = %.6f)\n", 
                       i, cpu_result[i], gpu_result[i], diff, relative_error, EPSILON);
            }
            error_count++;
        }
    }
    
    if (error_count > MAX_ERRORS_TO_PRINT) {
        printf("... and %d more errors\n", error_count - MAX_ERRORS_TO_PRINT);
    }
    
    if (error_count > 0) {
        printf("Total errors: %d out of %d (%.4f%%)\n", error_count, N, 100.0f * error_count / N);
    }
    
    return error_count == 0;
}

// Initialize data
void data_init(float *data, int N) {
    std::uniform_real_distribution<float> dist(0.1f, 10.0f);
    std::default_random_engine engine(42);
    for (int i = 0; i < N; i++) {
        data[i] = dist(engine);
    }
}

// Warm up GPU
__global__ void warmup_kernel(float *data) {
    int tid = threadIdx.x;
    data[tid] += (float)tid;
}

void warmup() {
    int N = 512;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    for (int i = 0; i < 5; i++) {
        warmup_kernel<<<1, N>>>(d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    CUDA_CHECK(cudaFree(d_data));
}

// Print array (for debugging)
void print_array(const char *name, float *arr, int N) {
    printf("%s: ", name);
    for (int i = 0; i < N; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main() {
    printf("=== WARP SHUFFLE PREFIX SUM TEST (FLOAT) ===\n");
    printf("Note: Single-precision floats are suitable for arrays up to ~1M elements.\n");
    printf("For larger arrays, use double precision to avoid accumulation errors.\n\n");
    
    // Warm up GPU
    warmup();
    
    // Test different sizes (limited to 1M for float precision)
    int test_sizes[] = {1024, 2048, 10000, 100000, 1000000};
    int num_tests = sizeof(test_sizes) / sizeof(int);
    
    for (int t = 0; t < num_tests; t++) {
        int N = test_sizes[t];
        printf("Testing N = %d\n", N);
        
        // Allocate memory
        float *data = (float*)malloc(N * sizeof(float));
        float *cpu_result = (float*)malloc(N * sizeof(float));
        float *gpu_result = (float*)malloc(N * sizeof(float));
        
        // Initialize data
        data_init(data, N);
        
        // CPU reference
        auto cpu_start = std::chrono::high_resolution_clock::now();
        scan_cpu(data, cpu_result, N);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        
        // GPU computation
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        if (N <= 1024) {
            warp_shuffle_scan_small(data, gpu_result, N);
        } else {
            warp_shuffle_scan_large(data, gpu_result, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float gpu_time;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
        
        // Verify
        bool correct = verify_result(cpu_result, gpu_result, N);
        
        if (correct) {
            printf("  ✓ PASSED - CPU: %.5f ms, GPU: %.5f ms", cpu_time, gpu_time);
            if (cpu_time > 0.001) {
                printf(" (Speedup: %.2fx)\n", cpu_time / gpu_time);
            } else {
                printf("\n");
            }
        } else {
            printf("  ✗ FAILED\n");
            if (N <= 32) {
                print_array("Data", data, N);
                print_array("CPU ", cpu_result, N);
                print_array("GPU ", gpu_result, N);
            }
        }
        
        // Cleanup
        free(data);
        free(cpu_result);
        free(gpu_result);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    printf("\n=== ALL TESTS COMPLETED ===\n");
    return 0;
}

