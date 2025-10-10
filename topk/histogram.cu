#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_BINS 512
#define BLOCK_SIZE 512

// CPU histogram implementation for reference
void histogram_cpu(const unsigned char* data, int* hist, int n) {
    // Initialize histogram to zero
    for (int i = 0; i < NUM_BINS; i++) {
        hist[i] = 0;
    }
    
    // Count occurrences
    for (int i = 0; i < n; i++) {
        hist[data[i]]++;
    }
}

// Naive GPU histogram - direct atomic operations to global memory
__global__ void histogram_naive_gpu(const unsigned char* data, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        atomicAdd(&hist[data[idx]], 1);
    }
}

// Optimized GPU histogram - using shared memory for local histograms
// KEY OPTIMIZATION: Two-phase approach to reduce global memory contention
__global__ void histogram_shared_gpu(const unsigned char* data, int* hist, int n) {
    __shared__ int hist_shared[NUM_BINS];
    
  
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        hist_shared[i] = 0;
    }
    __syncthreads();  // Wait for all threads to finish initialization
    
    // PHASE 1: Each thread processes its data and updates LOCAL shared histogram
    // Benefits: Fast atomics, only threads in THIS block compete
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;  // Total number of threads across all blocks
    
    // Grid-stride loop: each thread processes multiple elements
    // Example: With 100M elements and 100K threads, each processes ~1000 elements
    for (int i = idx; i < n; i += stride) {
        // Atomic operation on SHARED memory (fast!)
        // Only 256 threads in this block compete, not all threads globally
        atomicAdd(&hist_shared[data[i]], 1);
    }
    __syncthreads();  // Wait for all threads to finish local accumulation
    
    // PHASE 2: Merge local histogram to global histogram
    // This happens only ONCE per block, dramatically reducing global atomics
    // Instead of N atomic operations to global, we have only 256 per block
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (hist_shared[i] > 0) {
            // Atomic operation on GLOBAL memory (slow, but infrequent)
            // With 409,600 blocks, this is ~100K atomics vs 100M in naive version
            atomicAdd(&hist[i], hist_shared[i]);
        }
    }
}

// Advanced optimized version with privatization to reduce conflicts
// KEY OPTIMIZATION: Use FEWER blocks, but each thread does MORE work
// This dramatically reduces the final merge contention in Phase 2
__global__ void histogram_privatized_gpu(const unsigned char* data, int* hist, int n) {
    // Shared memory for local histogram (same as before)
    __shared__ int hist_shared[NUM_BINS];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        hist_shared[i] = 0;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;  // Total threads in grid
    
    // CRITICAL DIFFERENCE: We launch with fewer blocks (e.g., 128 instead of 409,600)
    for (int i = idx; i < n; i += stride) {
        unsigned char value = data[i];
        atomicAdd(&hist_shared[value], 1);  // we need atomic adds for shared memory
    }
    __syncthreads();
    
    // fewer blocks
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (hist_shared[i] > 0) {
            atomicAdd(&hist[i], hist_shared[i]);  // Much less contention now!
        }
    }
}

// Generate random data
void generate_random_data(unsigned char* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = rand() * 19 % NUM_BINS;
    }
}

// Verify histogram correctness
bool verify_histogram(const int* hist_ref, const int* hist_test, int num_bins) {
    for (int i = 0; i < num_bins; i++) {
        if (hist_ref[i] != hist_test[i]) {
            printf("Mismatch at bin %d: CPU=%d, GPU=%d\n", i, hist_ref[i], hist_test[i]);
            return false;
        }
    }
    return true;
}

// Print histogram summary
void print_histogram_summary(const int* hist, int num_bins, const char* name) {
    int total = 0;
    int min_count = hist[0];
    int max_count = hist[0];
    
    for (int i = 0; i < num_bins; i++) {
        total += hist[i];
        if (hist[i] < min_count) min_count = hist[i];
        if (hist[i] > max_count) max_count = hist[i];
    }
    
    printf("%s - Total: %d, Min: %d, Max: %d, Avg: %.2f\n", 
           name, total, min_count, max_count, (float)total / num_bins);
    
    // Print first 10 bins
    printf("First 10 bins: ");
    for (int i = 0; i < 10 && i < num_bins; i++) {
        printf("%d ", hist[i]);
    }
    printf("\n");
}

int main() {
    srand(12345);
    
    // Data size
    const int N = 100 * 1024 * 1024; // 100M elements
    printf("Histogram computation for %d elements\n", N);
    printf("Number of bins: %d\n\n", NUM_BINS);
    
    // Allocate host memory
    unsigned char* h_data = (unsigned char*)malloc(N * sizeof(unsigned char));
    int* h_hist_cpu = (int*)malloc(NUM_BINS * sizeof(int));
    int* h_hist_naive = (int*)malloc(NUM_BINS * sizeof(int));
    int* h_hist_shared = (int*)malloc(NUM_BINS * sizeof(int));
    int* h_hist_privatized = (int*)malloc(NUM_BINS * sizeof(int));
    
    // Generate random data
    printf("Generating random data...\n");
    generate_random_data(h_data, N);
    
    // Allocate device memory
    unsigned char* d_data;
    int* d_hist_naive;
    int* d_hist_shared;
    int* d_hist_privatized;
    
    cudaMalloc(&d_data, N * sizeof(unsigned char));
    cudaMalloc(&d_hist_naive, NUM_BINS * sizeof(int));
    cudaMalloc(&d_hist_shared, NUM_BINS * sizeof(int));
    cudaMalloc(&d_hist_privatized, NUM_BINS * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_data, h_data, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // ==================== CPU Implementation ====================
    printf("\n==================== CPU Implementation ====================\n");
    clock_t start_cpu = clock();
    histogram_cpu(h_data, h_hist_cpu, N);
    clock_t end_cpu = clock();
    double time_cpu = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU time: %.3f ms\n", time_cpu);
    print_histogram_summary(h_hist_cpu, NUM_BINS, "CPU Histogram");
    
    // ==================== GPU Naive Implementation ====================
    printf("\n==================== GPU Naive Implementation ====================\n");
    cudaMemset(d_hist_naive, 0, NUM_BINS * sizeof(int));
    
    dim3 block_naive(BLOCK_SIZE);
    dim3 grid_naive((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);
    
    cudaEventRecord(start_naive);
    histogram_naive_gpu<<<grid_naive, block_naive>>>(d_data, d_hist_naive, N);
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);
    
    float time_naive = 0;
    cudaEventElapsedTime(&time_naive, start_naive, stop_naive);
    
    cudaMemcpy(h_hist_naive, d_hist_naive, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("GPU Naive time: %.3f ms\n", time_naive);
    printf("Speedup vs CPU: %.2fx\n", time_cpu / time_naive);
    print_histogram_summary(h_hist_naive, NUM_BINS, "GPU Naive");
    
    if (verify_histogram(h_hist_cpu, h_hist_naive, NUM_BINS)) {
        printf("✓ GPU Naive result is CORRECT\n");
    } else {
        printf("✗ GPU Naive result is INCORRECT\n");
    }
    
    // ==================== GPU Shared Memory Implementation ====================
    printf("\n==================== GPU Shared Memory Implementation ====================\n");
    cudaMemset(d_hist_shared, 0, NUM_BINS * sizeof(int));
    
    dim3 block_shared(BLOCK_SIZE);
    dim3 grid_shared((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start_shared, stop_shared;
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);
    
    cudaEventRecord(start_shared);
    histogram_shared_gpu<<<grid_shared, block_shared>>>(d_data, d_hist_shared, N);
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);
    
    float time_shared = 0;
    cudaEventElapsedTime(&time_shared, start_shared, stop_shared);
    
    cudaMemcpy(h_hist_shared, d_hist_shared, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("GPU Shared Memory time: %.3f ms\n", time_shared);
    printf("Speedup vs CPU: %.2fx\n", time_cpu / time_shared);
    printf("Speedup vs Naive GPU: %.2fx\n", time_naive / time_shared);
    print_histogram_summary(h_hist_shared, NUM_BINS, "GPU Shared");
    
    if (verify_histogram(h_hist_cpu, h_hist_shared, NUM_BINS)) {
        printf("✓ GPU Shared result is CORRECT\n");
    } else {
        printf("✗ GPU Shared result is INCORRECT\n");
    }
    
    // ==================== GPU Privatized Implementation ====================
    printf("\n==================== GPU Privatized Implementation ====================\n");
    cudaMemset(d_hist_privatized, 0, NUM_BINS * sizeof(int));
    
    // Use fewer blocks for privatized version to reduce global memory conflicts
    dim3 block_privatized(BLOCK_SIZE);
    int num_blocks = 128; // Tuned for better performance
    dim3 grid_privatized(num_blocks);
    
    cudaEvent_t start_priv, stop_priv;
    cudaEventCreate(&start_priv);
    cudaEventCreate(&stop_priv);
    
    cudaEventRecord(start_priv);
    histogram_privatized_gpu<<<grid_privatized, block_privatized>>>(d_data, d_hist_privatized, N);
    cudaEventRecord(stop_priv);
    cudaEventSynchronize(stop_priv);
    
    float time_priv = 0;
    cudaEventElapsedTime(&time_priv, start_priv, stop_priv);
    
    cudaMemcpy(h_hist_privatized, d_hist_privatized, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("GPU Privatized time: %.3f ms\n", time_priv);
    printf("Speedup vs CPU: %.2fx\n", time_cpu / time_priv);
    printf("Speedup vs Naive GPU: %.2fx\n", time_naive / time_priv);
    print_histogram_summary(h_hist_privatized, NUM_BINS, "GPU Privatized");
    
    if (verify_histogram(h_hist_cpu, h_hist_privatized, NUM_BINS)) {
        printf("✓ GPU Privatized result is CORRECT\n");
    } else {
        printf("✗ GPU Privatized result is INCORRECT\n");
    }
    
    // ==================== Performance Summary ====================
    printf("\n==================== Performance Summary ====================\n");
    printf("Data size: %d elements (%.2f MB)\n", N, N / (1024.0 * 1024.0));
    printf("%-25s: %8.3f ms\n", "CPU", time_cpu);
    printf("%-25s: %8.3f ms (%.2fx speedup)\n", "GPU Naive", time_naive, time_cpu / time_naive);
    printf("%-25s: %8.3f ms (%.2fx speedup)\n", "GPU Shared Memory", time_shared, time_cpu / time_shared);
    printf("%-25s: %8.3f ms (%.2fx speedup)\n", "GPU Privatized", time_priv, time_cpu / time_priv);
    printf("=============================================================\n");
    
    // Cleanup
    free(h_data);
    free(h_hist_cpu);
    free(h_hist_naive);
    free(h_hist_shared);
    free(h_hist_privatized);
    
    cudaFree(d_data);
    cudaFree(d_hist_naive);
    cudaFree(d_hist_shared);
    cudaFree(d_hist_privatized);
    
    cudaEventDestroy(start_naive);
    cudaEventDestroy(stop_naive);
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);
    cudaEventDestroy(start_priv);
    cudaEventDestroy(stop_priv);
    
    return 0;
}

