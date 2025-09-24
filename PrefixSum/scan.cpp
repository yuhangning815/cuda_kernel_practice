#include "scan.cuh"
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <cstdio>
#include <random>

void data_init(int *data, int N)
{
    std::uniform_int_distribution<> int_generator(-10, 100);
    std::default_random_engine rand_engine(time(nullptr));
    for (int i = 0; i < N; i++)
    {
        data[i] = int_generator(rand_engine);
    }
}

void results_check(int *a, int *b, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != b[i])
        {
            printf("results_check fail\n");
            exit(1);
        }
    }
}

void print_int_arr(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n");
}

int next_power_of_two(int x)
{
    int power = 1;
    while (power < x)
    {
        power *= 2;
    }
    return power;
}


int main(int argc, char **argv)
{
    warm_up();
    int nums[] ={2048, 100000, 10000000};
    int len = sizeof(nums) / sizeof(int);
    for (int i = 0; i < len; i++)
    {
        int N = nums[i];
        size_t arr_size = N * sizeof(int);
        int *data = (int *)malloc(arr_size);
        int *prefix_sum_cpu = (int *)malloc(arr_size);
        int *prefix_sum_gpu = (int *)malloc(arr_size);
        float total_cost, kernel_cost;
        data_init(data, N);
        printf("-------------------------- N = %d --------------------------\n", N);

        total_cost = scan_cpu(data, prefix_sum_cpu, N);
        printf("%35s - total: %10.5f ms\n", "scan_cpu", total_cost);

        std::tie(total_cost, kernel_cost) = sequential_scan_gpu(data, prefix_sum_gpu, N);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "sequential_scan_gpu", total_cost, kernel_cost);

        if (N <= MAX_ELEMENTS_PER_BLOCK)
        {
            std::tie(total_cost, kernel_cost) = parallel_block_scan_gpu(data, prefix_sum_gpu, N, false);
            results_check(prefix_sum_cpu, prefix_sum_gpu, N);
            printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_block_scan_gpu", total_cost,
                   kernel_cost);

            std::tie(total_cost, kernel_cost) = parallel_block_scan_gpu(data, prefix_sum_gpu, N, true);
            results_check(prefix_sum_cpu, prefix_sum_gpu, N);
            printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_block_scan_gpu with bcao", total_cost,
                   kernel_cost);
        }

        std::tie(total_cost, kernel_cost) = parallel_large_scan_gpu(data, prefix_sum_gpu, N, false);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_large_scan_gpu", total_cost, kernel_cost);

        std::tie(total_cost, kernel_cost) = parallel_large_scan_gpu(data, prefix_sum_gpu, N, true);
        results_check(prefix_sum_cpu, prefix_sum_gpu, N);
        printf("%35s - total: %10.5f ms    kernel: %10.5f ms\n", "parallel_large_scan_gpu with bcao", total_cost,
               kernel_cost);

        free(data);
        free(prefix_sum_cpu);
        free(prefix_sum_gpu);
        printf("\n");
    }
}



