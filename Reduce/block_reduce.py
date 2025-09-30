import time

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="block_all_reduce_lib",
    sources=["block_reduce.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: callable,
    values: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
):
    # if perf_func.__name__ == torch.sum.__name__:
    #     values = values.float() # for precision
    for i in range(warmup):
        out = perf_func(values)  # warmup
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        out = perf_func(values)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.item()
    print(f"{out_info:>25}: {out_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


Ss = [2048, 4096]
Ks = [2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 80)
    print(" " * 40 + f"S={S}, K={K}")
    values = torch.randn((S, K)).cuda().float()
    run_benchmark(lib.block_all_reduce_sum_f32_f32, values, "f32f32")
    run_benchmark(lib.block_all_reduce_sum_f32x4_f32, values, "f32x4f32")
    run_benchmark(torch.sum, values, "f32f32_pytorch")

    