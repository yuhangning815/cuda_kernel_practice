#include <torch/extension.h>

torch::Tensor FlashAttention_base_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor FlashAttention_optimize1_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor FlashAttention_wmma_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("FlashAttention_base_forward", torch::wrap_pybind_function(FlashAttention_base_forward), "FlashAttention_base_forward");
    m.def("FlashAttention_optimize1_forward", torch::wrap_pybind_function(FlashAttention_optimize1_forward), "FlashAttention_optimize1_forward");
    m.def("FlashAttention_wmma_forward", torch::wrap_pybind_function(FlashAttention_wmma_forward), "FlashAttention_wmma_forward");
}