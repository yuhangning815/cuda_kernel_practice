#include <torch/extension.h>

torch::Tensor FlashAttention_base_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("FlashAttention_base_forward", torch::wrap_pybind_function(FlashAttention_base_forward), "FlashAttention_base_forward");
}