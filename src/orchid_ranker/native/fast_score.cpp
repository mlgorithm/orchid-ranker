#include <torch/extension.h>

torch::Tensor fast_score(torch::Tensor user_vec, torch::Tensor item_matrix) {
    TORCH_CHECK(user_vec.dim() == 2, "user_vec must be [B, D]");
    TORCH_CHECK(item_matrix.dim() == 2, "item_matrix must be [N, D]");
    TORCH_CHECK(
        user_vec.size(1) == item_matrix.size(1),
        "user_vec and item_matrix must share the same embedding dimension");

    auto user = user_vec.contiguous();
    auto items = item_matrix.contiguous();
    return torch::mm(user, items.transpose(0, 1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_score", &fast_score, "Dense user·item scoring (prototype)");
}

