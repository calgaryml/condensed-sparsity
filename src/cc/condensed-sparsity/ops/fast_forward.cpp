#include <torch/extension.h>
#include <vector>

using namespace torch::indexing;

std::vector<torch::Tensor> fast_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor indx_seq
){
    return at::sum_out(
        weight * input.index(Slice(), indx_seq), at::DimnameList(2)) + bias;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_forward", &fast_forward, "Fast forward");
}
