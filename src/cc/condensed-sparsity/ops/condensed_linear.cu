#include <torch/extension.h>
#include <vector>

// Cuda forward defn

std::vector<torch::Tensor> condensed_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor indx_seq
);


std::vector<torch::Tensor> condensed_linear_backward_cuda(
    torch::Tensor grad,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor indx_seq
);


// C++ API

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> condensed_linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor indx_seq
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(indx_seq);
    return condensed_linear_forward_cuda(input, weights, bias, indx_seq);
}

std::vector<torch::Tensor> condensed_linear_backward(
    torch::Tensor grad,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor indx_seq
) {
    CHECK_INPUT(grad);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    CHECK_INPUT(indx_seq);
    return condensed_linear_backward_cuda(grad, weights, bias, indx_seq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &condensed_linear_forward, "Condensed Linear Forward (CUDA)")
    m.def("backward", &condensed_linear_backward, "Condensed Linear Backward (CUDA)")
}
