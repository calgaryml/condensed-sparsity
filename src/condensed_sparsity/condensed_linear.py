import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear  # noqa
from typing import Any, Optional, Callable, List  # noqa
from functools import partial  # noqa
from torch_sparse.ops import ffi_mul


# TODO Create factory methods for each type of condensed layer


def _get_active_neuron_idx(weight: torch.Tensor) -> torch.Tensor:
    # We find all-zero rows in first dimension of weight tensor
    return weight.sum(dim=list(range(1, weight.dim()))) != 0


def _get_fine_grained_idx(
    weight: torch.Tensor, active_neuron_idx
) -> torch.Tensor:
    return (weight[active_neuron_idx] != 0).to(torch.bool)


def _default_weight_getter(
    module: nn.Module, attr_name: str = "weight"
) -> torch.Tensor:
    return getattr(module, attr_name)


def structured_condensed_conv2d_factory(
    module: nn.Module,
    weight_getter: Optional[Callable] = _default_weight_getter,
    dtype: Optional[torch.typename] = None,
) -> nn.Conv2d:
    if dtype is None:
        dtype = module.weight.dtype
    with torch.no_grad():
        original_weight = weight_getter(module)
        active_neuron_idx = _get_active_neuron_idx(original_weight)
        module.weight = nn.Parameter(
            torch.clone(original_weight[active_neuron_idx].detach().type(dtype))
        )
        if hasattr(module, "bias"):
            module.bias = nn.Parameter(
                torch.clone(module.bias[active_neuron_idx].detach().type(dtype))
            )
        module.out_channels = module.weight.shape[0]
    return module


class CSRLinear(nn.Module):
    def __init__(
        self, module: nn.Module, dtype: torch.typename = torch.float32
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype
        with torch.no_grad():
            self.sparse_weight = nn.Parameter(
                torch.clone(module.weight.detach().type(dtype).to_sparse_csr()),
                requires_grad=False,
            )
            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    torch.clone(module.bias.detach().type(dtype)),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.sparse_weight, self.bias)


class CondensedLinearStructured(nn.Module):
    # TODO: Experiment with __constants__ and __final__
    # TODO: Going to need some functionality to capture weight getter,
    # maybe a callable/str union?
    # TODO: Type annotations may help speed up TorchScript

    def __init__(
        self, module: nn.Module, dtype: torch.typename = torch.float32
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype
        self.active_neuron_idx = module.weight.sum(dim=1) != 0
        self.fine_grained_idx = (module.weight[self.active_neuron_idx] != 0).to(
            torch.bool
        )
        _, self.input_mask = self.fine_grained_idx.nonzero(as_tuple=True)
        self.input_mask = self.input_mask.reshape(
            shape=(module.weight[self.active_neuron_idx].shape[0], -1)
        )
        with torch.no_grad():
            self.weight = nn.Parameter(
                torch.clone(
                    module.weight[self.active_neuron_idx].detach().type(dtype)
                )
            )
            self.condensed_weight = nn.Parameter(
                torch.clone(
                    self.weight[self.fine_grained_idx]
                    .reshape(shape=(self.weight.shape[0], -1))
                    .detach()
                    .type(dtype)
                ),
                requires_grad=False,
            )
            self.sparse_weight = nn.Parameter(
                torch.clone(self.weight.detach().type(dtype).to_sparse_csr()),
                requires_grad=False,
            )
            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    torch.clone(
                        module.bias[self.active_neuron_idx].detach().type(dtype)
                    ),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)
        self._clean_up_unused_params()

    def _clean_up_unused_params(self):
        del self.condensed_weight
        del self.sparse_weight
        del self.input_mask
        del self.active_neuron_idx
        del self.fine_grained_idx

    @torch.no_grad()
    def _register_idx(self, module: nn.Module):
        self.active_neuron_idx = module.weight.sum(dim=1) != 0
        self.fine_grained_idx = (module.weight[self.active_neuron_idx] != 0).to(
            torch.bool
        )
        _, self.input_mask = self.fine_grained_idx.nonzero(as_tuple=True)
        self.input_mask = self.input_mask.reshape(
            shape=(module.weight[self.active_neuron_idx].shape[0], -1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        out_features, in_features = self.weight.shape
        return "in_features={}, out_features={}, bias={}".format(
            in_features, out_features, self.bias is not None
        )

    # @classmethod
    # def convert_condensed_linear(cls, module):
    #     # Based on convert_sync_batchnorm
    #     module_output = module
    #     if type(module) in cls.__TARGET_TYPES:
    #         # Introspection to determine subclass
    #         module_output = cls.__new__(module)
    #         # TODO: Move cls method to each condensed class
    #         if hasattr(module, "qconfig"):
    #             module_output.qconfig = module.qconfig
    #     for name, child in module.named_children():
    #         module_output.add_module(name, cls.convert_condensed_linear(child))  # noqa
    #     del module
    #     return module_output


class CondensedLinearFineGrained(nn.Module):
    def __init__(
        self, module: nn.Module, dtype: torch.typename = torch.float32
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype
        with torch.no_grad():
            active_neuron_idx = module.weight.sum(dim=1) != 0
            fine_grained_idx = (module.weight[active_neuron_idx] != 0).to(
                torch.bool
            )
            _, self.input_mask = fine_grained_idx.nonzero(as_tuple=True)
            self.input_mask = self.input_mask.reshape(
                shape=(module.weight[active_neuron_idx].shape[0], -1)
            )
            self.input_mask = self.input_mask.to(torch.int32)
            weight = module.weight[active_neuron_idx].detach().type(dtype)
            self.condensed_weight = nn.Parameter(
                torch.clone(
                    weight[fine_grained_idx]
                    .reshape(shape=(weight.shape[0], -1))
                    .detach()
                    .type(dtype)
                ),
                requires_grad=False,
            )
            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    torch.clone(
                        module.bias[active_neuron_idx].detach().type(dtype)
                    ),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(
                self.condensed_weight * input[..., self.input_mask],
                dim=input.dim(),
            )
            + self.bias
        )


class FixedFanInCuda(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        dtype: torch.typename = torch.float32,
        transpose: bool = True,
        vectorize: bool = False,
        index_dtype: torch.typename = torch.int32,
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype

        self.transpose = transpose
        with torch.no_grad():
            active_neuron_idx = module.weight.sum(dim=1) != 0
            fine_grained_idx = (module.weight[active_neuron_idx] != 0).to(
                torch.bool
            )
            _, self.input_mask = fine_grained_idx.nonzero(as_tuple=True)
            self.input_mask = self.input_mask.reshape(
                shape=(module.weight[active_neuron_idx].shape[0], -1)
            ).to(index_dtype)
            weight = module.weight[active_neuron_idx].detach().type(dtype)
            weight = torch.clone(
                weight[fine_grained_idx]
                .reshape(shape=(weight.shape[0], -1))
                .detach()
                .type(dtype)
            )
            # padding to multiple of 4
            if vectorize:
                pad = (
                    self.input_mask.shape[1] + 3
                ) // 4 * 4 - self.input_mask.shape[1]
                self.input_mask = F.pad(self.input_mask, [0, pad])
                weight = F.pad(weight, [0, pad])

            self.condensed_weight = nn.Parameter(
                weight,
                requires_grad=False,
            )

            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    torch.clone(
                        module.bias[active_neuron_idx].detach().type(dtype)
                    ),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ffi_mul(
            input,
            self.condensed_weight,
            self.input_mask,
            self.bias,
            transpose=self.transpose,
        )


class CondensedLinearFineGrainedSparseOp(nn.Module):
    def __init__(
        self, module: nn.Module, dtype: torch.typename = torch.float32
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype
        active_neuron_idx = module.weight.sum(dim=1) != 0
        fine_grained_idx = (module.weight[active_neuron_idx] != 0).to(
            torch.bool
        )
        _, input_mask = fine_grained_idx.nonzero(as_tuple=True)
        input_mask = input_mask.reshape(
            shape=(module.weight[active_neuron_idx].shape[0], -1)
        )
        with torch.no_grad():
            weight = nn.Parameter(
                torch.clone(
                    module.weight[active_neuron_idx].detach().type(dtype)
                )
            )
            self.sparse_weight = nn.Parameter(
                torch.clone(weight.detach().type(dtype).to_sparse_csr()),
                requires_grad=False,
            )
            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    torch.clone(
                        module.bias[active_neuron_idx].detach().type(dtype)
                    ),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.sparse_weight, self.bias)


class VmapCondensed(nn.Module):
    def __init__(
        self, module: nn.Module, dtype: torch.typename = torch.float32
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype
        with torch.no_grad():
            active_neuron_idx = module.weight.sum(dim=1) != 0
            fine_grained_idx = (module.weight[active_neuron_idx] != 0).to(
                torch.bool
            )
            _, self.input_mask = fine_grained_idx.nonzero(as_tuple=True)
            self.input_mask = self.input_mask.reshape(
                shape=(module.weight[active_neuron_idx].shape[0], -1)
            )
            weight = module.weight[active_neuron_idx].detach().type(dtype)
            self.condensed_weight = nn.Parameter(
                torch.clone(
                    weight[fine_grained_idx]
                    .reshape(shape=(weight.shape[0], -1))
                    .detach()
                    .type(dtype)
                ),
                requires_grad=False,
            )
            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    torch.clone(
                        module.bias[active_neuron_idx].detach().type(dtype)
                    ),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor):
        return forward_fast(
            input, self.condensed_weight, self.bias, self.input_mask
        )


class forward_neuron_single:
    def __init__(self, input: torch.Tensor) -> torch.Tensor:
        self.input = input

    def __call__(self, weights, indices: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.input[indices] * weights)


class forward_neuron_v:
    def __init__(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        indx_seqs: torch.LongTensor,
    ) -> torch.Tensor:
        self.weights = weights
        self.bias = bias
        self.indx_seqs = indx_seqs

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return (
            torch.vmap(forward_neuron_single(input), in_dims=0, out_dims=0)(
                self.weights, self.indx_seqs
            )
            + self.bias
        )


class forward_neuron:
    def __init__(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        indx_seqs: torch.LongTensor,
    ):
        self.weights = weights
        self.bias = bias
        self.indx_seqs = indx_seqs

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return torch.vmap(
            forward_neuron_v(self.weights, self.bias, self.indx_seqs)
        )(input)


def forward_sparsity_single(
    input: torch.Tensor, weights: torch.Tensor, indices: torch.LongTensor
) -> torch.Tensor:
    return input[indices] * weights


def forward_sparsity_v(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    indx_seqs: torch.LongTensor,
) -> torch.Tensor:
    output_neurons = torch.vmap(
        lambda w, idx: forward_sparsity_single(
            input=input, weights=w, indx_seqs=idx
        ),
        in_dims=1,
        out_dims=1,
    )(weights, indx_seqs)
    return torch.sum(output_neurons, axis=1) + bias


def forward_sparsity(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    indx_seqs: torch.LongTensor,
) -> torch.Tensor:
    return torch.vmap(
        lambda i: forward_sparsity_v(
            input=i, weights=weights, bias=bias, indx_seqs=indx_seqs
        )
    )(input)


def forward_fast(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    indx_seqs: torch.LongTensor,
) -> torch.Tensor:
    # TODO: Remove control flow, conditionals will be set as constants after
    # jit tracing
    return forward_neuron(weights, bias, indx_seqs)(input)
    # if number of neurons is greater than sparsity, vmap over neurons
    if weights.shape[0] > weights.shape[1]:
        return forward_neuron(input, weights, bias, indx_seqs)
    # otherwise vmap over sparsity
    else:
        return forward_sparsity(input, weights, bias, indx_seqs)


## TODO: Can use these with torch.compile, but not TorchScript. See: https://pytorch.org/docs/stable/jit_language_reference.html#:~:text=No%20support%20for%20inheritance%20or%20any%20other%20polymorphism%20strategy%2C%20except%20for%20inheriting%20from%20object%20to%20specify%20a%20new%2Dstyle%20class.  # noqa
# class CondensedLinearFineGrained(CondensedLinearStructured):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _clean_up_unused_params(self):
#         del self.weight
#         del self.active_neuron_idx
#         del self.fine_grained_idx
#         del self.sparse_weight

# def forward(self, input: torch.Tensor) -> torch.Tensor:
#     return (
#         torch.sum(
#             self.condensed_weight * input[:, self.input_mask], axis=2)
#         + self.bias
#     )


# class CondensedLinearFineGrainedSparseOp(CondensedLinearStructured):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _clean_up_unused_params(self):
#         del self.weight
#         del self.input_mask
#         del self.active_neuron_idx
#         del self.fine_grained_idx
#         del self.condensed_weight

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return F.linear(input, self.sparse_weight, self.bias)
