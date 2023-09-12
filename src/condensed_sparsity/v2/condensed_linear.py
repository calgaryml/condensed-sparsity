import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear  # noqa
from typing import Optional


class CondensedLinearStructured(nn.Module):
    # TODO: Experiment with __constants__ and __final__
    # SEE:
    # __TARGET_TYPES = [nn.Linear, NonDynamicallyQuantizableLinear]
    __TARGET_TYPES = [nn.Linear]
    # __constants__ = ["active_neuron_idx", "fine_grained_idx"]
    # in_features: int
    # out_features: int
    # active_neuron_idx: torch.Tensor
    # fine_grained_idx: torch.Tensor
    # weight: torch.Tensor

    def __init__(
        self, module: nn.Module, dtype: Optional[torch.typename] = None
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype
        # self._register_idx(module)
        self.active_neuron_idx = module.weight.sum(dim=1) != 0
        self.fine_grained_idx = (module.weight[self.active_neuron_idx] != 0).to(
            torch.bool
        )
        _, self.input_mask = self.fine_grained_idx.nonzero(as_tuple=True)
        self.input_mask = self.input_mask.reshape(
            shape=(module.weight[self.active_neuron_idx].shape[0], -1)
        )
        with torch.no_grad():
            # self.weight = nn.Parameter(
            #     module.weight[self.active_neuron_idx].contiguous()
            # )
            # self.condensed_weight = nn.Parameter(
            #     self.weight[self.fine_grained_idx]
            #     .reshape(shape=(self.weight.shape[0], -1))
            #     .contiguous()
            # )
            # self.sparse_weight = nn.Parameter(
            #     self.weight.to_sparse_csr()
            # )
            # if hasattr(module, "bias"):
            #     self.bias = nn.Parameter(
            #         module.bias[self.active_neuron_idx].contiguous()
            #     )
            # else:
            #     self.register_parameter("bias", None)
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
        # self.in_features = module.in_features
        # self.out_features=module.out_features
        self._clean_up_unused_params()

    def _clean_up_unused_params(self):
        del self.condensed_weight
        del self.sparse_weight
        del self.input_mask
        del self.active_neuron_idx
        del self.fine_grained_idx

    @torch.no_grad()
    def _register_idx(self, module):
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

    @classmethod
    def convert_condensed_linear(cls, module):
        # Based on convert_sync_batchnorm
        module_output = module
        if type(module) in cls.__TARGET_TYPES:
            # Introspection to determine subclass
            module_output = cls.__new__(module)
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_condensed_linear(child))
        del module
        return module_output


class CondensedLinearFineGrained(nn.Module):
    def __init__(
        self, module: nn.Module, dtype: Optional[torch.typename] = None
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(self.condensed_weight * input[:, self.input_mask], dim=2)
            + self.bias
        )


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


## TODO: Can use these with torch.compile, but not TorchScript. See: https://pytorch.org/docs/stable/jit_language_reference.html#:~:text=No%20support%20for%20inheritance%20or%20any%20other%20polymorphism%20strategy%2C%20except%20for%20inheriting%20from%20object%20to%20specify%20a%20new%2Dstyle%20class.  # noqa
# class CondensedLinearFineGrained(CondensedLinearStructured):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _clean_up_unused_params(self):
#         del self.weight
#         del self.active_neuron_idx
#         del self.fine_grained_idx
#         del self.sparse_weight

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return (
#             torch.sum(self.condensed_weight * input[:, self.input_mask], axis=2)
#             + self.bias
#         )


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
