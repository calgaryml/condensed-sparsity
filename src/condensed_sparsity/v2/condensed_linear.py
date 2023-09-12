import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear  # noqa


class CondensedLinearStructured(nn.Module):
    # __TARGET_TYPES = [nn.Linear, NonDynamicallyQuantizableLinear]
    __TARGET_TYPES = [nn.Linear]
    # __constants__ = ["active_neuron_idx", "fine_grained_idx"]
    # in_features: int
    # out_features: int
    active_neuron_idx: torch.Tensor
    fine_grained_idx: torch.Tensor
    weight: torch.Tensor

    def __init__(self, module: nn.Module):
        super().__init__()
        self._register_idx(module)
        with torch.no_grad():
            self.weight = nn.Parameter(
                module.weight[self.active_neuron_idx].contiguous()
            )
            self.condensed_weight = nn.Parameter(
                self.weight[self.fine_grained_idx]
                .reshape(shape=(self.weight.shape[0], -1))
                .contiguous()
            )
            self.sparse_weight = nn.Parameter(
                self.weight.to_sparse_csr().contiguous()
            )
            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    module.bias[self.active_neuron_idx].contiguous()
                )
            else:
                self.register_parameter("bias", None)
        # self.in_features = module.in_features
        # self.out_features=module.out_features

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

    def fine_grained_forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(self.condensed_weight * input[:, self.input_mask], axis=2)
            + self.bias
        )

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


class CondensedLinearFineGrained(CondensedLinearStructured):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(self.condensed_weight * input[:, self.input_mask], axis=2)
            + self.bias
        )


class CondensedLinearFineGrainedSparseOp(CondensedLinearStructured):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # todo
        return F.linear(input, self.sparse_weight, self.bias)
