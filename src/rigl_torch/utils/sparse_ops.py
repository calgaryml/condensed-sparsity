import torch
import copy
from sparseprop.modules import SparseConv2d, SparseLinear


class SparseModelFactory:
    def __init__(self):
        self.module_mapping = {
            torch.nn.Linear: SparseLinear,
            torch.nn.Conv2d: SparseConv2d,
        }

    def get_sparse_model(self, model, input_shape):
        for name, mod in model.named_modules():
            if type(mod) in self.module_mapping.keys():
                new_mod = self._get_new_mod(
                    current_mod=mod, input_shape=input_shape
                )
                self._swap_module(model, name, new_mod)
        return model

    def _swap_module(self, network, module_name, new_module):
        name_parts = module_name.split(".")
        parent = copy.deepcopy(network)
        for part in name_parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        last_part = name_parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = new_module
        else:
            setattr(parent, last_part, new_module)

    def _get_new_mod(
        self,
        current_mod,
        input_shape,
    ):
        if isinstance(current_mod, torch.nn.Linear):
            return self._get_new_linear_mod(current_mod, input_shape)
        elif isinstance(current_mod, torch.nn.Conv2d):
            return self._get_new_conv_mod(current_mod, input_shape)

    def _get_new_linear_mod(self, mod, input_shape):
        bias = None if mod.bias is None else torch.nn.Parameter(mod.bias.data)
        return SparseLinear(dense_weight=mod.weight.data, bias=bias)

    def _get_new_conv_mod(self, mod, input_shape):
        bias = None if mod.bias is None else torch.nn.Parameter(mod.bias.data)
        dense_weight = mod.weight.data
        stride = mod.stride[0]
        padding = mod.padding[0]
        return SparseConv2d(
            dense_weight=dense_weight,
            bias=bias,
            padding=padding,
            stride=stride,
            vectorizing_over_on=True,
        )
