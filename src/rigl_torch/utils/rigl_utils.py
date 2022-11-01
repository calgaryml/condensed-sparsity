import torch
import torch.nn as nn
import torchvision
from hydra import DictConfig
from typing import Tuple, Union


EXCLUDED_TYPES = (torch.nn.BatchNorm2d,)


def get_weighted_layers(model, i=0, layers=None, linear_layers_mask=None):
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    items = model._modules.items()
    if i == 0:
        items = [(None, model)]

    for layer_name, p in items:
        if isinstance(p, torch.nn.Linear):
            layers.append([p])
            linear_layers_mask.append(1)
        elif hasattr(p, "weight") and type(p) not in EXCLUDED_TYPES:
            layers.append([p])
            linear_layers_mask.append(0)
        elif isinstance(p, torchvision.models.resnet.Bottleneck) or isinstance(
            p, torchvision.models.resnet.BasicBlock
        ):
            _, linear_layers_mask, i = get_weighted_layers(
                p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask
            )
        else:
            _, linear_layers_mask, i = get_weighted_layers(
                p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask
            )

    return layers, linear_layers_mask, i


def get_W(model, return_linear_layers_mask=False):
    layers, linear_layers_mask, _ = get_weighted_layers(model)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], "weight") else 1
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)

    if return_linear_layers_mask:
        return W, linear_layers_mask
    return W


def calculate_fan_in_and_fan_out(
    module: Union[nn.Module, nn.parameter.Parameter]
) -> Tuple[int]:
    """Get tuple of fan_in and fan_out for a parameter / module

    Args:
        module (Union[nn.Module, nn.parameter.Parameter]): Module or parameter
            to obtain fan in / out for.

    Raises:
        ValueError: If dim of parameter < 2

    Returns:
        Tuple[int]: Fan-in, fan out tuple
    """
    if isinstance(module, nn.Module):
        tensor = module._parameters["weight"]
    else:
        tensor = module
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than "
            "2 dimensions"
        )
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:  # If module has a kernel
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def get_fan_in_tensor(mask: torch.Tensor) -> torch.Tensor:
    """Get tensor of fan-in per filter / neuron

    Args:
        mask (torch.Tensor): Boolean mask or weight matrix for layer

    Raises:
        ValueError: If mask dim < 2

    Returns:
        torch.Tensor: Tensor of shape [num_filters,] with each element == number
            of fan-in for that filter / neuron.
    """
    if mask.dim() < 2:
        raise ValueError(
            "Fan in can not be computed for tensor with fewer than 2 dimensions"
        )
    if mask.dtype == torch.bool:
        fan_in_tensor = mask.sum(axis=list(range(1, mask.dim())))
    else:
        fan_in_tensor = (mask != 0.0).sum(axis=list(range(1, mask.dim())))
    return fan_in_tensor


def validate_constant_fan_in(fan_in_tensor: torch.Tensor) -> bool:
    """Returns True if all filters / neurons in fan-in tensor are equal.

    Args:
        fan_in_tensor (torch.Tensor): Fan in tensor returneed by
            get_fan_in_tensor

    Returns:
        bool: True if fan-in are all equal.
    """
    return (fan_in_tensor == fan_in_tensor[0]).all()


def get_T_end(
    cfg: DictConfig, train_loader: torch.utils.data.DataLoader
) -> int:
    """Get step number to terminate pruning / regrowth based on cfg settings.

    Args:
        cfg (DictConfig): Config object loaded from ./configs
        train_loader (torch.utils.data.DataLoader): Train loader used to train
            model

    Returns:
        int: Step number at which to terminate pruning / regrowth.
    """
    if cfg.training.max_steps is None:
        if cfg.compute.distributed:
            # In distributed mode, len(train_loader) will be reduced by
            # 1/world_size compared to single device
            T_end = int(
                0.75
                * cfg.training.epochs
                * len(train_loader)  # Dataset length // batch_size
                * cfg.compute.world_size
            )
        else:
            T_end = int(0.75 * cfg.training.epochs * len(train_loader))
    else:
        T_end = int(0.75 * cfg.training.max_steps)
    if not cfg.rigl.use_t_end:
        T_end = int(1 / 0.75 * T_end)  # We use the full number of steps
    return T_end
