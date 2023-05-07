import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from omegaconf import DictConfig
from math import prod
from typing import Tuple, Union, List, Optional


_EXCLUDED_TYPES = (
    torch.nn.BatchNorm2d,
    torch.nn.LayerNorm,
)


def get_names_and_W(
    model: torch.nn.Module,
    names: list = None,
    weights: List[torch.Tensor] = None,
    skip_linear: bool = False,
    skip_mha: bool = False,
) -> Tuple[List[str], List[torch.nn.parameter.Parameter]]:
    """Much simpler implementation"""

    target_types = [
        torch.nn.Conv2d,
    ]
    if not skip_linear:
        target_types.append(torch.nn.Linear)
    if not skip_mha:
        target_types.append(NonDynamicallyQuantizableLinear)
    target_layers = []
    names = []
    for n, m in model.named_modules():
        if type(m) in target_types:
            target_layers.append(m)
            names.append(n)
    weights = [layer.weight for layer in target_layers]
    return names, weights


def get_weighted_layers(
    model, i=0, layers=None, linear_layers_mask=None, mha_layer_mask=None
):
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []
    if mha_layer_mask is None:
        mha_layer_mask = []

    items = model._modules.items()
    if i == 0:
        items = [(None, model)]

    for layer_name, p in items:
        if type(p) is NonDynamicallyQuantizableLinear:
            layers.append([p])
            mha_layer_mask.append(1)
            linear_layers_mask.append(0)
        elif type(p) is torch.nn.Linear:
            layers.append([p])
            linear_layers_mask.append(1)
            mha_layer_mask.append(0)
        elif hasattr(p, "weight") and type(p) not in _EXCLUDED_TYPES:
            layers.append([p])
            linear_layers_mask.append(0)
            mha_layer_mask.append(0)
        elif isinstance(p, torchvision.models.resnet.Bottleneck) or isinstance(
            p, torchvision.models.resnet.BasicBlock
        ):
            _, linear_layers_mask, mha_layer_mask, i = get_weighted_layers(
                p, i + 1, layers, linear_layers_mask, mha_layer_mask
            )
        else:
            _, linear_layers_mask, mha_layer_mask, i = get_weighted_layers(
                p, i + 1, layers, linear_layers_mask, mha_layer_mask
            )

    return layers, linear_layers_mask, mha_layer_mask, i


def get_W(model):
    layers, linear_layers_mask, mha_layers_mask, _ = get_weighted_layers(model)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], "weight") else 1
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)
    assert len(W) == len(mha_layers_mask)

    return W, linear_layers_mask, mha_layers_mask


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
    receptive_field_size = _get_receptive_field_size(tensor)
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
    if cfg.training.max_steps is None or cfg.training.max_steps == 0:
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
    if cfg.training.simulated_batch_size is not None:
        # We need to correct T_end to account for sim bs / actual bs
        T_end = int(
            T_end
            / (cfg.training.simulated_batch_size / cfg.training.batch_size)
        )
    return T_end


def get_static_filters_to_ablate(
    weight_tensor: torch.Tensor,
    sparsity: float,
    filter_ablation_threshold: float,
) -> int:
    """Return number of filters to ablate for a given weight tensor and sparsity

    Args:
        weight_tensor (torch.Tensor): Weight tensor for given convolutional
            layer
        sparsity (float): Sparisty target of layer. eg., 0.9 means 90% of
            weights set to zero
        filter_ablation_threshold (float): Threshold for maximum

    Returns:
        int: _description_
    """
    with torch.no_grad():
        dense_fan_in, _ = calculate_fan_in_and_fan_out(weight_tensor)
        receptive_field_size = _get_receptive_field_size(weight_tensor)
        out_channels = weight_tensor.shape[0]
        sparse_fan_in = int(dense_fan_in * (1 - sparsity))
        unadjusted_filter_sparsity = sparse_fan_in / (
            out_channels * receptive_field_size
        )
        if unadjusted_filter_sparsity < filter_ablation_threshold:
            filters_to_ablate = out_channels - int(
                sparse_fan_in
                / (filter_ablation_threshold * receptive_field_size)
            )
            if filters_to_ablate >= out_channels:
                filters_to_ablate = out_channels - 1
            return filters_to_ablate
        else:
            return 0  # No filters to remove


def get_fan_in_after_ablation(
    weight_tensor: torch.Tensor,
    num_neurons_to_ablate: int,
    sparsity: float,
):
    with torch.no_grad():
        active_neurons = weight_tensor.shape[0] - num_neurons_to_ablate
        remaining_non_zero_elements = int(
            weight_tensor.numel() * (1 - sparsity)
        )
        fan_in_after_ablation = remaining_non_zero_elements // active_neurons
        if fan_in_after_ablation == 0:
            raise ValueError(
                "Fan in after ablation is 0! "
                "Reduce sparsity or increase layer width"
            )
        return fan_in_after_ablation


def _get_receptive_field_size(tensor: torch.Tensor) -> int:
    receptive_field_size = 1
    if tensor.dim() > 2:  # If module has a kernel
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    return receptive_field_size


def get_conv_idx_from_flat_idx(
    flat_idx: int, conv_shape: Tuple[int]
) -> Tuple[int]:
    """Convert flat_idx to tuple idx based to match conv_shape.

    Args:
        flat_idx (int): Index of weight after calling flatten() on tensor
        conv_shape (Tuple): Shape of 2D convoltional layer (NCHW)

    Returns:
        Tuple[int]: Tuple index of the same connection in the 4D tensor.
    """
    fan_in = prod(conv_shape[1:])
    filter_idx = flat_idx // fan_in
    in_channel_idx = (flat_idx - (filter_idx * fan_in)) // prod(conv_shape[2:])
    kernel_row_idx = (
        flat_idx - filter_idx * fan_in - in_channel_idx * prod(conv_shape[2:])
    ) // prod(conv_shape[3:])
    kernel_col_idx = (
        flat_idx
        - filter_idx * fan_in
        - in_channel_idx * prod(conv_shape[2:])
        - kernel_row_idx * prod(conv_shape[3:])
    )
    return (filter_idx, in_channel_idx, kernel_row_idx, kernel_col_idx)


@torch.no_grad()
def active_neuron_count_in_layer(
    mask: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> int:
    if mask is None:
        return len(weight)
    else:
        active_count = sum([n.any() for n in mask])
    return active_count


if __name__ == "__main__":
    t = torch.zeros(size=(16, 3, 3, 3), dtype=torch.bool)
    w = torch.ones(size=t.size(), dtype=torch.bool)
    active_n = 16
    t[:active_n] = True
    assert active_n == active_neuron_count_in_layer(None, w)
    # from rigl_torch.models import ModelFactory

    # vit = ModelFactory.load_model(model="vit", dataset="imagenet")
    # n, w = get_names_and_W(vit)
    # W = get_W(vit)
