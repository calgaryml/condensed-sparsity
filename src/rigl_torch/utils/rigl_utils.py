import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from math import prod
from typing import Tuple, Union, Optional, List
import logging
import math

_EXCLUDED_TYPES = (torch.nn.BatchNorm2d,)


def get_names_and_W(
    model: torch.nn.Module,
) -> List[torch.nn.parameter.Parameter]:
    """Much simpler implementation"""
    target_types = [torch.nn.Conv2d, torch.nn.Linear]
    target_layers = []
    names = []
    for n, m in model.named_modules():
        if type(m) in target_types:
            target_layers.append(m)
            names.append(n)
    weights = [layer.weight for layer in target_layers]
    return names, weights


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
        elif hasattr(p, "weight") and type(p) not in _EXCLUDED_TYPES:
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
    return T_end


def sparse_kaiming_normal(
    tensor: torch.Tensor,
    sparsity_mask: Optional[torch.Tensor] = None,
    neurons_ablated: Optional[int] = None,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    logger: Optional[logging.Logger] = None,
):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where
    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}
    Also known as He initialization.

    This implementation is modified from the original pytorch implementation to
    use the fan_in from a given sparsity mask. In effect, this will decrease the
    std of the initalization values to account for the reduced fan_in from the
    sparse mask.

    tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
            ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'``
                (default).
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """  # noqa
    if mode.lower() != "fan_in":
        raise NotImplementedError(
            "Only mode==`fan_in` has currently been implemented at this time."
        )
    if sparsity_mask.shape != tensor.shape:
        raise ValueError("Sparsity mask and tensor shape do not match!")
    if logger is None:
        logger = logging.Logger(name=__file__, level=logging.INFO)
    if 0 in tensor.shape:
        logger.warning("Initializing zero-element tensors is a no-op")
        return tensor
    if sparsity_mask is None:
        fan_in, _ = calculate_fan_in_and_fan_out(tensor)
    else:
        fan_in = get_fan_in_tensor(
            sparsity_mask, neurons_ablated=neurons_ablated
        )[0]
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================
    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalisation
        effect for more stable gradient flow in rectangular layers.
    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function
    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html

    NOTE: This function copied from torch.nn.init module. Copied here to avoid
        any breaking changes from revisions to pytorch API.
    """  # noqa
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(
                "negative_slope {} not a valid number".format(param)
            )
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


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
