import torch
import logging
from typing import Optional
import math

from rigl_torch.utils.rigl_utils import get_fan_in_tensor


def sparse_kaiming_normal(
    tensor: torch.Tensor,
    sparsity_mask: Optional[torch.Tensor] = None,
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
        fan_in_tensor = get_fan_in_tensor(tensor)
    else:
        fan_in_tensor = get_fan_in_tensor(sparsity_mask)
    gain = calculate_gain(nonlinearity, a)
    for i in range(len(tensor)):
        fan_in = fan_in_tensor[i]
        if fan_in != 0:  # Neuron has some active connections
            std = gain / math.sqrt(fan_in)
        with torch.no_grad():
            if fan_in == 0:  # Neuron has been ablated
                tensor[i] = 0
            tensor[i] = tensor[i].normal_(0, std)
    if sparsity_mask is not None:
        tensor = tensor * sparsity_mask
    return tensor


def sparse_kaiming_uniform(
    tensor: torch.Tensor,
    sparsity_mask: Optional[torch.Tensor] = None,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    logger: Optional[logging.Logger] = None,
):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the magnitudes
            in the backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'``
            (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
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
        fan_in_tensor = get_fan_in_tensor(tensor)
    else:
        fan_in_tensor = get_fan_in_tensor(sparsity_mask)
    gain = calculate_gain(nonlinearity, a)
    for i in range(len(tensor)):
        fan_in = fan_in_tensor[i]
        if fan_in != 0:  # Neuron has some active connections
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
        with torch.no_grad():
            if fan_in == 0:  # Neuron has been ablated
                tensor[i] = 0
            tensor[i] = tensor[i].uniform_(-bound, bound)
    if sparsity_mask is not None:
        tensor = tensor * sparsity_mask
    return tensor


def sparse_torch_init(
    tensor: torch.Tensor,
    sparsity_mask: Optional[torch.Tensor] = None,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
    logger: Optional[logging.Logger] = None,
):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the magnitudes
            in the backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'``
            (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
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
        fan_in_tensor = get_fan_in_tensor(tensor)
    else:
        fan_in_tensor = get_fan_in_tensor(sparsity_mask)
    for i in range(len(tensor)):
        fan_in = fan_in_tensor[i]
        if fan_in != 0:  # Neuron has some active connections
            bound = math.sqrt(1 / fan_in)
        with torch.no_grad():
            if fan_in == 0:  # Neuron has been ablated
                tensor[i] = 0
            tensor[i] = tensor[i].uniform_(-bound, bound)
    if sparsity_mask is not None:
        tensor = tensor * sparsity_mask
    return tensor


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
