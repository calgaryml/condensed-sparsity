import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def _gen_indx_seqs(fan_in: int, num_out: int, num_in: int, fan_out_const: bool) -> torch.LongTensor:
    """
    Generates indices for drawing recombination vectors from the input vector v.
        
        number recomb.vectors = fan_in (also called k)
        length of each recomb.vector = num_out (= n2)

    Args:
        fan_in: Number of incoming connections for each neuron.
        num_out: Number of neurons in the current layer.
        num_in: TODO.
        fan_out_const: If True, nearly constant fan-out will be ensured.
        
    Returns:
        A 2d array of indices of the same shape as weights.
    """
    indx_seqs = np.zeros((num_out, fan_in))

    # indices of v (the input of length d)
    v_inds = np.arange(num_in)

    # initialize an array of probabs for every index of v (initially uniform)
    probs = 1/num_in*np.ones(num_in)

    for row_nr in range(num_out):
        chosen_inds = np.random.choice(v_inds, size=fan_in, replace=False, 
                                       p=probs/sum(probs) )
        chosen_inds.sort()
        # update probabs only if want to control fan_out
        if fan_out_const: 
            probs[chosen_inds] /= (100*num_in)

        indx_seqs[row_nr, :] = chosen_inds
    
    return torch.LongTensor(indx_seqs.astype(int))


class LinearCondensed(nn.Module):
    r"""Applies a special condensed matmul
    transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        fan_in: the number of incoming connections for each neuron in the layer.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    fan_in: int
    weight: torch.Tensor
    indx_seqs: torch.Tensor

    def __init__(self, in_features: int, out_features: int,
                 fan_in: int, fan_out_const: bool, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearCondensed, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in = fan_in
        self.weight = Parameter(torch.empty((out_features, fan_in), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # ===== INDICES FOR RECOMBS =====
        self.indx_seqs = _gen_indx_seqs(fan_in=fan_in, num_out=out_features, num_in=in_features,
                                        fan_out_const=fan_out_const)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            dense_fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(dense_fan_in) if dense_fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.sum(self.weight * input[:, self.indx_seqs], axis=2) + self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, fan_in={}, bias={}'.format(
            self.in_features, self.out_features, self.fan_in, self.bias is not None
        )
