import torch
import torch.nn as nn
import numpy as np
from utils import gen_indx_seqs
import math
from torch.nn.parameter import Parameter



class LinearCondensed(nn.Module):
    r"""Applies a special condensed-matmul transformation to the incoming data.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
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
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, indx_seqs: torch.Tensor, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearCondensed, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, indx_seqs: torch.Tensor) -> torch.Tensor:
        output= torch.sum(self.weight * input[:, indx_seqs], axis=2) + self.bias
        return output


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )








####################    CondNet    ####################
#######################################################

class CondNet(nn.Module):
    def __init__(self, num_layers, num_in, num_out, num_mid, fan_in, fan_out_const, make_linear=False, add_bias=False, individ_indx_seqs=False):
        super(CondNet, self).__init__()
        
        """
            X = num_layers

            This is a X-layer NN with first and last layers dense, and all hidden layers sparse and in condensed representation.
            All hidden layers have the same width. 

            - inLayer:    num_in  x num_mid
            - midLayers:  num_mid x num_mid -- CONDENSED TYPE
            - outLayer:   num_mid x num_out
        """

        self.printout= True
        self.init_distrib= 'normal'
        self.make_linear= make_linear
        self.individ_indx_seqs= individ_indx_seqs


        num_cond_layers= num_layers-2
        
        # ===== INDICES FOR RECOMBS =====
        if individ_indx_seqs:
            self.indx_seqs={}
            for i in range(num_cond_layers):
                self.indx_seqs[i]= torch.LongTensor( 
                            gen_indx_seqs(num_in=fan_in, num_out=num_mid, input_len= num_mid, 
                                fan_out_const=fan_out_const
                                ) 
                            )
        else:
            self.indx_seqs= torch.LongTensor( 
                            gen_indx_seqs(num_in=fan_in, num_out=num_mid, input_len= num_mid, 
                                fan_out_const=fan_out_const
                                ) 
                            )

        #output_dir= 'output'
        #torch.save(self.indx_seqs, f'{output_dir}/indx_seqs.pt')

        # ===== LAYERS =====
        self.inLayer = nn.Linear(num_in, num_mid, bias=add_bias)
        self.outLayer= nn.Linear(num_mid, num_out, bias=add_bias)

        
        if individ_indx_seqs:
            self.midLayers= nn.ModuleList([LinearCondensed(fan_in, num_mid, bias=add_bias, indx_seqs=self.indx_seqs[i]) for i in range(num_cond_layers)])
        else:
            self.midLayers= nn.ModuleList([LinearCondensed(fan_in, num_mid, bias=add_bias, indx_seqs=self.indx_seqs) for i in range(num_cond_layers)])


        if self.printout:
            print(f'inLayer {self.inLayer.weight.shape}')
            for i in range(num_cond_layers):
                print(f'midLayers[{i}] {self.midLayers[i].weight.shape}')
            if individ_indx_seqs:
                print('individ_indx_seqs= True')
            else:
                print('individ_indx_seqs= False')
            print(f'outLayer {self.outLayer.weight.shape}')


        # ===== INITIALIZATION ADJUSTMENT =====
        if self.init_distrib== 'normal':
            for layer in [self.inLayer, self.outLayer]:
                stddev= 1/np.sqrt(layer.weight.shape[-1])
                self.reinit_parameters(layer, stddev)                
            for layer in self.midLayers:
                stddev= 1/np.sqrt(layer.weight.shape[-1])
                self.reinit_parameters(layer, stddev)

        # ===== ACTIVATION FUNCTION =====
        if self.make_linear==False:
            self.activation_funct = nn.ReLU()


    def forward(self, x):

        out= self.inLayer(x)
        if self.make_linear==False: out= self.activation_funct(out)
        
        for i, cond_layer in enumerate(self.midLayers):
            if self.individ_indx_seqs:
                out= cond_layer(out, self.indx_seqs[i])
            else:
                out= cond_layer(out, self.indx_seqs)

            if self.make_linear==False: out= self.activation_funct(out)

        out= self.outLayer(out)
        return out


    def reinit_parameters(self, layer, stddev):
        if self.init_distrib=='normal':
            if self.printout:
                print(f'Reinit layer with shape {layer.weight.shape}, apply normal distrib with stddev={stddev:.5f}')
            nn.init.normal_(layer.weight, mean=0.0, std=stddev)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=stddev)
        else:
            print('init distrib not normal')

