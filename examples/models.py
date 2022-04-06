import torch
import torch.nn as nn
import numpy as np
from utils import gen_indx_seqs, make_smask
import math
from torch.nn.parameter import Parameter



class LinearCondensed(nn.Module):
    r"""Applies a special condensed matmul
    transformation to the incoming data: :math:`y = xA^T + b`

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
    indx_seqs: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool, 
                 input_len: int, fan_out_const: bool, device=None, dtype=None) -> None:
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

        # ===== INDICES FOR RECOMBS =====
        self.indx_seqs= torch.LongTensor( 
                            gen_indx_seqs(num_in=in_features, num_out=out_features, input_len=input_len, 
                                fan_out_const=fan_out_const
                                ) 
                            )

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output= torch.sum(self.weight * input[:, self.indx_seqs], axis=2) + self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )





######################    Net    ######################
#######################################################

class Net(nn.Module):
    def __init__(self, num_layers, num_in, num_out, num_mid, make_linear=False, add_bias=False):
        super(Net, self).__init__()
        
        """
            X = num_layers

            This is a X-layer NN with all layers dense. All hidden layers have the same width. 

            - inLayer:    num_in  x num_mid
            - midLayers:  num_mid x num_mid
            - outLayer:   num_mid x num_out
        """

        self.printout= True
        self.init_distrib= 'normal'
        self.make_linear= make_linear

        num_mid_layers= num_layers-2

        # ===== LAYERS =====
        self.inLayer = nn.Linear(num_in, num_mid, bias=add_bias)
        self.outLayer= nn.Linear(num_mid, num_out, bias=add_bias)
        self.midLayers= nn.ModuleList([ nn.Linear(num_mid, num_mid, bias=add_bias) for i in range(num_mid_layers)])

        if self.printout:
            print(f'inLayer {self.inLayer.weight.shape}')
            for i in range(num_mid_layers):
                print(f'midLayers[{i}] {self.midLayers[i].weight.shape}')
            print(f'outLayer {self.outLayer.weight.shape}\n')


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
        
        for mid_layer in self.midLayers:
            out= mid_layer(out)
            if self.make_linear==False: out= self.activation_funct(out)

        out= self.outLayer(out)
        return out


    def reinit_parameters(self, layer, stddev):
        if self.init_distrib=='normal':
            if self.printout:
                print(f'Reinit layer with shape {layer.weight.shape} from normal distrib')
                print(f'stddev={stddev:.5f}') 
            nn.init.normal_(layer.weight, mean=0.0, std=stddev)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=stddev)
        else:
            print('init distrib not normal')





####################    CondNet    ####################
#######################################################

class CondNet(nn.Module):
    def __init__(self, num_layers, num_in, num_out, num_mid, fan_in, fan_out_const, make_linear=False, add_bias=False):
        super(CondNet, self).__init__()
        
        """
            X = num_layers

            This is a X-layer NN with first and last layers dense, and all hidden layers sparse in condensed representation.
            All hidden layers have the same width. 

            - inLayer:    num_in  x num_mid
            - midLayers:  fan_in  x num_mid -- CONDENSED TYPE
            - outLayer:   num_mid x num_out
        """

        self.printout= True
        self.init_distrib= 'normal'
        self.make_linear= make_linear

        num_cond_layers= num_layers-2

        # ===== LAYERS =====
        self.inLayer = nn.Linear(num_in, num_mid, bias=add_bias)
        self.outLayer= nn.Linear(num_mid, num_out, bias=add_bias)
        self.midLayers= nn.ModuleList([
                LinearCondensed(in_features=fan_in, out_features=num_mid, 
                                bias=add_bias, input_len=num_mid, 
                                fan_out_const=fan_out_const) 
                        for i in range(num_cond_layers)])

        if self.printout:
            print(f'inLayer {self.inLayer.weight.shape}')
            for i in range(num_cond_layers):
                print(f'midLayers[{i}] {self.midLayers[i].weight.shape}')
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
            out= cond_layer(out)
            if self.make_linear==False: out= self.activation_funct(out)

        out= self.outLayer(out)
        return out


    def reinit_parameters(self, layer, stddev):
        if self.init_distrib=='normal':
            if self.printout:
                print(f'Reinit layer with shape {layer.weight.shape} from normal distrib')
                print(f'stddev={stddev:.5f}') 
            nn.init.normal_(layer.weight, mean=0.0, std=stddev)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=stddev)
        else:
            print('init distrib not normal')





###################    SparseNet    ###################
#######################################################

class SparseNet(nn.Module):
    def __init__(self, num_layers, num_in, num_out, num_mid, make_linear, add_bias, fan_in, sparsity_type, connect_type, fan_out_const):
        super(SparseNet, self).__init__()
        
        """
            X = num_layers

            This is a X-layer NN with all hidden layers of same width and sparse. There are various sparsity types.

            - inLayer:    num_in  x num_mid
            - midLayers:  num_mid x num_mid
            - outLayer:   num_mid x num_out
        """

        self.printout= True
        self.init_distrib= 'normal'
        self.make_linear= make_linear

        num_mid_layers= num_layers-2

        # ===== LAYERS =====
        self.inLayer = nn.Linear(num_in, num_mid, bias=add_bias)
        self.outLayer= nn.Linear(num_mid, num_out, bias=add_bias)
        self.midLayers= nn.ModuleList([ nn.Linear(num_mid, num_mid, bias=add_bias) for i in range(num_mid_layers)])

        if self.printout:
            print(f'inLayer {self.inLayer.weight.shape}')
            for i in range(num_mid_layers):
                print(f'midLayers[{i}] {self.midLayers[i].weight.shape}')
            print(f'outLayer {self.outLayer.weight.shape}\n')


        # ===== INITIALIZATION ADJUSTMENT =====
        if self.init_distrib== 'normal':
            for layer in [self.inLayer, self.outLayer]:
                stddev= 1/np.sqrt(layer.weight.shape[-1])
                self.reinit_parameters(layer, stddev)                
            for layer in self.midLayers:
                stddev= 1/np.sqrt(fan_in)
                self.reinit_parameters(layer, stddev)

        # ===== ACTIVATION FUNCTION =====
        if self.make_linear==False:
            self.activation_funct = nn.ReLU()

        # ===== SPARSITY =====
        for layer in self.midLayers:
            M= make_smask(dims=layer.weight.shape, fan_in=fan_in, sparsity_type=sparsity_type, connect_type=connect_type, fan_out_const=fan_out_const)
            # apply mask & zero out weights
            print('Apply smask.')
            with torch.no_grad(): layer.weight[M] = 0

    def forward(self, x):
        out= self.inLayer(x)
        if self.make_linear==False: out= self.activation_funct(out)
        
        for mid_layer in self.midLayers:
            out= mid_layer(out)
            if self.make_linear==False: out= self.activation_funct(out)

        out= self.outLayer(out)
        return out


    def reinit_parameters(self, layer, stddev):
        if self.init_distrib=='normal':
            if self.printout:
                print(f'Reinit layer with shape {layer.weight.shape} from normal distrib with stddev={stddev:.5f}')
            nn.init.normal_(layer.weight, mean=0.0, std=stddev)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=stddev)
        else:
            print('init distrib not normal')










#from utils import get_layer_dims_for_simpleCNN2


# class simpleCNN2(nn.Module):
#     def __init__(self, input_img_size, num_classes, num_channels, 
#                  num_out_conv1, num_out_conv2, num_out_fc, fan_in, fan_out_const=False):
#         super(simpleCNN2, self).__init__()

#         """
#             This is a minimal CNN with 2 convolutional layers and one fully-connected layer. 
#             The fully-connected layer is a special "condensed" layer.
#             - convLayer1 gets the input; num_channels in, num_out_conv1 out

#             -- after flattening, the output is of length num_out_conv*pooled_img_size**2 (see utils for details)
#             - FCcondLayer is an implicitly sparse layer represented by a dense ("condensed") matrix of shape fan_in x num_out_fc
#             - outLayer maps from num_out_fc to num_classes
#         """
#         self.printout= True
#         self.init_distrib= 'normal'

#         # ===== LAYERS =====
#         num_in, num_out, pooled_img_size= get_layer_dims_for_simpleCNN2(
#             num_out_conv1=num_out_conv1, 
#             num_out_conv2=num_out_conv2, 
#             num_out_fc=num_out_fc, 
#             fan_in= fan_in, 
#             input_img_size=input_img_size, 
#             num_channels=num_channels, 
#             num_classes=num_classes
#             )

#         lkeys= {'convLayer1', 'convLayer2', 'FCcondLayer', 'outLayer'}

#         # ===== ACTIVATION FUNCTION =====
#         self.activation_funct= nn.ReLU()

#         #=== CONV layer 1
#         lkey='convLayer1'
#         self.convLayer1= nn.Conv2d(num_in[lkey], num_out[lkey], kernel_size=3, stride=1, padding=1)
#         self.bn1= nn.BatchNorm2d(num_out[lkey])
        
#         # #=== CONV layer 2
#         lkey='convLayer2'
#         self.convLayer2= nn.Conv2d(num_in[lkey], num_out[lkey], kernel_size=3, stride=1, padding=1)
#         self.bn2= nn.BatchNorm2d(num_out[lkey])

#         #=== pooling, used after each conv layer
#         self.pooling= nn.MaxPool2d(kernel_size=2)

#         #=== FC-COND layer
#         lkey='FCcondLayer'
#         self.FCcondLayer= nn.Linear(num_in[lkey], num_out[lkey])

#         #=== CLASSIFIER
#         lkey='outLayer'
#         self.outLayer= nn.Linear(num_in[lkey], num_out[lkey])


#         if self.printout:
#             print('-=- Layers and Sizes: -=-')
#             for lkey in lkeys:
#                 layer= getattr(self,lkey)
#                 print(f'{lkey}: {layer.weight.shape}')


#         # ===== INITIALIZATION ADJUSTMENT =====
#         if self.init_distrib== 'normal':
#             if self.printout: print('Adjusting init distrib to normal...')
#             for lkey in ['FCcondLayer','outLayer']:
#                 layer= getattr(self,lkey)
#                 stddev= 1/np.sqrt(layer.weight.shape[-1])
#                 self.reinit_parameters(layer, stddev)


#         # ===== INDICES FOR RECOMBS =====
#         self.indx_seqs= torch.LongTensor( 
#                             gen_indx_seqs(self.FCcondLayer.weight.data, num_out_conv2*pooled_img_size**2, fan_out_const) 
#                             )

#     def forward(self, x):
#         # conv layer 1
#         out= self.convLayer1(x)
#         out= self.bn1(out)
#         out= self.activation_funct(out)
#         out= self.pooling(out)
        
#         # conv layer 2
#         out= self.convLayer2(out)
#         out= self.bn2(out)
#         out= self.activation_funct(out)
#         out= self.pooling(out)
        
#         out= out.reshape(out.size(0), -1)

#         # fc layer
#         # <!> special op: condensed-mat-mult happens here <!>
#         out= spe_matmul(inputs=out, weights=self.FCcondLayer.weight.data, indx_seqs=self.indx_seqs)
#         out= out+self.FCcondLayer.bias.data
#         out= self.activation_funct(out)

#         # classifier
#         out= self.outLayer(out)

#         return out

#     def reinit_parameters(self, layer, stddev): 
#         if self.init_distrib=='normal':
#             if self.printout: print(f'distribution: {self.init_distrib}')
#             if self.printout: print(f'stddev={stddev}') 
#             nn.init.normal_(layer.weight, mean=0.0, std=stddev)
#             if layer.bias is not None:
#                 nn.init.normal_(layer.bias, mean=0.0, std=stddev)
#         elif self.init_distrib=='uniform':
#             nn.init.uniform_(layer.weight, -stddev, stddev)
#             if layer.bias is not None:
#                 nn.init.uniform_(layer.bias, -stddev, stddev)




