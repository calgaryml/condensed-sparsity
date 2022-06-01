from dataclasses import dataclass
from condensed_sparsity.models import CondNet
from rigl_torch.models import ModelFactory


@dataclass
class CondNetConfig:
    num_layers: int = 4
    num_in: int = 784
    num_out: int = 10
    num_mid: int = int(784 / 10)
    fan_in: int = int(784 / 10)
    fan_out_const: bool = True
    make_linear: bool = False
    add_bias: bool = True


@ModelFactory.register_model_loader(model="cond_net", dataset="mnist")
def get_cond_net():
    args = CondNetConfig()
    return CondNet(**args.__dict__)


if __name__ == "__main__":
    args = CondNetConfig()
    print(args.__dict__)
    net = CondNet(**args.__dict__)
    print([x for x in net.modules()])
    # make_linear = args.make_linear
    # add_bias = not args.no_bias
    # num_layers = args.num_layers
    # num_mid = args.num_mid
    # fan_in = args.fan_in
    # fan_out_const = args.fan_out_const
    # net = CondNet()
