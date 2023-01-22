"""Implementation from:
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from rigl_torch.models import ModelFactory
from rigl_torch.utils.rigl_utils import get_names_and_W


class BasicBlock(nn.Module):
    # Not used for this block, but included to maintain compatibility with
    # Bottleneck block
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = (
        4  # Expansion * planes == number of output channels in expansion block
    )
    # See more in mobile net paper: https://arxiv.org/pdf/1801.04381.pdf

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SkinnyBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        skip_conn_output_channels: Optional[int] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels=out_channels[0],
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(
            out_channels[0],
            out_channels[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels[1]:
            if skip_conn_output_channels is not None:
                if out_channels[1] != skip_conn_output_channels:
                    print(
                        f"WARNING: Matching channel dims in shortcut..."
                        f"{out_channels[1]} != {skip_conn_output_channels})"
                    )
                    skip_conn_output_channels = out_channels[1]
            else:
                skip_conn_output_channels = out_channels[1]

            # Required for addition
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    skip_conn_output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(skip_conn_output_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SkinnyResNet18(nn.Module):
    def __init__(
        self,
        full_width_network: nn.Module,
        diet: List[int],
        num_blocks: List[int],
        num_classes=10,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.names, full_w = get_names_and_W(full_width_network)
        self.diet = diet
        self.full_width = [len(w) for w in full_w]
        self.width = [fw - d for fw, d in list(zip(self.full_width, self.diet))]
        print(f"Building skinny resnet18 with widths: {self.width}")
        self._in_channel_idx = 0

        self.conv1 = nn.Conv2d(
            3, self.width[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.width[0])
        self.layer1 = self._make_layer(
            num_blocks=self.num_blocks[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            num_blocks=self.num_blocks[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            num_blocks=self.num_blocks[2],
            stride=2,
        )
        self.layer4 = self._make_layer(num_blocks=self.num_blocks[3], stride=2)
        self.linear = nn.Linear(
            self.width[-2] * SkinnyBlock.expansion, num_classes
        )
        _, W = get_names_and_W(self)
        print(
            "Actual dims after account for  skinny resnet18 with widths: "
            f"{[len(w) for w in W]}\n"
            f"Original target: {self.width}]\n"
            f"Equal? {[len(w) for w in W] == self.width}\n"
            f"Extra layers == {len([len(w) for w in W]) - len(self.width)}"
        )

    @property
    def _block_start_idx(self):
        return self._in_channel_idx + 1

    def _make_layer(self, num_blocks, stride):
        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers = []
        print(f"strides {strides}")
        for stride in strides:
            skip_conn_output_channels = None
            if stride != 1:  # Then we have a skip connection!
                print("found skip")
                skip_conn_output_channels = self.width[
                    self._block_start_idx + num_blocks
                ]
            layers.append(
                SkinnyBlock(
                    self.width[self._in_channel_idx],
                    self.width[
                        self._block_start_idx : self._block_start_idx
                        + num_blocks
                    ],
                    stride=stride,
                    skip_conn_output_channels=skip_conn_output_channels,
                )
            )
            if skip_conn_output_channels is None:
                # We only need to move index num blocks away
                self._in_channel_idx += num_blocks
            else:
                # Else we need to move past this layers skip conn too
                self._in_channel_idx += num_blocks + 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@ModelFactory.register_model_loader(model="resnet18", dataset="cifar10")
def ResNet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2])


@ModelFactory.register_model_loader(model="skinny_resnet18", dataset="cifar10")
def get_skinny_resnet18(diet: List[int], *args, **kwargs):
    full_width_network = ModelFactory.load_model("resnet18", "cifar10")
    return SkinnyResNet18(
        full_width_network=full_width_network,
        diet=diet,
        num_blocks=[2, 2, 2, 2],
        num_classes=10,
    )


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == "__main__":
    get_skinny_resnet18(
        diet=[
            17,
            16,
            17,
            14,
            16,
            33,
            31,
            31,
            32,
            30,
            65,
            63,
            63,
            65,
            63,
            122,
            124,
            124,
            114,
            125,
            0,
        ]
    )
