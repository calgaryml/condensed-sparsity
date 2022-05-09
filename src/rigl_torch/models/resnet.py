from torchvision.models.resnet import _resnet, Bottleneck
from torchvision.models import resnet18


def get_wide_resnet_22(num_classes: int, width_multiplier: int = 2):
    kwargs = dict(
        width_per_group=64 * width_multiplier,
        num_classes=num_classes,
        pretrained=False,
        progress=True,
        layers=[2, 2, 4, 2],
    )
    wide_resnet_22 = _resnet("wide_resnet22_2", Bottleneck, **kwargs)
    return wide_resnet_22


def get_resnet18(num_classes: int):
    return resnet18(num_classes=num_classes)
