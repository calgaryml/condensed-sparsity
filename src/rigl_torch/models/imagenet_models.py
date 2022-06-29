from torchvision.models.resnet import _resnet, Bottleneck
from torchvision.models import resnet18

from rigl_torch.models.model_factory import ModelFactory


@ModelFactory.register_model_loader(model="wide_resnet22", dataset="imagenet")
def get_wide_resnet_22():
    kwargs = dict(
        width_per_group=64 * 2,
        num_classes=10,
        pretrained=False,
        progress=True,
        layers=[2, 2, 4, 2],
    )
    wide_resnet_22 = _resnet("wide_resnet22_2", Bottleneck, **kwargs)
    return wide_resnet_22


@ModelFactory.register_model_loader(model="resnet18", dataset="imagenet")
def get_resnet18(num_classes: int):
    return resnet18(num_classes=num_classes)
