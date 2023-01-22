from torchvision.models.resnet import _resnet, Bottleneck
from torchvision.models import resnet18, resnet50

from rigl_torch.models.model_factory import ModelFactory


@ModelFactory.register_model_loader(model="wide_resnet22", dataset="imagenet")
def get_imagenet_wide_resnet_22(*args, **kwargs):
    wide_resnet_22_kwargs = dict(
        width_per_group=64 * 2,
        num_classes=1000,
        pretrained=False,
        progress=True,
        layers=[2, 2, 4, 2],
    )
    wide_resnet_22 = _resnet(
        "wide_resnet22_2", Bottleneck, **wide_resnet_22_kwargs
    )
    return wide_resnet_22


@ModelFactory.register_model_loader(model="resnet18", dataset="imagenet")
def get_imagenet_resnet18(*args, **kwargs):
    return resnet18(num_classes=1000)


@ModelFactory.register_model_loader(model="resnet50", dataset="imagenet")
def get_imagenet_resnet50(*args, **kwargs):
    return resnet50(num_classes=1000)
