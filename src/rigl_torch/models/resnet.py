from torchvision.models.resnet import _resnet, Bottleneck
from torchvision.models import resnet18

from rigl_torch.models.model_factory import ModelFactory


@ModelFactory.register_model_loader(model="wide_resnet22", dataset="cifar10")
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


# ImageNet kernel sizes
def get_resnet18(num_classes: int):
    return resnet18(num_classes=num_classes)


if __name__ == "__main__":
    net = ModelFactory.load_model("wide_resnet22", "cifar10")
    print(net)
