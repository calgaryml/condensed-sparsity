from rigl_torch.datasets import _data_stem
from typing import Dict, Any

# import torch
# from torchvision import transforms, datasets
# from ._utils import PerImageStandarization


class ImageNetDataStem(_data_stem.ABCDataStem):
    # TODO
    _IMAGE_HEIGHT = 224
    _IMAGE_WIDTH = 224

    def __init__(self, cfg: Dict[str, Any]):
        raise NotImplementedError("ImageNet stem not implemented!")
        super().__init__(cfg)

    # def get_train_test_loaders(self):
    #     transform = self._get_transform()
    #     train_dataset = datasets.CIFAR10(
    #         self.data_path, train=True, download=True, transform=transform
    #     )
    #     test_dataset = datasets.CIFAR10(
    #         self.data_path, train=False, transform=transform
    #     )
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset, **self.train_kwargs
    #     )
    #     test_loader = torch.utils.data.DataLoader(
    #         test_dataset, **self.test_kwargs
    #     )
    #     return train_loader, test_loader

    # def _get_transform(self):
    #     transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             PerImageStandarization(inplace=False),
    #             transforms.Pad(padding=4, padding_mode="reflect"),
    #             # Equivalent to: https://github.com/google-research/rigl/blob/master/rigl/cifar_resnet/data_helper.py#L29  # noqa
    #             transforms.CenterCrop(
    #                 size=[self._IMAGE_WIDTH, self._IMAGE_HEIGHT]
    #             ),
    #             transforms.RandomHorizontalFlip(p=0.5),
    #         ]
    #     )
    #     return transform
