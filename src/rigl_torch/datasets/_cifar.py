from rigl_torch.datasets import _data_stem
from typing import Dict, Any
import torch
from torchvision import transforms, datasets
from ._transforms import PerImageStandarization  # noqa: F401


class CIFAR10DataStem(_data_stem.ABCDataStem):
    _IMAGE_HEIGHT = 32
    _IMAGE_WIDTH = 32

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

    def get_train_test_loaders(self):
        transform_train = self._get_transform(
            train=True, normalize=self.cfg.dataset.normalize
        )
        transform_test = self._get_transform(
            train=False, normalize=self.cfg.dataset.normalize
        )
        train_dataset = datasets.CIFAR10(
            self.data_path, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            self.data_path, train=False, transform=transform_test
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, **self.train_kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **self.test_kwargs
        )
        return train_loader, test_loader

    def _get_transform(self, train: bool, normalize: bool):
        transforms_list = []
        transforms_list.append(transforms.ToTensor())
        if normalize:
            transforms_list.append(PerImageStandarization(inplace=False))
            # Torch dataset already min/max scaled
        if train:
            transforms_list.extend(
                [
                    transforms.Pad(padding=4, padding_mode="reflect"),
                    # Equivalent to: https://github.com/google-research/rigl/blob/master/rigl/cifar_resnet/data_helper.py#L29  # noqa
                    transforms.RandomCrop(
                        size=[self._IMAGE_WIDTH, self._IMAGE_HEIGHT]
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        transform = transforms.Compose(transforms_list)
        return transform
