from typing import Dict, Any, Union
import pathlib
from torchvision import transforms, datasets

from rigl_torch.datasets import _data_stem
from ._cc_imagenet_folder import CCImageNetFolder


class ImageNetDataStem(_data_stem.ABCDataStem):
    _IMAGE_HEIGHT = 224
    _IMAGE_WIDTH = 224
    _MEAN_RGB = [0.485, 0.456, 0.406]
    _STDDEV_RGB = [0.229, 0.224, 0.225]

    def __init__(
        self,
        cfg: Dict[str, Any],
        data_path_override: Union[pathlib.Path, str],
    ):
        super().__init__(cfg, data_path_override=data_path_override)

    def _get_datasets(self):
        train_transform = self._get_transform()
        test_transform = self._get_test_transform()
        if not self.cfg.dataset.use_cc_data_loaders:
            train_dataset = datasets.ImageNet(
                self.data_path, split="train", transform=train_transform
            )
            test_dataset = datasets.ImageNet(
                self.data_path, split="val", transform=test_transform
            )
        else:
            train_dataset = CCImageNetFolder(
                self.data_path,
                split="train",
                transform=train_transform,
                meta_file_path=self.cfg.paths.data_folder,
            )
            test_dataset = CCImageNetFolder(
                self.data_path,
                split="validation",
                transform=test_transform,
                meta_file_path=self.cfg.paths.data_folder,
            )
        return train_dataset, test_dataset

    def _get_transform(self):
        transform = transforms.Compose(
            [
                transforms.RandomChoice(
                    [transforms.Resize(256), transforms.Resize(480)]
                ),
                transforms.RandomCrop(
                    size=[self._IMAGE_WIDTH, self._IMAGE_HEIGHT]
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),  # Applies min/max scaling
                transforms.Normalize(mean=self._MEAN_RGB, std=self._STDDEV_RGB),
            ]
        )
        return transform

    def _get_test_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(
                    size=[self._IMAGE_WIDTH, self._IMAGE_HEIGHT]
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self._MEAN_RGB, std=self._STDDEV_RGB),
            ]
        )
        return transform
