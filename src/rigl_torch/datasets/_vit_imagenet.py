from typing import Dict, Any, Union
import pathlib
import torch
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataloader import default_collate

from rigl_torch.datasets import _data_stem
from ._cc_imagenet_folder import CCImageNetFolder
from ._transforms import RandomCutmix, RandomMixup


class VitImageNetDataStem(_data_stem.ABCDataStem):
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
        self._append_mix_up_to_train_kwargs()
        return train_dataset, test_dataset

    def _get_transform(self):
        # aa_policy = transforms.autoaugment.AutoAugmentPolicy("IMAGENET")
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self._IMAGE_HEIGHT, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.autoaugment.RandAugment(
                    interpolation=InterpolationMode.BILINEAR, magnitude=9
                ),
                # transforms.autoaugment.AutoAugment(
                #     policy=aa_policy, interpolation=InterpolationMode.BILINEAR
                # ),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=self._MEAN_RGB, std=self._STDDEV_RG),
            ]
        )
        return transform

    def _get_test_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(
                    size=[self._IMAGE_WIDTH, self._IMAGE_HEIGHT]
                ),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=self._MEAN_RGB, std=self._STDDEV_RGB),
            ]
        )
        return transform

    def _append_mix_up_to_train_kwargs(self) -> None:
        self.train_kwargs.update({"collate_fn": self._get_mix_up_collate_fn()})

    def _get_mix_up_collate_fn(self) -> callable:
        mixup_transforms = [
            RandomMixup(1000, p=1.0, alpha=0.2),
            RandomCutmix(1000, p=1.0, alpha=1.0),
        ]
        mixupcutmix = transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

        return collate_fn
