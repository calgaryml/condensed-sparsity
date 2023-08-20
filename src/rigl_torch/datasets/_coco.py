from typing import Dict, Any, Optional, Union
import torch
import pathlib
from collections import defaultdict
import PIL.Image

from rigl_torch.datasets import _data_stem

import torchvision

# We disable the beta transforms warning as it will print many times
torchvision.disable_beta_transforms_warning()
from torchvision import datasets  # noqa: E402
import torchvision.transforms.v2 as transforms  # noqa: E402


class CocoSegmentationDataStem(_data_stem.ABCDataStem):
    def __init__(
        self,
        cfg: Dict[str, Any],
        data_path_override: Optional[Union[str, pathlib.Path]] = None,
    ):
        super().__init__(cfg, data_path_override)

    def _get_datasets(self):
        train_transformer = self._get_transform()
        test_transformer = self._get_test_transform()

        train_dataset = datasets.CocoDetection(
            root=self.data_path / "train2014",
            annFile=self.data_path / "annotations" / "instances_train2014.json",
            transforms=train_transformer,
        )
        test_dataset = datasets.CocoDetection(
            root=self.data_path / "val2014",
            annFile=self.data_path / "annotations" / "instances_val2014.json",
            transforms=test_transformer,
        )
        # NOTE: We need to wrap datasets for v2 transformers.
        # See: https://pytorch.org/vision/0.15/auto_examples/plot_transforms_v2_e2e.html  # noqa
        train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset)
        test_dataset = datasets.wrap_dataset_for_transforms_v2(test_dataset)
        self._append_collate_fn_to_dataloader_kwargs()
        return train_dataset, test_dataset

    def _get_transform(self):
        train_transform = transforms.Compose(
            [
                transforms.RandomPhotometricDistort(),
                transforms.RandomZoomOut(
                    fill=defaultdict(
                        lambda: 0, {PIL.Image.Image: (123, 117, 104)}
                    )
                ),
                # transforms.RandomIoUCrop(),  # Deleting lots of bboxes
                transforms.RandomHorizontalFlip(),
                transforms.ToImageTensor(),
                transforms.ConvertImageDtype(torch.float32),
                # transforms.SanitizeBoundingBox(),  # Throwing exceptions
            ]
        )
        return train_transform

    def _get_test_transform(self):
        test_transform = transforms.Compose(
            [
                transforms.ToImageTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )
        return test_transform

    def _append_collate_fn_to_dataloader_kwargs(self) -> None:
        def collate_fn(batch):
            return tuple(zip(*batch))

        self.train_kwargs.update({"collate_fn": collate_fn})
        self.test_kwargs.update({"collate_fn": collate_fn})
