"""Modified detection dataset that can be pickled. See links below for more info
https://github.com/pytorch/vision/issues/6753#:~:text=The%20transforms%20v2%20API%20looks%20very%20nice
https://github.com/pytorch/vision/pull/7860
"""
from typing import Any, Tuple, Callable, Optional, List

import torch
from pycocotools import mask
from torchvision import datapoints
import logging
from torchvision.datapoints._dataset_wrapper import (
    list_of_dicts_to_dict_of_lists,
)
from torchvision.datasets import CocoDetection
from torchvision.transforms.v2 import functional as F


class CocoDetectionV2(CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transforms: Callable[..., Any] | None = None,
        no_add_ids: Optional[List[int]] = None,
    ) -> None:
        self.__logger = logging.getLogger(__name__)
        super().__init__(root, annFile)
        self.v2_transforms = transforms
        self.__logger.info(
            f"COCO dataset size before removal of missing anns {len(self.ids)}"
        )
        valid_ids = [id for id in self.ids if id not in no_add_ids]
        self.ids = valid_ids
        self.__logger.info(
            f"COCO dataset size after removal of missing anns: {len(self.ids)}"
        )
        self.no_add_ids = no_add_ids

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = super().__getitem__(index)
        sample = self.wrapper(index, sample)
        if self.v2_transforms is not None:
            sample = self.v2_transforms(*sample)
        return sample

    def segmentation_to_mask(self, segmentation, *, spatial_size):
        """Copied from `torchvision/datapoints/_dataset_wrapper.py`"""

        segmentation = (
            mask.frPyObjects(segmentation, *spatial_size)
            if isinstance(segmentation, dict)
            else mask.merge(mask.frPyObjects(segmentation, *spatial_size))
        )
        return torch.from_numpy(mask.decode(segmentation))

    def wrapper(self, idx, sample):
        """Copied from `torchvision/datapoints/_dataset_wrapper.py`"""
        image_id = self.ids[idx]

        image, target = sample

        if not target:
            return image, dict(image_id=image_id)

        batched_target = list_of_dicts_to_dict_of_lists(target)

        batched_target["image_id"] = image_id

        spatial_size = tuple(F.get_spatial_size(image))
        batched_target["boxes"] = F.convert_format_bounding_box(
            datapoints.BoundingBox(
                batched_target["bbox"],
                format=datapoints.BoundingBoxFormat.XYWH,
                spatial_size=spatial_size,
            ),
            new_format=datapoints.BoundingBoxFormat.XYXY,
        )
        batched_target["masks"] = datapoints.Mask(
            torch.stack(
                [
                    self.segmentation_to_mask(
                        segmentation, spatial_size=spatial_size
                    )
                    for segmentation in batched_target["segmentation"]
                ]
            ),
        )
        batched_target["labels"] = torch.tensor(batched_target["category_id"])
        return image, batched_target


def collate(batch):
    return tuple(zip(*batch))
