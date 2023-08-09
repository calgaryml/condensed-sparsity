from typing import Dict, Any, Optional, Union
from torchvision import datasets
import pathlib
from rigl_torch.datasets import _data_stem


class CocoSegmentationDataStem(_data_stem.ABCDataStem):
    def __init__(self, cfg: Dict[str, Any], data_path_override: Optional[Union[str, pathlib.Path]] = None,):
        super().__init__(cfg, data_path_override)

    def _get_datasets(self):
        train_dataset = datasets.CocoDetection(
            root=self.data_path / "train2014",
            annFile=self.data_path / "annotations" / "instances_train2014.json"
        )
        test_dataset = datasets.CocoDetection(
            root=self.data_path / "val2014",
            annFile=self.data_path / "annotations" / "instances_val2014.json"
        )

        return train_dataset, test_dataset

    def _get_transform(self):
        return None
