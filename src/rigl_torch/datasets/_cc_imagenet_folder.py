import pathlib
import yaml
from typing import Dict
from torchvision.datasets import ImageFolder

from torchvision.datasets.utils import verify_str_arg


class CCImageNetFolder(ImageFolder):
    _WNID_TO_CLASSES_FILE_NAME = "imagenet_labels.yaml"

    def __init__(
        self,
        root: str,
        split: str,
        meta_file_path: str,
        **kwargs,
    ):
        self.meta_file_path = meta_file_path
        self.split = verify_str_arg(split, "split", ("train", "validation"))
        root = pathlib.Path(root) / split
        super().__init__(root=root, **kwargs)
        wnid_to_classes = self._load_meta_file(self.meta_file_path)
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def _load_meta_file(self, meta_file_path: str) -> Dict[str, str]:
        meta_file = (
            pathlib.Path(meta_file_path) / self._WNID_TO_CLASSES_FILE_NAME
        )
        if not meta_file.is_file():
            raise FileNotFoundError(f"Unable to locate {meta_file}")
        with meta_file.open() as handle:
            meta_data = yaml.safe_load(handle)
        return meta_data
