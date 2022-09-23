from rigl_torch.datasets import _data_stem
from typing import Dict, Any
from torchvision import transforms, datasets


class MNISTDataStem(_data_stem.ABCDataStem):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

    def _get_datasets(self):
        transform = self._get_transform()
        train_dataset = datasets.MNIST(
            self.data_path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            self.data_path, train=False, transform=transform
        )
        return train_dataset, test_dataset

    def _get_transform(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return transform
