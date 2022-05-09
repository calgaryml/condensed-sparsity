from rigl_torch.datasets import _data_stem
from typing import Dict, Any
import torch
from torchvision import transforms, datasets


class MnistDataStem(_data_stem.ABCDataStem):

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

    def get_train_test_loaders(self):
        transform = self._get_transform()
        train_dataset = datasets.MNIST(
            self.data_path, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(self.data_path, train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, **self.train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **self.test_kwargs)
        return train_loader, test_loader

    def _get_transform(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        return transform
