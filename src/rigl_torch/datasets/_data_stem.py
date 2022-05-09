from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class ABCDataStem(ABC):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.initalize_from_cfg()

    def initalize_from_cfg(self) -> None:
        self.use_cuda = not self.cfg.compute.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.cfg.training.seed)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.train_kwargs = {"batch_size": self.cfg.training.batch_size}
        self.test_kwargs = {"batch_size": self.cfg.training.test_batch_size}
        if self.use_cuda:
            self.train_kwargs.update(self.cfg.compute.cuda_kwargs)
            self.test_kwargs.update(self.cfg.compute.cuda_kwargs)
        self.data_path = self.cfg.paths.data_folder

    @abstractmethod
    def get_train_test_loaders(self):
        ...

    @abstractmethod
    def _get_transform(self):
        ...
