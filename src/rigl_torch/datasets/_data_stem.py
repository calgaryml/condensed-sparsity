from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import pathlib
import torch


class ABCDataStem(ABC):
    def __init__(
        self,
        cfg: Dict[str, Any],
        data_path_override: Optional[Union[str, pathlib.Path]] = None,
    ):
        self.cfg = cfg
        self._data_path_override = data_path_override
        self._initalize_from_cfg()

    def _initalize_from_cfg(self) -> None:
        self.use_cuda = (
            not self.cfg.compute.no_cuda and torch.cuda.is_available()
        )
        torch.manual_seed(self.cfg.training.seed)
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # train_gen = self._get_generator(seed=self.cfg.training.seed)
        # test_gen = self._get_generator(seed=self.cfg.training.seed)
        self.train_kwargs = {
            "batch_size": self.cfg.training.batch_size,
            # "generator": train_gen,
            "shuffle": True,
        }
        self.test_kwargs = {
            "batch_size": self.cfg.training.test_batch_size,
            # "generator": test_gen,
        }
        if self.use_cuda:
            self.train_kwargs.update(self.cfg.compute.cuda_kwargs)
            self.test_kwargs.update(self.cfg.compute.cuda_kwargs)
        if self._data_path_override is not None:
            self.data_path = self._data_path_override
        else:
            self.data_path = self.cfg.paths.data_folder

    def _get_generator(self, seed: int) -> torch.Generator:
        device_string = "cuda" if self.use_cuda else "cpu"
        gen = torch.Generator(device=device_string)
        gen.manual_seed(seed)
        return gen

    @abstractmethod
    def get_train_test_loaders(self):
        ...

    @abstractmethod
    def _get_transform(self):
        ...
