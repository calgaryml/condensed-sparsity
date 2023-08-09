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
        if self.cfg.compute.distributed:
            self.train_kwargs = {
                "batch_size": int(
                    self.cfg.training.batch_size / self.cfg.compute.world_size
                ),
                "shuffle": False,
            }
            self.test_kwargs = {
                "batch_size": int(
                    self.cfg.training.test_batch_size
                    / self.cfg.compute.world_size
                ),
            }
        else:
            self.train_kwargs = {
                "batch_size": self.cfg.training.batch_size,
                "shuffle": True,
            }
            self.test_kwargs = {
                "batch_size": self.cfg.training.test_batch_size,
            }
        if self.use_cuda:
            self.train_kwargs.update(self.cfg.compute.cuda_kwargs)
            self.test_kwargs.update(self.cfg.compute.cuda_kwargs)
        if self._data_path_override is not None:
            self.data_path = self._data_path_override
        else:
            self.data_path = self.cfg.paths.data_folder
        self.data_path = pathlib.Path(self.data_path)

    def get_train_test_loaders(self):
        train_dataset, test_dataset = self._get_datasets()
        if self.cfg.compute.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, shuffle=False
            )
        else:
            train_sampler = None
            test_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            drop_last=True,
            **self.train_kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            sampler=test_sampler,
            drop_last=False,
            **self.test_kwargs
        )
        return train_loader, test_loader

    @abstractmethod
    def _get_datasets(self):
        ...

    @abstractmethod
    def _get_transform(self):
        ...
