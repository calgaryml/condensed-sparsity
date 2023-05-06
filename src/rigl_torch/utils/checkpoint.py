from __future__ import annotations
from typing import Optional, Union, Dict, Any
import pathlib
import torch
import torch.nn as nn
from dataclasses import dataclass
from omegaconf import DictConfig
import datetime
import logging
import glob

from rigl_torch.rigl_scheduler import RigLScheduler


@dataclass(init=True)
class Checkpoint(object):
    run_id: str
    cfg: DictConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    pruner: Optional[RigLScheduler] = None
    epoch: int = 0
    step: int = 0
    best_acc: float = -float("inf")
    current_acc: float = 0.0
    is_best: bool = False
    checkpoint_dir: Optional[Union[pathlib.Path, str]] = None
    f_name: str = "checkpoint.pt.tar"
    best_file_name: Optional[str] = f"best-{f_name}"
    parent_dir: pathlib.Path = pathlib.Path.cwd()
    _logger = logging.getLogger(__name__)
    _RUN_ID_DELIMITER: str = "_"

    def __post_init__(self) -> None:
        self.checkpoint_dir = self._format_checkpoint_dir(self.checkpoint_dir)

    def _format_checkpoint_dir(
        self, checkpoint_dir: Optional[Union[pathlib.Path, str]]
    ) -> pathlib.Path:
        if checkpoint_dir is None:
            dir_name = (
                f"{datetime.datetime.today().strftime('%Y%m%d')}"
                f"{self._RUN_ID_DELIMITER}{self.run_id}"
            )
            checkpoint_dir = (
                pathlib.Path(self.cfg.paths.artifacts)
                / "checkpoints"
                / dir_name
            )
        if type(checkpoint_dir) == str:
            checkpoint_dir = pathlib.Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
        return checkpoint_dir

    def save_checkpoint(self) -> None:
        state = self.get_state()
        torch.save(state, self.checkpoint_dir / self.f_name)
        self._logger.info("Checkpoint state saved!")
        if self.is_best:
            self._save_best_checkpoint(state)

    def _save_best_checkpoint(self, state: Dict[str, Any]) -> None:
        best_file_name = f"best-{self.f_name}"
        torch.save(state, self.checkpoint_dir / best_file_name)
        self._logger.info("Best checkpoint state saved!")

    def get_state(self) -> Dict[str, Any]:
        self._update_best_flag()
        state = {}
        for attr_name, attr_obj in self.__dict__.items():
            if attr_name[0] == "_":
                continue
            if attr_obj is not None and "state_dict" in attr_obj.__dir__():
                state.update({attr_name: attr_obj.state_dict()})
            else:
                state.update({attr_name: attr_obj})
        return state

    def _update_best_flag(self) -> None:
        if self.current_acc > self.best_acc:
            self._logger.info(
                "New best checkpoint accuracy "
                f"({self.current_acc:4f} > {self.best_acc:4f})!"
            )
            self.is_best = True
            self.best_acc = self.current_acc
        else:
            self.is_best = False
        return

    @classmethod
    def load_best_checkpoint(
        cls,
        checkpoint_dir: Optional[Union[pathlib.Path, str]] = None,
        run_id: str = None,
        rank: int = 0,
    ) -> Checkpoint:
        checkpoint_dir = cls._get_existing_checkpoint_dir(
            checkpoint_dir, run_id
        )
        return cls._load_checkpoint(
            cls.best_file_name, checkpoint_dir, run_id, rank
        )

    @classmethod
    def load_last_checkpoint(
        cls,
        checkpoint_dir: Optional[Union[pathlib.Path, str]] = None,
        parent_dir: Optional[Union[pathlib.Path, str]] = None,
        run_id: str = None,
        rank: int = 0,
    ) -> Checkpoint:
        if parent_dir is not None:
            cls.parent_dir = pathlib.Path(parent_dir)
        return cls._load_checkpoint(cls.f_name, checkpoint_dir, run_id, rank)

    @classmethod
    def _load_checkpoint(
        cls,
        f_name: str,
        checkpoint_dir: Optional[Union[pathlib.Path, str]] = None,
        run_id: str = None,
        rank: int = 0,
    ) -> Checkpoint:
        checkpoint_dir = cls._get_existing_checkpoint_dir(
            checkpoint_dir, run_id
        )
        checkpoint_path = checkpoint_dir / f_name
        if not checkpoint_path.is_file():
            raise ValueError(f"{checkpoint_path} not found!")
        cls._logger.info(f"Loading checkpoint from {checkpoint_path}...")
        if torch.cuda.is_available():
            map_location = f"cuda:{rank}"
        else:
            map_location = "cpu"
        state = torch.load(checkpoint_path, map_location=map_location)
        return Checkpoint(**state)

    @classmethod
    def _get_existing_checkpoint_dir(
        cls,
        checkpoint_dir: Optional[Union[pathlib.Path, str]] = None,
        run_id: str = None,
    ) -> None:
        if checkpoint_dir is None:
            if run_id is None:
                raise ValueError("Must provide checkpoint_dir or run_id!")
            checkpoint_dir = cls._get_checkpoint_dir_from_run_id(run_id)
        else:
            if type(checkpoint_dir) == str:
                checkpoint_dir = pathlib.Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise ValueError(
                f"Checkpoint dir: {checkpoint_dir.__str__()} not found!"
            )
        return checkpoint_dir

    @classmethod
    def _get_checkpoint_dir_from_run_id(cls, run_id: str) -> pathlib.Path:
        dir_glob = glob.glob((cls.parent_dir / "*").__str__())
        for dir in dir_glob:
            if dir.split(cls._RUN_ID_DELIMITER)[-1] == run_id:
                return pathlib.Path(dir)
        raise ValueError(
            f"run_id :{run_id} not found in {cls.parent_dir}. "
            f"I can see files: {dir_glob}"
        )

    def get_single_process_model_state_from_distributed_state(
        self,
    ) -> Dict[str, torch.Tensor]:
        """Returns model state from a distributed training run in a format
        suitable for a single process model.

        In distributed training, `module.<param_name>` is appended to every
        parameter. If we wish to test/train this model further in a single
        process, we simply strip the `module` prefix to match keys expected in
        the model.

        Returns:
            Dict[str, torch.Tensor]: Model state ordered dict from distributed
                trained model for loading in single process.
        """
        return {".".join(k.split(".")[1:]): v for k, v in self.model.items()}
