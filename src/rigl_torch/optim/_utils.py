import torch
from omegaconf import DictConfig, OmegaConf
from functools import partial
from typing import Optional, Dict, Any

from rigl_torch.models import ModelFactory
from .cosine_annealing_with_linear_warm_up import (
    CosineAnnealingWithLinearWarmUp,
)
from .step_lr_with_linear_warm_up import StepLrWithLinearWarmUp


def get_optimizer(
    cfg: OmegaConf, model, state_dict: Optional[Dict[str, Any]] = None
) -> torch.optim.Optimizer:
    optimizers = {
        "sgd": partial(
            torch.optim.SGD,
            params=model.parameters(),
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            dampening=0,
            weight_decay=cfg.training.weight_decay,
            nesterov=True,
            # maximize=False,
        ),
        "adadelta": partial(
            torch.optim.Adadelta,
            params=model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        ),
        "adamw": partial(
            torch.optim.AdamW,
            params=model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            betas=cfg.training.betas,
        ),
        "adam": partial(
            torch.optim.Adam,
            params=model.parameters(),
            lr=cfg.training.lr,
            betas=cfg.training.betas,
            weight_decay=cfg.training.weight_decay,
        ),
    }
    if cfg.training.optimizer.lower() not in optimizers:
        raise ValueError(
            f"Unrecongized optmizier: {cfg.training.optimizer}."
            f" Select from: {list(optimizers.keys())}"
        )
    else:
        optim = optimizers[cfg.training.optimizer.lower()]()
        if state_dict is not None:
            optim.load_state_dict(state_dict)
        return optim


def get_lr_scheduler(
    cfg: OmegaConf,
    optim: torch.optim.Optimizer,
    state_dict: Optional[Dict[str, Any]] = None,
    logger=None,
) -> torch.optim.lr_scheduler._LRScheduler:
    if state_dict is not None:
        last_epoch = state_dict["last_epoch"]
    else:
        last_epoch = -1
    schedulers = {
        "step_lr": partial(
            torch.optim.lr_scheduler.StepLR,
            optimizer=optim,
            step_size=cfg.training.step_size,
            gamma=cfg.training.gamma,
            last_epoch=last_epoch,
        ),
        "step_lr_with_warm_up": partial(  # For imagnet
            StepLrWithLinearWarmUp,
            optimizer=optim,
            step_size=cfg.training.step_size,
            warm_up_steps=cfg.training.warm_up_steps,
            gamma=cfg.training.gamma,
            init_lr=cfg.training.init_lr,
            lr=cfg.training.lr,
            last_epoch=last_epoch,
            logger=logger,
        ),
        "cosine_annealing_with_warm_up": partial(
            CosineAnnealingWithLinearWarmUp,
            optimizer=optim,
            T_max=cfg.training.epochs,
            eta_min=0,
            lr=cfg.training.lr,
            warm_up_steps=cfg.training.warm_up_steps,
            last_epoch=last_epoch,
        ),
    }
    if cfg.training.scheduler.lower() not in list(schedulers.keys()):
        raise ValueError(
            f"{cfg.training.scheduler.lower()} is not a valid scheudler. "
            f"Select from: {list(schedulers.keys())} "
        )
    else:
        sch = schedulers[cfg.training.scheduler.lower()]()
        if state_dict is not None:
            sch.load_state_dict(state_dict)
        return sch


if __name__ == "__main__":
    cfg = {
        "training": {
            "optimizer": "adamW",
            "momentum": 1.0,
            "weight_decay": 1e-4,
            "lr": 1.0,
        }
    }
    cfg = DictConfig(cfg)
    print(cfg.training.optimizer)
    model = ModelFactory.load_model("mnist", "mnist")
    optim = get_optimizer(cfg, model)
    print(optim)
