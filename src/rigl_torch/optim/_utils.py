import torch
from omegaconf import DictConfig, OmegaConf
from functools import partial

from rigl_torch.models import ModelFactory
from .cosine_annealing_with_linear_warm_up import (
    CosineAnnealingWithLinearWarmUp,
)


def get_optimizer(cfg: OmegaConf, model) -> torch.optim.Optimizer:
    optimizers = {
        "sgd": partial(
            torch.optim.SGD,
            params=model.parameters(),
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            dampening=0,
            weight_decay=cfg.training.weight_decay,
            nesterov=True,
            maximize=False,
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
        ),
    }
    if cfg.training.optimizer.lower() not in optimizers:
        raise ValueError(
            f"Unrecongized optmizier: {cfg.training.optimizer}. Select from: {list(optimizers.keys())}"
        )
    else:
        return optimizers[cfg.training.optimizer.lower()]()


def get_lr_scheduler(
    cfg: OmegaConf, optim: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    schedulers = {
        "step_lr": partial(
            torch.optim.lr_scheduler.StepLR,
            optimizer=optim,
            step_size=cfg.training.step_size,
            gamma=cfg.training.gamma,
        ),
        # "step_lr_with_warm_up": partial( # For imagnet
        #     torch.optim.Adadelta,
        #     params=model.parameters(),
        #     lr=cfg.training.lr,
        #     weight_decay=cfg.training.weight_decay,
        # ),
        "cosine_annealing_with_warm_up": partial(
            CosineAnnealingWithLinearWarmUp,
            optimizer=optim,
<<<<<<< HEAD
            T_max=cfg.training.epochs * cfg.dataset.train_len,
            eta_min=0,
            lr=cfg.training.lr,
            warm_up_steps=cfg.training.warm_up_steps * cfg.training.batch_size,
=======
            T_max=cfg.training.epochs,
            eta_min=0,
            lr=cfg.training.lr,
            warm_up_steps=cfg.training.warm_up_steps,
>>>>>>> lr scheduler options added
        ),
    }
    if cfg.training.scheduler.lower() not in list(schedulers.keys()):
        raise ValueError(
            f"{cfg.training.scheduler.lower()} is not a valid scheudler. "
            f"Select from: {list(schedulers.keys())} "
        )
    else:
        return schedulers[cfg.training.scheduler.lower()]()


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
