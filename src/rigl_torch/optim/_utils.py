import torch
from omegaconf import DictConfig, OmegaConf
from functools import partial

from rigl_torch.models import ModelFactory


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
