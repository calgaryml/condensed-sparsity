from omegaconf import DictConfig
import torch

from rigl_torch.models.mnist import MnistNet
from rigl_torch.models.resnet import get_wide_resnet_22, get_resnet18
from functools import partial
from rigl_torch.models.model_factory import ModelFactory


def get_model(cfg: DictConfig) -> torch.nn.Module:
    model_loader = {
        "wideresnet22": partial(
            get_wide_resnet_22,
            num_classes=cfg.dataset.num_classes,
            width_multiplier=2,
        ),
        "mnist": partial(
            ModelFactory.get_model, cfg.model.name, cfg.dataset.name
        ),
        "resnet18": partial(
            ModelFactory.get_model, cfg.model.name, cfg.dataset.name
        )
        # partial(get_resnet18, num_classes=cfg.dataset.num_classes),
    }
    if cfg.model.name not in model_loader.keys():
        raise ValueError(
            f"{cfg.model.name} not an implemented model option. Choose from:\n {model_loader.keys()}"
        )
    model = model_loader[cfg.model.name]()
    return model
