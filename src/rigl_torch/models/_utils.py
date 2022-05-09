from omegaconf import DictConfig
import torch

from rigl_torch.models.mnist import MnistNet
from rigl_torch.models.resnet import get_wide_resnet_22
from functools import partial


def get_model(cfg: DictConfig) -> torch.nn.Module:
    model_loader = {
        "wideresnet22": partial(
            get_wide_resnet_22,
            num_classes=cfg.dataset.num_classes,
            width_multiplier=2,
        ),
        "mnist": lambda: MnistNet(),
    }
    if cfg.model.name not in model_loader.keys():
        raise ValueError(
            f"{cfg.model.name} not an implemented model option. Choose from:\n {model_loader.keys()}"
        )
    model = model_loader[cfg.model.name]()
    return model