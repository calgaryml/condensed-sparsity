from omegaconf import DictConfig
import torch

from rigl_torch.models.mnist import MnistNet
from rigl_torch.models.resnet18 import get_resnet18


def get_model(cfg: DictConfig) -> torch.nn.Module:
    model_loader = {
        "resnet18": get_resnet18,
        "mnist": lambda: MnistNet(),
    }
    if cfg.model.name not in model_loader.keys():
        raise ValueError(
            f"{cfg.model.name} not an implemented model option. Choose from:\n {model_loader.keys()}"
        )
    model = model_loader[cfg.model.name]()
    return model
