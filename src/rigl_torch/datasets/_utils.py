import torch
from omegaconf import DictConfig

from rigl_torch.datasets._mnist import MNISTDataStem
from rigl_torch.datasets._cifar import CIFAR10DataStem
from rigl_torch.datasets._imagenet import ImageNetDataStem


def get_dataloaders(cfg: DictConfig) -> torch.utils.data.DataLoader:
    if cfg.dataset.name.lower() == "mnist":
        data_stem = MNISTDataStem(cfg)
    elif cfg.dataset.name.lower() == "cifar10":
        data_stem = CIFAR10DataStem(cfg)
    elif cfg.dataset.name.lower() == "imagenet":
        data_stem = ImageNetDataStem(cfg, data_path_override=cfg.dataset.root)
    else:
        raise ValueError(
            f"{cfg.dataset.name.lower()} is not a recognized dataset name!"
        )
    train_dataloader, test_dataloader = data_stem.get_train_test_loaders()
    return train_dataloader, test_dataloader
