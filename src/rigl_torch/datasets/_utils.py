import torch
from rigl_torch.datasets._mnist import MnistDataStem
from rigl_torch.datasets._cifar import Cifar10DataStem
from omegaconf import DictConfig

# from rigl_torch.datasets._mnist import MnistDataStem


def get_dataloaders(cfg: DictConfig) -> torch.utils.data.DataLoader:
    if cfg.dataset.name.lower() == "mnist":
        data_stem = MnistDataStem(cfg)
    elif cfg.dataset.name.lower() == "cifar10":
        data_stem = Cifar10DataStem(cfg)
    elif cfg.dataset.name.lower() == "imagenet":
        raise NotImplementedError("Imagenet stem not yet implemented")
    else:
        raise ValueError(
            f"{cfg.dataset.name.lower()} is not a recognized dataset name!"
        )
    train_dataloader, test_dataloader = data_stem.get_train_test_loaders()
    return train_dataloader, test_dataloader
