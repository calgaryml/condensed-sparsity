import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchvision
from rigl_torch.datasets import get_dataloaders
from omegaconf import DictConfig
import dotenv
from rigl_torch.models import ModelFactory
import hydra
from rigl_torch.datasets import get_dataloaders


def main(rank, cfg):
    dist.init_process_group(
        backend=cfg.compute.dist_backend,
        world_size=cfg.compute.world_size,
        rank=rank,
    )
    # cifar = torchvision.datasets.CIFAR10(
    #     root="/home/condensed-sparsity/data", download=True
    # )
    # device = torch.device(f"cuda:{rank}")
    # train_sampler = torch.utils.data.distributed.DistributedSampler(cifar)
    # train_loader = torch.utils.data.DataLoader(
    #     cifar,
    #     sampler=train_sampler,
    #     batch_size=cfg.training.batch_size,
    #     shuffle=False,
    #     drop_last=True,
    # )
    train_loader, test_loader = get_dataloaders(cfg)
    print(
        f"Num samples in rank {rank}: {len(train_loader)*train_loader.batch_size}"
    )
    print(f"dataset length: {len(train_loader.dataset)}")
    dist.barrier()
    print(
        f"Num test samples in rank {rank}: {len(test_loader)*test_loader.batch_size}"
    )
    print(f"dataset length: {len(test_loader.dataset)}")


if __name__ == "__main__":
    dotenv.load_dotenv(
        dotenv_path="/home/condensed-sparsity/.env", override=True
    )
    import os

    print(os.environ["MASTER_PORT"])
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(
            config_name="config.yaml",
            # overrides=["dataset=cifar10", "model=wide_resnet22"],
        )
    mp.spawn(
        main,
        args=(cfg,),
        nprocs=cfg.compute.world_size,
    )
