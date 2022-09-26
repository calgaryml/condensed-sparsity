import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
import random
import dotenv
import omegaconf
import hydra
import logging
import wandb
from datetime import date
import pathlib
from typing import Dict, Any

from rigl_torch.models.model_factory import ModelFactory
from rigl_torch.rigl_scheduler import RigLScheduler
from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
from rigl_torch.datasets import get_dataloaders
from rigl_torch.optim import (
    get_optimizer,
    get_lr_scheduler,
)
from rigl_torch.utils.checkpoint import Checkpoint


def _get_checkpoint(cfg: omegaconf.DictConfig, rank: int, logger) -> Checkpoint:
    if cfg.experiment.run_id is None:
        raise ValueError(
            "Must provide wandb run_id when "
            "cfg.training.resume_from_checkpoint is True"
        )
    checkpoint = Checkpoint.load_last_checkpoint(
        run_id=cfg.experiment.run_id,
        parent_dir=cfg.paths.checkpoints,
        rank=rank,
    )
    logger.info(f"Resuming training with run_id: {cfg.experiment.run_id}")
    return checkpoint


def init_wandb(cfg: omegaconf.DictConfig, wandb_init_kwargs: Dict[str, Any]):
    run = wandb.init(
        name=cfg.experiment.name,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg=cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method=cfg.wandb.start_method),
        **wandb_init_kwargs,
    )
    return run


@hydra.main(config_path="configs/", config_name="config", version_base="1.2")
def initalize_main(cfg: omegaconf.DictConfig) -> None:
    if cfg.compute.distributed:
        wandb.setup()
        _validate_distributed_cfg(cfg)
        mp.spawn(
            main,
            args=(cfg,),
            nprocs=cfg.compute.world_size,
        )
    else:
        main(0, cfg)  # Single GPU


def _get_logger(rank, cfg: omegaconf.DictConfig) -> logging.Logger:
    log_path = pathlib.Path(cfg.paths.logs)
    logger = logging.getLogger(__file__)
    logger.setLevel(level=logging.INFO)
    current_date = date.today().strftime("%Y-%m-%d")
    # logformat = "[%(levelname)s] %(asctime)s G- %(name)s -%(rank)s - %(funcName)s (%(lineno)d) : %(message)s"
    logformat = "[%(levelname)s] %(asctime)s G- %(name)s - %(funcName)s (%(lineno)d) : %(message)s"
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format=logformat,
        handlers=[
            logging.FileHandler(log_path / f"processor_{current_date}.log"),
            logging.StreamHandler(),
        ],
    )
    # logger = logging.LoggerAdapter(logger, {"rank": f"rank: {rank}"})
    # logger.info("hell world")
    return logger


def main(rank: int, cfg: omegaconf.DictConfig) -> None:
    logger = _get_logger(rank, cfg)
    if cfg.experiment.resume_from_checkpoint:
        checkpoint = _get_checkpoint(cfg, rank, logger)
        wandb_init_resume = "must"
        run_id = checkpoint.run_id
        cfg = checkpoint.cfg
    else:
        run_id = None
        wandb_init_resume = "never"
        checkpoint = None
    logger.info(f"Running train_rigl.py with config:\n{cfg}")

    if cfg.compute.distributed:
        dist.init_process_group(
            backend=cfg.compute.dist_backend,
            world_size=cfg.compute.world_size,
            rank=rank,
        )
    run_id, optimizer_state, scheduler_state, pruner_state, model_state = (
        None,
        None,
        None,
        None,
        None,
    )

    if checkpoint is not None:
        run_id = checkpoint.run_id
        optimizer_state = checkpoint.optimizer
        scheduler_state = checkpoint.scheduler
        pruner_state = checkpoint.pruner
        model_state = checkpoint.model
        logger.info(f"Resuming training with run_id: {run_id}")
        cfg = checkpoint.cfg

    if rank == 0:
        wandb_init_kwargs = dict(resume=wandb_init_resume, id=run_id)
        run = init_wandb(cfg, wandb_init_kwargs)

    cfg = set_seed(cfg)
    use_cuda = not cfg.compute.no_cuda and torch.cuda.is_available()

    if cfg.compute.distributed:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_dataloaders(cfg)

    model = ModelFactory.load_model(
        model=cfg.model.name, dataset=cfg.dataset.name
    )
    if cfg.compute.distributed:
        model = DistributedDataParallel(model, device_ids=[rank])
    if model_state is not None:
        model.load_state_dict(model_state)
    model.to(device)
    optimizer = get_optimizer(cfg, model, state_dict=optimizer_state)
    scheduler = get_lr_scheduler(cfg, optimizer, state_dict=scheduler_state)

    pruner = None  # noqa: E731
    if cfg.rigl.dense_allocation is not None:
        T_end = int(0.75 * cfg.training.epochs * len(train_loader))
        if cfg.rigl.const_fan_in:
            rigl_scheduler = RigLConstFanScheduler
            logger.info("Using constant fan in rigl scheduler...")
        else:
            logger.info("Using vanilla rigl scheduler...")
            rigl_scheduler = RigLScheduler
        pruner = rigl_scheduler(
            model,
            optimizer,
            dense_allocation=cfg.rigl.dense_allocation,
            alpha=cfg.rigl.alpha,
            delta=cfg.rigl.delta,
            static_topo=cfg.rigl.static_topo,
            T_end=T_end,
            ignore_linear_layers=False,
            grad_accumulation_n=cfg.rigl.grad_accumulation_n,
            sparsity_distribution=cfg.rigl.sparsity_distribution,
            erk_power_scale=cfg.rigl.erk_power_scale,
            state_dict=pruner_state,
        )
    else:
        logger.warning(
            "cfg.rigl.dense_allocation is `null`, training with dense "
            "network..."
        )

    writer = SummaryWriter(log_dir="./graphs")
    if rank == 0:
        wandb.watch(model, criterion=F.nll_loss, log="all", log_freq=100)
    logger.info(f"Model Summary: {model}")
    if not cfg.experiment.resume_from_checkpoint:
        step = 0
        if rank == 0:
            checkpoint = Checkpoint(
                run_id=run.id,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                pruner=pruner,
                epoch=0,
                step=step,
                parent_dir=cfg.paths.checkpoints,
            )
        epoch_start = 1
    else:
        checkpoint.model = model
        checkpoint.optimizer = optimizer
        checkpoint.scheduler = scheduler
        checkpoint.pruner = pruner
        # Start at the next epoch after the last that successfully was saved
        epoch_start = checkpoint.epoch + 1
        step = checkpoint.step
    for epoch in range(epoch_start, cfg.training.epochs + 1):
        logger.info(pruner)
        if cfg.compute.distributed:
            train_loader.sampler.set_epoch(epoch)
        step = train(
            cfg,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            pruner=pruner,
            scheduler=scheduler,
            step=step,
            logger=logger,
        )
        loss, acc = test(
            cfg, model, device, test_loader, epoch, step, rank, logger
        )
        if rank == 0:
            writer.add_scalar("loss", loss, epoch)
            writer.add_scalar("accuracy", acc, epoch)
            wandb.log({"Learning Rate": scheduler.get_last_lr()[0]}, step=step)
            checkpoint.current_acc = acc
            checkpoint.step = step
            checkpoint.epoch = epoch
            checkpoint.save_checkpoint()
        if cfg.training.dry_run:
            break
        if cfg.training.max_steps is not None and step > cfg.training.max_steps:
            break

    if cfg.training.save_model and rank == 0:
        save_path = pathlib.Path(cfg.paths.artifacts)
        if not save_path.is_dir():
            save_path.mkdir()
        f_path = save_path / f"{cfg.experiment.name}.pt"
        torch.save(model.state_dict(), f_path)
        art = wandb.Artifact(name=cfg.experiment.name, type="model")
        art.add_file(f_path)
        logging.info(f"artifact path: {f_path}")
        wandb.log_artifact(art)
    if rank == 0:
        run.finish()


def train(
    cfg,
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    pruner,
    scheduler,
    step,
    logger,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        step += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        output = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()

        if pruner is not None and pruner():
            optimizer.step()
        scheduler.step()

        if batch_idx % cfg.training.log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        if cfg.training.dry_run:
            logger.warning("Dry run, exiting after one training step")
            return step
        if cfg.training.max_steps is not None and step > cfg.training.max_steps:
            return step
    return step


def test(cfg, model, device, test_loader, epoch, step, rank, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            output = F.log_softmax(logits, dim=1)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            )  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()
    if cfg.compute.distributed:
        if rank == 0:
            logger.info(f"test loss before all reduce: {test_loss}")
            logger.info(f"correct before all reduce: {correct}")
        dist.all_reduce(test_loss, dist.ReduceOp.AVG, async_op=False)
        dist.all_reduce(correct, dist.ReduceOp.SUM, async_op=False)
        if rank == 0:
            logger.info(f"test loss after all reduce: {test_loss}")
            logger.info(f"correct after all reduce: {correct}")
    logger.info(f"len of test loader: {len(test_loader.dataset)}")
    test_loss /= len(test_loader.dataset)
    if rank == 0:
        wandb_log(
            epoch,
            test_loss,
            correct / len(test_loader.dataset),
            data,
            output,
            target,
            pred,
            step,
        )
        logger.info(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return test_loss, correct / len(test_loader.dataset)


def wandb_log(epoch, loss, accuracy, inputs, logits, captions, pred, step):
    wandb.log(
        {
            "epoch": epoch,
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "inputs": wandb.Image(inputs),
            "logits": wandb.Histogram(logits.cpu()),
            "captions": wandb.Html(captions.cpu().numpy().__str__()),
            "predictions": wandb.Html(pred.cpu().numpy().__str__()),
        },
        step=step,
    )


def set_seed(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    if cfg.training.seed is None:
        cfg.training.seed = random.randint(0, 10000)
        logger = logging.getLogger(__file__)
        logger.info(
            f"No seed set in config! Generated random seed: {cfg.training.seed}"
        )
    pl.utilities.seed.seed_everything(cfg.training.seed)
    return cfg


def _validate_distributed_cfg(cfg: omegaconf.DictConfig) -> None:
    if cfg.compute.no_cuda:
        raise ValueError(
            "Cannot use distributed training with cfg.compute.no_cuda == True"
        )
    if not torch.cuda.is_available():
        raise ValueError("torch.cuda.is_available() returned False!")
    if cfg.compute.world_size < torch.cuda.device_count():
        raise ValueError(
            f"cfg.compute.world_size == {cfg.compute.world_size}"
            f" but I only see {torch.cuda.device_count()} cuda devices!"
        )
    return


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    import os

    print(os.environ["NUM_WORKERS"])
    initalize_main()
