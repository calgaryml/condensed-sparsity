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
from typing import Dict, Any, Optional
from copy import deepcopy

from rigl_torch.models.model_factory import ModelFactory
from rigl_torch.rigl_scheduler import RigLScheduler
from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
from rigl_torch.datasets import get_dataloaders
from rigl_torch.optim import (
    get_optimizer,
    get_lr_scheduler,
)
from rigl_torch.utils.checkpoint import Checkpoint
from rigl_torch.utils.rigl_utils import get_T_end


def _get_checkpoint(cfg: omegaconf.DictConfig, rank: int, logger) -> Checkpoint:
    run_id = cfg.experiment.run_id
    if run_id is None:
        raise ValueError(
            "Must provide wandb run_id when "
            "cfg.training.resume_from_checkpoint is True"
        )
    checkpoint = Checkpoint.load_last_checkpoint(
        run_id=run_id,
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
    use_cuda = not cfg.compute.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        raise SystemError("GPU has stopped responding...waiting to die!")
    if cfg.compute.distributed:
        # We initalize train and val loaders here to ensure .tar balls have
        # been decompressed before parallel workers try and write the same
        # directories!
        single_proc_cfg = deepcopy(cfg)
        single_proc_cfg.compute.distributed = False
        train_loader, test_loader = get_dataloaders(single_proc_cfg)
        del train_loader
        del test_loader
        del single_proc_cfg
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
    # logformat = "[%(levelname)s] %(asctime)s G- %(name)s -%(rank)s -
    # %(funcName)s (%(lineno)d) : %(message)s"
    logformat = (
        "[%(levelname)s] %(asctime)s G- %(name)s - %(funcName)s "
        "(%(lineno)d) : %(message)s"
    )
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
        cfg.experiment.run_id = run_id
        cfg.experiment.resume_from_checkpoint = True
    else:
        run_id = None
        wandb_init_resume = None
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
    if not use_cuda:
        raise SystemError("GPU has stopped responding...waiting to die!")
        logger.warning(
            "Using CPU! Verify cfg.compute.no_cuda and "
            "torch.cuda.is_available() are properly set if this is unexpected"
        )

    if cfg.compute.distributed:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_dataloaders(cfg)

    model = ModelFactory.load_model(
        model=cfg.model.name, dataset=cfg.dataset.name
    )
    model.to(device)
    if cfg.compute.distributed:
        model = DistributedDataParallel(model, device_ids=[rank])
    if model_state is not None:
        model.load_state_dict(model_state)
    optimizer = get_optimizer(cfg, model, state_dict=optimizer_state)
    scheduler = get_lr_scheduler(cfg, optimizer, state_dict=scheduler_state)

    if "filter_ablation_threshold" not in cfg.rigl:
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg.rigl.filter_ablation_threshold = None
    pruner = None
    if cfg.rigl.dense_allocation is not None:
        T_end = get_T_end(cfg, train_loader)
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
            filter_ablation_threshold=cfg.rigl.filter_ablation_threshold,
            static_ablation=cfg.rigl.static_ablation,
            dynamic_ablation=cfg.rigl.dynamic_ablation,
            min_salient_weights_per_neuron=cfg.rigl.min_salient_weights_per_neuron,  # noqa
        )
    else:
        logger.warning(
            "cfg.rigl.dense_allocation is `null`, training with dense "
            "network..."
        )

    writer = SummaryWriter(log_dir="./graphs")
    if rank == 0:
        if cfg.wandb.watch_model_grad_and_weights:
            log = "all"
        else:
            log = None
        wandb.watch(
            model,
            criterion=F.nll_loss,
            log=log,
            log_freq=cfg.training.log_interval,
        )
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
        if pruner is not None:
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
            step=step,
            logger=logger,
            rank=rank,
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
        scheduler.step()

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
    step,
    logger,
    rank,
):
    model.train()
    steps_to_accumulate_grad = _get_steps_to_accumulate_grad(
        cfg.training.simulated_batch_size, cfg.training.batch_size
    )
    for batch_idx, (data, target) in enumerate(train_loader):
        apply_grads = (
            True
            if steps_to_accumulate_grad == 1
            or (batch_idx != 0 and batch_idx % steps_to_accumulate_grad == 0)
            else False
        )
        step += 1
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = F.cross_entropy(
            logits,
            target,
            label_smoothing=cfg.training.label_smoothing,
        )
        loss.backward()

        if apply_grads:
            optimizer.step()
            if pruner is not None:
                # pruner.__call__ returns False if rigl step taken
                if not pruner() and cfg.wandb.log_filter_stats:
                    # If we update the pruner
                    # log filter-wise statistics to wandb
                    pruner.log_meters(step=step)
            optimizer.zero_grad()

        if batch_idx % cfg.training.log_interval == 0 and rank == 0:
            world_size = (
                1
                if cfg.compute.distributed is False
                else cfg.compute.world_size
            )
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data) * world_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            wandb.log({"ITOP Rate": pruner.itop_rs}, step=step)
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
    top_k_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            test_loss += F.cross_entropy(
                logits,
                target,
                label_smoothing=cfg.training.label_smoothing,
                reduction="mean",
            )
            pred = logits.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()
            if cfg.dataset.name == "imagenet":
                _, top_5_indices = torch.topk(logits, k=5, dim=1, largest=True)
                top_5_pred = (
                    target.reshape(-1, 1).expand_as(top_5_indices)
                    == top_5_indices
                ).any(dim=1)
                top_k_correct += top_5_pred.sum()
            else:
                top_k_correct = None
    if cfg.compute.distributed:
        dist.all_reduce(test_loss, dist.ReduceOp.AVG, async_op=False)
        dist.all_reduce(correct, dist.ReduceOp.SUM, async_op=False)
        if cfg.dataset.name == "imagenet":
            dist.all_reduce(top_k_correct, dist.ReduceOp.SUM, async_op=False)
            top_k_correct = top_k_correct / len(test_loader.dataset)
    if rank == 0:
        wandb_log(
            epoch,
            test_loss,
            correct / len(test_loader.dataset),
            top_k_correct,
            data,
            logits,
            target,
            pred,
            step,
            cfg.wandb.log_images,
        )
        logger.info(
            (
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
            ).format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return test_loss, correct / len(test_loader.dataset)


def wandb_log(
    epoch,
    loss,
    accuracy,
    top_k_accuracy: Optional[torch.Tensor],
    inputs,
    logits,
    captions,
    pred,
    step,
    log_images,
):
    log_data = {
        "epoch": epoch,
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "logits": wandb.Histogram(logits.cpu()),
    }
    if top_k_accuracy is not None:
        log_data.update({"top_5_accuracy": top_k_accuracy.item()})
    if log_images:
        log_data.update(
            {
                "inputs": wandb.Image(inputs),
                "captions": wandb.Html(captions.cpu().numpy().__str__()),
                "predictions": wandb.Html(pred.cpu().numpy().__str__()),
            }
        )
    wandb.log(log_data, step=step)


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
    if cfg.compute.world_size > torch.cuda.device_count():
        raise ValueError(
            f"cfg.compute.world_size == {cfg.compute.world_size}"
            f" but I only see {torch.cuda.device_count()} cuda devices!"
        )
    return


def _get_steps_to_accumulate_grad(
    simulated_batch_size: int, batch_size: int
) -> int:
    if simulated_batch_size is None:
        return 1
    if simulated_batch_size % batch_size != 0:
        raise ValueError(
            "Effective batch size must be a multiple of batch size! "
            f"{simulated_batch_size} % {batch_size} !=0"
        )
    return int(simulated_batch_size / batch_size)


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    initalize_main()
