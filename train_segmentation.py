import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import pytorch_lightning as pl
import random
import dotenv
import omegaconf
from datetime import datetime
import hydra
import logging
import wandb
import pathlib

from rigl_torch.models.model_factory import ModelFactory
from rigl_torch.rigl_scheduler import RigLScheduler
from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
from rigl_torch.datasets import get_dataloaders
from rigl_torch.optim import (
    get_optimizer,
    get_lr_scheduler,
)
from rigl_torch.utils.checkpoint import Checkpoint, get_checkpoint
from rigl_torch.utils.rigl_utils import get_T_end
from rigl_torch.meters import SegmentationMeter
from rigl_torch.utils.wandb_utils import init_wandb
from rigl_torch.utils.dist_utils import get_steps_to_accumulate_grad
from rigl_torch.utils.logging_utils import get_logger
from rigl_torch.utils.coco_eval import CocoEvaluator
from rigl_torch.utils.coco_utils import show_gt_vs_dt


@hydra.main(config_path="configs/", config_name="config", version_base="1.2")
def initalize_main(cfg: omegaconf.DictConfig) -> None:
    use_cuda = not cfg.compute.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        raise SystemError("GPU has stopped responding...waiting to die!")
    if cfg.training.max_steps in ["None", "null"]:
        cfg.training.max_steps = None
    if cfg.rigl.dense_allocation in ["None", "null"]:
        cfg.rigl.dense_allocation = None
    if "diet" not in cfg.rigl:
        with omegaconf.open_dict(cfg):
            cfg.rigl.diet = None
    if "keep_first_layer_dense" not in cfg.rigl:
        with omegaconf.open_dict(cfg):
            cfg.rigl.keep_first_layer_dense = False
    if "initialize_grown_weights" not in cfg.rigl:
        with omegaconf.open_dict(cfg):
            cfg.rigl.initialize_grown_weights = 0.0

    if cfg.compute.distributed:
        wandb.setup()
        _validate_distributed_cfg(cfg)
        mp.set_start_method("spawn")
        mp.spawn(
            main,
            args=(cfg,),
            nprocs=cfg.compute.world_size,
        )
    else:
        main(0, cfg)  # Single GPU


def main(rank: int, cfg: omegaconf.DictConfig) -> None:
    logger = get_logger(cfg.paths.logs, __name__, rank)
    import sys

    logger.info(f"Running main on exec: {sys.executable}")
    print(f"Running main on exec: {sys.executable}")
    if cfg.experiment.resume_from_checkpoint:
        checkpoint = get_checkpoint(cfg, rank, logger)
        wandb_init_resume = "must"
        run_id = checkpoint.run_id
        cfg = checkpoint.cfg
        cfg.experiment.run_id = run_id
        cfg.experiment.resume_from_checkpoint = True
    else:
        run_id = None
        wandb_init_resume = None
        checkpoint = None
    if rank == 0:
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
        cfg = checkpoint.cfg
        if rank == 0:
            logger.info(f"Resuming training with run_id: {run_id}")

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

    # load_model_kwargs = dict(model=cfg.model.name, datasets=cfg.dataset.name)
    model = ModelFactory.load_model(
        model=cfg.model.name, dataset=cfg.dataset.name, diet=cfg.rigl.diet
    )
    model.to(device)
    if cfg.compute.distributed:
        model = DistributedDataParallel(model, device_ids=[rank])
        # TODO: experiment with this line
        pass
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if model_state is not None:
        model.load_state_dict(model_state)
    optimizer = get_optimizer(cfg, model, state_dict=optimizer_state)
    scheduler = get_lr_scheduler(
        cfg, optimizer, state_dict=scheduler_state, logger=logger
    )
    pruner = None

    if cfg.rigl.dense_allocation is not None:
        if cfg.model.name == "skinny_resnet18":
            dense_allocation = (
                cfg.rigl.dense_allocation * cfg.model.sparsity_scale_factor
            )
            logger.warning(
                f"Scaling {cfg.rigl.dense_allocation} by "
                f"{cfg.model.sparsity_scale_factor:.2f} for SkinnyResNet18 "
                f"New Dense Alloc == {dense_allocation:.6f}"
            )
        else:
            dense_allocation = cfg.rigl.dense_allocation
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
            dense_allocation=dense_allocation,
            alpha=cfg.rigl.alpha,
            delta=cfg.rigl.delta,
            static_topo=cfg.rigl.static_topo,
            T_end=T_end,
            ignore_linear_layers=cfg.rigl.ignore_linear_layers,
            ignore_mha_layers=cfg.rigl.ignore_mha_layers,
            grad_accumulation_n=cfg.rigl.grad_accumulation_n,
            sparsity_distribution=cfg.rigl.sparsity_distribution,
            erk_power_scale=cfg.rigl.erk_power_scale,
            state_dict=pruner_state,
            filter_ablation_threshold=cfg.rigl.filter_ablation_threshold,
            static_ablation=cfg.rigl.static_ablation,
            dynamic_ablation=cfg.rigl.dynamic_ablation,
            min_salient_weights_per_neuron=cfg.rigl.min_salient_weights_per_neuron,  # noqa
            use_sparse_init=cfg.rigl.use_sparse_initialization,
            init_method_str=cfg.rigl.init_method_str,
            use_sparse_const_fan_in_for_ablation=cfg.rigl.use_sparse_const_fan_in_for_ablation,  # noqa
            keep_first_layer_dense=cfg.rigl.keep_first_layer_dense,
            initialize_grown_weights=cfg.rigl.initialize_grown_weights,
            no_ablation_module_names=cfg.model.no_ablation_module_names,
        )
    else:
        logger.warning(
            "cfg.rigl.dense_allocation is `null`, training with dense "
            "network..."
        )

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
    if rank == 0:
        logger.info(f"Model Summary: {model}")
    segmentation_meter = SegmentationMeter()
    if not cfg.experiment.resume_from_checkpoint:
        step = 0
        if rank == 0:
            if run is None:
                run_id = datetime.now().strftime("%h-%m-%d-%H-%M")
            else:
                run_id = run.id
            checkpoint = Checkpoint(
                run_id=run_id,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                pruner=pruner,
                epoch=0,
                step=step,
                parent_dir=cfg.paths.checkpoints,
            )
            if (pruner is not None) and (cfg.wandb.log_filter_stats):
                # Log inital filter stats before pruning
                pruner.log_meters(step=step)

        epoch_start = 1
    else:  # Resuming from checkpoint
        checkpoint.model = model
        checkpoint.optimizer = optimizer
        checkpoint.scheduler = scheduler
        checkpoint.pruner = pruner
        # Start at the next epoch after the last that successfully was saved
        epoch_start = checkpoint.epoch + 1
        step = checkpoint.step
        # NOTE: we will use acc for checkpointing but this will hold mask_mAP
        segmentation_meter._max_mask_mAP = checkpoint.best_acc

    for epoch in range(epoch_start, cfg.training.epochs + 1):
        if pruner is not None and rank == 0:
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
        _, mask_mAP = test(
            cfg,
            model,
            device,
            test_loader,
            epoch,
            step,
            rank,
            logger,
            segmentation_meter,
        )
        if rank == 0:
            wandb.log({"Learning Rate": scheduler.get_last_lr()[0]}, step=step)
            logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]}")
            checkpoint.current_acc = mask_mAP
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
    if rank == 0 and cfg.wandb.log_to_wandb:
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
    steps_to_accumulate_grad = get_steps_to_accumulate_grad(
        cfg.training.simulated_batch_size, cfg.training.batch_size
    )
    for batch_idx, (images, targets) in enumerate(train_loader):
        apply_grads = (
            True
            if steps_to_accumulate_grad == 1
            or (
                batch_idx != 0
                and (batch_idx + 1) % steps_to_accumulate_grad == 0
            )
            else False
        )
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]
        index_to_pop = [
            idx for idx, t in enumerate(targets) if "boxes" not in t
        ]
        if len(index_to_pop) >= 1:
            if rank == 0:
                logger.warning(
                    f"Found {len(index_to_pop)} target(s) missing 'boxes' key "
                    f" in batch_idx == {batch_idx}"
                )
            _images, _targets = [], []
            for idx, (ii, tt) in enumerate(list(zip(images, targets))):
                if idx in index_to_pop:
                    continue
                else:
                    _images.append(ii)
                    _targets.append(tt)
            images = _images
            targets = _targets
        if len(images) == 0:
            continue

        loss_dict = model(images, targets)
        # loss_dict includes losses for classification, bbox regression, masks,
        # objectness, rpn_box regression
        losses = sum(loss for loss in loss_dict.values())

        # Normalize loss for accumulated grad!
        losses = losses / steps_to_accumulate_grad

        # Will call backwards hooks on model and accumulate dense grads if
        # within cfg.rigl.grad_accumulation_n mini-batch steps from update
        losses.backward()

        if apply_grads:  # If we apply grads, check for topology update and log
            if cfg.training.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.training.clip_grad_norm
                )
            step += 1
            optimizer.step()
            if pruner is not None:
                # pruner.__call__ returns False if rigl step taken
                pruner_called = not pruner()

            if step % cfg.training.log_interval == 0 and rank == 0:
                world_size = (
                    1
                    if cfg.compute.distributed is False
                    else cfg.compute.world_size
                )
                logger.info(
                    "Step: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(  # noqa
                        step,
                        epoch,
                        batch_idx * len(images) * world_size,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        losses.item(),
                    )
                )
                wandb_data = {
                    "Training Losses": losses.item(),
                }
                if pruner is not None:
                    wandb_data["ITOP Rate"] = pruner.itop_rs
                    if (
                        cfg.wandb.log_filter_stats
                        and rank == 0
                        and pruner_called
                    ):
                        # If we updated the pruner
                        # log filter-wise statistics to wandb
                        pruner.log_meters(step=step)
                wandb.log(wandb_data, step=step)

            # We zero grads after logging pruner filter meters
            optimizer.zero_grad()
            if cfg.training.dry_run:
                logger.warning("Dry run, exiting after one training step")
                return step
            if (
                cfg.training.max_steps is not None
                and step > cfg.training.max_steps
            ):
                return step
    return step


def test(
    cfg, model, device, test_loader, epoch, step, rank, logger, training_meter
):
    model.eval()
    cpu_device = torch.device("cpu")
    iou_types = ["bbox", "segm"]
    evaluator = CocoEvaluator(test_loader.dataset.coco, iou_types=iou_types)
    with torch.no_grad():
        for images, targets in test_loader:
            images = list(image.to(device) for image in images)
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]
            outputs = model(images)
            outputs = [
                {k: v.to(cpu_device) for k, v in t.items()} for t in outputs
            ]
            res = {
                target["image_id"]: output
                for target, output in zip(targets, outputs)
            }
            evaluator.update(res)
    if cfg.compute.distributed:
        evaluator.synchronize_between_processes()
    evaluator.accumulate()
    if rank == 0:
        logger.info("\nTest set summary:")
        evaluator.summarize()
    # Extract relevant metrics
    if len(evaluator.coco_eval["bbox"].stats) == 0:
        bbox_mAP, mask_mAP = 0, 0
        logger.warn("No stats recovered from this test loop, setting mAPs to 0")
    else:
        bbox_mAP = evaluator.coco_eval["bbox"].stats[0]
        mask_mAP = evaluator.coco_eval["segm"].stats[0]
    training_meter.bbox_mAP = bbox_mAP
    training_meter.mask_mAP = mask_mAP
    if rank == 0:
        wandb_log(
            epoch,
            step,
            images,
            targets,
            res,
            training_meter.bbox_mAP,
            training_meter.max_bbox_mAP,
            training_meter.mask_mAP,
            training_meter.max_mask_mAP,
            cfg.wandb.log_images,
        )
    return bbox_mAP, mask_mAP


def wandb_log(
    epoch,
    step,
    images,
    targets,
    outputs,
    bbox_mAP,
    max_bbox_mAP,
    mask_mAP,
    max_mask_mAP,
    log_images,
):
    log_data = {
        "epoch": epoch,
        "bbox_mAP": bbox_mAP,
        "max_bbox_mAP": max_bbox_mAP,
        "mask_mAP": mask_mAP,
        "max_mask_mAP": max_mask_mAP,
    }
    if log_images:
        annotated_images = []
        for image, output, target in list(
            zip(images, outputs.values(), targets)
        ):
            annotated_images.append(show_gt_vs_dt(image, target, output))
        for idx, ann_image in enumerate(annotated_images):
            log_data.update(
                {f"Annotated Predictions {idx}": wandb.Image(ann_image)}
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


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv_path=".env", override=True)
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
    os.environ["NCCL_DEBUG"] = "INFO"
    print(f"Base Path: {os.environ['BASE_PATH']}")
    initalize_main()
