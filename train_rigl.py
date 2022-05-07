import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import omegaconf
import hydra
import logging
import wandb
import pathlib

from rigl_torch.rigl_scheduler import RigLScheduler
from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
from rigl_torch.models import MnistNet


@hydra.main(config_path="configs/", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    logger.info(f"Running train_rigl.py with config:\n{cfg}")

    run = wandb.init(
        name=cfg.experiment.name,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg=cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method=cfg.wandb.start_method),
    )

    use_cuda = not cfg.compute.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": cfg.training.batch_size}
    test_kwargs = {"batch_size": cfg.training.test_batch_size}
    if use_cuda:
        train_kwargs.update(cfg.compute.cuda_kwargs)
        test_kwargs.update(cfg.compute.cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_path = cfg.paths.data_folder
    dataset1 = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST(data_path, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MnistNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=cfg.training.gamma
    )

    pruner = lambda: True  # noqa: E731
    if cfg.rigl.dense_allocation is not None:
        T_end = int(0.75 * cfg.training.epochs * len(train_loader))
        pruner = RigLConstFanScheduler(
            model,
            optimizer,
            dense_allocation=cfg.rigl.dense_allocation,
            alpha=cfg.rigl.alpha,
            delta=cfg.rigl.delta,
            static_topo=cfg.rigl.static_topo,
            T_end=T_end,
            ignore_linear_layers=False,
            grad_accumulation_n=cfg.rigl.grad_accumulation_n,
        )
    else:
        logger.warning(
            f"{cfg.rigl.dense_allocation} is None, training with dense network..."
        )

    writer = SummaryWriter(log_dir="./graphs")

    wandb.watch(model, criterion=F.nll_loss, log="all", log_freq=100)
    logger.info(f"Model Summary: {model}")
    for epoch in range(1, cfg.training.epochs + 1):
        logger.info(pruner)
        train(
            cfg, model, device, train_loader, optimizer, epoch, pruner=pruner,
        )
        loss, acc = test(model, device, test_loader, epoch)
        scheduler.step()

        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("accuracy", acc, epoch)
        if cfg.training.dry_run:
            break

    if cfg.training.save_model:
        save_path = pathlib.Path(cfg.paths.artifacts)
    if not save_path.is_dir():
        save_path.mkdir()
    torch.save(model.state_dict(), save_path / "mnist_cnn.pt")
    run.finish()


def train(cfg, model, device, train_loader, optimizer, epoch, pruner):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if pruner():
            optimizer.step()

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
            return


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    wandb_log(
        epoch, test_loss, correct / len(test_loader), data, output, target, pred
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


def wandb_log(epoch, loss, accuracy, inputs, logits, captions, pred):
    wandb.log(
        {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "inputs": wandb.Image(inputs),
            "logits": wandb.Histogram(logits),
            "captions": wandb.Html(captions.numpy().__str__()),
            "predictions": wandb.Html(pred.numpy().__str__()),
        }
    )


if __name__ == "__main__":
    logger = logging.getLogger(__file__)
    main()
