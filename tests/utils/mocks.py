import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from rigl_torch.rigl_scheduler import RigLScheduler


def mock_image_dataloader(
    image_dimensionality: Tuple[int],
    num_classes: int,
    batch_size: int,
    max_iters: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Initalizes a synthetic dataloader consisting of noise and labels.

    Index access operator of returned data load will return tuples consisting of
    2D images of noise of shape `(batch_size, *image_dimensionality)` and random
    labels of shape `(batch_size, 1)` with num_classes unique labels.

    Args:
        image_dimensionality (Tuple[int]): Shape of images to generate.
            Eg., (224, 224)
        num_classes (int): Number of synethic labels to use, Eg., a value of 2
            will yield binary labels.
        batch_size (int): Batch size to generate image, label tensor tuples in.
        max_iters (Optional[int], optional): Length of dataloader to create.
            Defaults to None. If None, len(dataloader) will equal batch_size.

    Returns:
        torch.utils.data.DataLoader: _description_
    """
    if max_iters is None:
        max_iters = batch_size
    X = torch.rand((max_iters, *image_dimensionality))
    T = (torch.rand(max_iters) * num_classes).long()
    dataset = torch.utils.data.TensorDataset(X, T)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_model(
    log_interval: int,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    pruner: RigLScheduler,
) -> None:
    """Trains model with RigL pruner.

    Args:
        log_interval (int): Frequency of log messages per step.
        model (torch.nn.Module): Model to train.
        device (torch.device): Device to train model on.
        train_loader (torch.utils.data.DataLoader): Train data loader.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        epoch (int): Number of epochs to train model for.
        pruner (RigLScheduler): RigL pruner to use.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if pruner():
            optimizer.step()
        else:
            print(f"rigl step at batch_idx {batch_idx}")

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def evaluate_model(
    model: torch.nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
) -> Tuple[float]:
    """Tests model using test_loader.

    Args:
        model (torch.nn.Module): Trained model to test.
        device (torch.device): Device where model parameters have been loaded.
        test_loader (torch.utils.data.DataLoader): Test data to use to evaluate
            model.

    Returns:
        Tuple[float]: Negative log-likelihood loss and accuracy.
    """
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

    print(
        "\nTest set: Average loss: {:.4f}, "
        "Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, correct / len(test_loader.dataset)
