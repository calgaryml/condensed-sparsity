import pytest
from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
import torch
import torch.nn.functional as F
import torch.optim as optim
from rigl_torch import util
from tests.utils.mocks import mock_image_dataloader, MNISTNet


__BATCH_SIZE = 64
__USE_CUDA = torch.cuda.is_available()


@pytest.fixture(scope="function")
def net():
    _net = MNISTNet()
    yield _net
    del _net


@pytest.fixture(scope="module")
def data_loaders():
    image_dimensionality = (1, 28, 28)
    num_classes = 10
    dataloader = mock_image_dataloader(
        image_dimensionality, num_classes, __BATCH_SIZE
    )
    yield dataloader, dataloader
    del dataloader


@pytest.fixture(scope="function")
def pruner(net, data_loaders):
    # TODO: Can load these from a config file
    train_loader, test_loader = data_loaders
    torch.manual_seed(42)
    device = torch.device("cuda" if __USE_CUDA else "cpu")
    lr = 0.001
    alpha = 0.3
    static_topo = 0
    dense_allocation = 0.1
    grad_accumulation_n = 1
    delta = 100
    epochs = 3
    T_end = int(
        0.75 * epochs * len(train_loader)
    )  # Stop rigl after this many steps
    model = net.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    _pruner = RigLConstFanScheduler(
        model,
        optimizer,
        dense_allocation=dense_allocation,
        alpha=alpha,
        delta=delta,
        static_topo=static_topo,
        T_end=T_end,
        ignore_linear_layers=False,
        grad_accumulation_n=grad_accumulation_n,
    )
    yield _pruner
    del _pruner


def test_random_sparsify(pruner):
    for w in pruner.W:
        if w is None:
            continue
        assert len(util.get_fan_in_tensor(w).unique()) == 1


@pytest.mark.slow
def test_train_const_fan(pruner, data_loaders):
    train_loader, test_loader = data_loaders
    log_interval = 100
    device = torch.device("cuda" if __USE_CUDA else "cpu")
    epochs = 3
    losses = []
    accuracy = []
    model = pruner.model
    optimizer = pruner.optimizer

    def train(
        log_interval, model, device, train_loader, optimizer, epoch, pruner
    ):
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

    def test(model, device, test_loader):
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

    for epoch in range(1, epochs + 1):
        print(pruner)
        train(
            log_interval,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            pruner=pruner,
        )
        loss, acc = test(model, device, test_loader)

        losses.append(loss)
        accuracy.append(acc)
        for w in pruner.W:
            if w is None:
                continue
        assert len(util.get_fan_in_tensor(w).unique()) == 1
