import pytest
from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
import torch
import torch.optim as optim
from rigl_torch.utils import rigl_utils
from utils.mocks import (
    mock_image_dataloader,
    MNISTNet,
    train_model,
    evaluate_model,
)


@pytest.fixture(scope="function")
def net():
    _net = MNISTNet()
    yield _net
    del _net


@pytest.fixture(scope="module", params=[1, 64], ids=["batch_1", "batch_64"])
def data_loaders(request):
    image_dimensionality = (1, 28, 28)
    num_classes = 10
    dataloader = mock_image_dataloader(
        image_dimensionality, num_classes, request.param
    )
    yield dataloader, dataloader
    del dataloader


@pytest.fixture(scope="function", params=["cuda", "cpu"], ids=["cuda", "cpu"])
def pruner(request, net, data_loaders):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda not available!")
    train_loader, _ = data_loaders
    torch.manual_seed(42)
    device = torch.device(request.param)
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
        assert len(rigl_utils.get_fan_in_tensor(w).unique()) == 1


@pytest.mark.slow
def test_train_const_fan(pruner, data_loaders):
    train_loader, test_loader = data_loaders
    log_interval = 100
    device = pruner.model.device
    epochs = 3
    losses = []
    accuracy = []
    model = pruner.model
    optimizer = pruner.optimizer

    for epoch in range(1, epochs + 1):
        print(pruner)
        train_model(
            log_interval,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            pruner=pruner,
        )
        loss, acc = evaluate_model(model, device, test_loader)

        losses.append(loss)
        accuracy.append(acc)
        for w in pruner.W:
            if w is None:
                continue
        assert len(rigl_utils.get_fan_in_tensor(w).unique()) == 1


def test_str(pruner):
    pruner_str = pruner.__str__()
    str_components = [
        "ITOP rate",
        "Active Neuron Count",
    ]
    for comp in str_components:
        assert comp in pruner_str
