import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import optim

from rigl_torch.rigl_scheduler import RigLScheduler
from rigl_torch.utils.rigl_utils import get_W
from utils.mocks import MNISTNet, mock_image_dataloader
from rigl_torch.utils.checkpoint import Checkpoint
from .test_utils import get_test_cfg
from rigl_torch.optim import get_lr_scheduler, get_optimizer
import pytest
import dotenv
from rigl_torch.models import ModelFactory
import pathlib

# Load custom .env
_CUDA_RANK = 0
_CUDA_RANKS = [0]
dotenv.load_dotenv(dotenv_path=f"{os.getcwd()}/.env", override=True)
# set up environment
# torch.manual_seed(1)
device = (
    torch.device(f"cuda:{_CUDA_RANK}")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# hyperparameters
arch = "resnet50"
image_dimensionality = (3, 224, 224)
num_classes = 1000
max_iters = 15
T_end = int(max_iters * 0.75)
delta = 3
dense_allocation = 0.1
criterion = torch.nn.functional.cross_entropy


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


@pytest.fixture(
    scope="function", params=["cuda", "cpu"], ids=[f"cuda:{_CUDA_RANK}", "cpu"]
)
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
    _pruner = RigLScheduler(
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


def test_lengths_of_W():
    resnet18 = torch.hub.load("pytorch/vision:v0.6.0", "resnet18", weights=None)
    resnet18_W, *_ = get_W(resnet18)
    assert len(resnet18_W) == 21, 'resnet18 should have 21 "weight" matrices'

    resnet50 = torch.hub.load("pytorch/vision:v0.6.0", "resnet50", weights=None)
    resnet50_W, *_ = get_W(resnet50)
    assert len(resnet50_W) == 54, 'resnet50 should have 54 "weight" matrices'


def get_dummy_dataloader():
    X = torch.rand((max_iters, *image_dimensionality))
    T = (torch.rand(max_iters) * num_classes).long()
    dataset = torch.utils.data.TensorDataset(X, T)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    return dataloader


def get_new_scheduler(
    static_topo=False, use_ddp=False, state_dict=None, model=None
):
    if model is None:
        model = torch.hub.load("pytorch/vision:v0.6.0", arch, weights=None).to(
            device
        )

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=_CUDA_RANKS
        )
    # # TODO: Decomission the data parallel code
    # elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=_CUDA_RANKS)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)
    scheduler = RigLScheduler(
        model,
        optimizer,
        dense_allocation=dense_allocation,
        T_end=T_end,
        delta=delta,
        static_topo=static_topo,
        state_dict=state_dict,
    )
    print(model)
    print(scheduler)
    return scheduler


def assert_actual_sparsity_is_valid(scheduler, verbose=False):
    for l, (W, target_S, N, mask) in enumerate(
        zip(scheduler.W, scheduler.S, scheduler.N, scheduler.backward_masks)
    ):
        target_zeros = int(target_S * N)
        actual_zeros = torch.sum(W == 0).item()
        sum_of_zeros = torch.sum(W[mask == 0]).item()
        if verbose:
            print("----- layer %i ------" % l)
            print("target_zeros", target_zeros)
            print("actual_zeros", actual_zeros)
            print("mask_sum", torch.sum(mask).item())
            print("mask_shape", mask.shape)
            print("w_shape", W.shape)
            print("sum_of_nonzeros", torch.sum(W[mask]).item())
            print("sum_of_zeros", sum_of_zeros)
            print(
                "num_of_zeros that are NOT actually zeros",
                torch.sum(W[mask == 0] != 0).item(),
            )
            print("avg_of_zeros", torch.mean(W[mask == 0]).item())
        assert sum_of_zeros == 0


def assert_sparse_elements_remain_zeros(
    static_topo, use_ddp=False, verbose=False, scheduler=None
):
    if scheduler is None:
        scheduler = get_new_scheduler(static_topo, use_ddp=use_ddp)

    model = scheduler.model
    optimizer = scheduler.optimizer

    dataloader = get_dummy_dataloader()
    model.train()
    for i, (X, T) in enumerate(dataloader):
        Y = model(X.to(device))
        loss = criterion(Y, T.to(device))
        loss.backward()

        is_rigl_step = True
        if scheduler():
            is_rigl_step = False
            optimizer.step()

        if verbose:
            print(
                "iteration: %i\trigl steps completed: %i\tis_rigl_step=%s"
                % (i, scheduler.rigl_steps, str(is_rigl_step))
            )
        assert_actual_sparsity_is_valid(scheduler, verbose=verbose)


def assert_sparse_momentum_remain_zeros(static_topo, use_ddp=False):
    scheduler = get_new_scheduler(static_topo, use_ddp=use_ddp)
    model = scheduler.model
    optimizer = scheduler.optimizer
    dataloader = get_dummy_dataloader()

    model.train()
    for i, (X, T) in enumerate(dataloader):
        optimizer.zero_grad()
        Y = model(X.to(device))
        loss = criterion(Y, T.to(device))
        loss.backward()

        is_rigl_step = True
        if scheduler():
            is_rigl_step = False
            optimizer.step()

        print(
            "iteration: %i\trigl steps completed: %i\tis_rigl_step=%s"
            % (i, scheduler.rigl_steps, str(is_rigl_step))
        )

        # check momentum
        for l, (w, mask) in enumerate(
            zip(scheduler.W, scheduler.backward_masks)
        ):
            param_state = optimizer.state[w]
            assert "momentum_buffer" in param_state
            buf = param_state["momentum_buffer"]
            sum_zeros = torch.sum(buf[mask == 0]).item()
            print("layer %i" % l)
            assert sum_zeros == 0


def assert_sparse_gradients_remain_zeros(static_topo, use_ddp=False):
    scheduler = get_new_scheduler(static_topo, use_ddp=use_ddp)
    model = scheduler.model
    optimizer = scheduler.optimizer
    dataloader = get_dummy_dataloader()
    device = next(model.parameters()).device

    model.train()
    for i, (X, T) in enumerate(dataloader):
        optimizer.zero_grad()
        Y = model(X.to(device))
        loss = criterion(Y, T.to(device))
        loss.backward()

        is_rigl_step = True
        if scheduler():
            is_rigl_step = False
            optimizer.step()

        print(
            "iteration: %i\trigl steps completed: %i\tis_rigl_step=%s"
            % (i, scheduler.rigl_steps, str(is_rigl_step))
        )

        # check gradients
        for l, (w, mask) in enumerate(
            zip(scheduler.W, scheduler.backward_masks)
        ):
            grads = w.grad
            sum_zeros = torch.sum(grads[mask == 0]).item()
            print("layer %i" % l)
            assert sum_zeros == 0


checkpoint_fn = "test_checkpoint"
_CKPT_MODEL = None


class TestRigLScheduler:
    def test_initial_sparsity(self):
        scheduler = get_new_scheduler()
        assert_actual_sparsity_is_valid(scheduler)

    def test_checkpoint_saving(self):
        cfg_args = [
            "paths.checkpoints=test_ckp",
        ]
        cfg = get_test_cfg(cfg_args)
        model = ModelFactory.load_model(
            model=cfg.model.name,
            dataset=cfg.dataset.name,
        )
        model.to(device)
        pruner = get_new_scheduler(model=model)
        optimizer = get_optimizer(cfg, pruner.model, state_dict=None)
        scheduler = get_lr_scheduler(cfg, optimizer, state_dict=None)

        ckpt = Checkpoint(
            run_id="test_ckpt_id",
            cfg=cfg,
            model=pruner.model,
            optimizer=optimizer,
            scheduler=scheduler,
            pruner=pruner,
            checkpoint_dir="test_ckpt",
        )
        ckpt.save_checkpoint()
        global _CKPT_MODEL
        _CKPT_MODEL = model
        assert pathlib.Path("test_ckpt/checkpoint.pt.tar").exists()

    def test_checkpoint_loading(self):
        global _CKPT_MODEL
        cfg_args = [
            "paths.checkpoints=test_ckp",
        ]
        cfg = get_test_cfg(cfg_args)

        ckpt = Checkpoint.load_last_checkpoint(
            checkpoint_dir="test_ckpt", rank=_CUDA_RANK
        )
        pruner_state = ckpt.pruner
        model_state = ckpt.model
        model = ModelFactory.load_model(
            model=cfg.model.name,
            dataset=cfg.dataset.name,
        ).to(device)

        pruner = get_new_scheduler(state_dict=pruner_state, model=model)
        model = pruner.model
        model.load_state_dict(model_state)
        # os.remove(checkpoint_fn)

        # first make sure the original model is the same as the loaded one
        original_W, *_ = get_W(_CKPT_MODEL)
        assert len(original_W) == len(pruner.W)
        for oW, nW in zip(original_W, pruner.W):
            assert torch.equal(oW, nW)
        import shutil

        shutil.rmtree("test_ckpt")

        # assert_sparse_elements_remain_zeros(False, scheduler=scheduler)

    def test_sparse_momentum_remain_zeros_STATIC_TOPO(self):
        assert_sparse_momentum_remain_zeros(True)

    def test_sparse_momentum_remain_zeros_RIGL_TOPO(self):
        assert_sparse_momentum_remain_zeros(False)

    def test_sparse_elements_remain_zeros_STATIC_TOPO(self):
        assert_sparse_elements_remain_zeros(True)

    def test_sparse_elements_remain_zeros_RIGL_TOPO(self):
        assert_sparse_elements_remain_zeros(False)

    def test_sparse_gradients_remain_zeros_STATIC_TOPO(self):
        assert_sparse_gradients_remain_zeros(True)

    def test_sparse_gradients_remain_zeros_RIGL_TOPO(self):
        assert_sparse_gradients_remain_zeros(False)

    def test_rigl_step(self):
        assert 1 == 1


# distributed testing setup
BACKEND = "nccl"  # mpi, gloo, or nccl
WORLD_SIZE = 2
# init_method = "file://%s/distributed_test" % os.getcwd()
init_method = None


def assert_actual_sparsity_is_valid_DISTRIBUTED(rank, static_topo=False):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        BACKEND, init_method=init_method, rank=rank, world_size=WORLD_SIZE
    )
    scheduler = get_new_scheduler(static_topo=static_topo, use_ddp=True)
    assert_actual_sparsity_is_valid(scheduler)


def assert_sparse_momentum_remain_zeros_DISTRIBUTED(rank, static_topo):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        BACKEND, init_method=init_method, rank=rank, world_size=WORLD_SIZE
    )
    assert_sparse_momentum_remain_zeros(static_topo, use_ddp=True)


def assert_sparse_elements_remain_zeros_DISTRIBUTED(rank, static_topo):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        BACKEND, init_method=init_method, rank=rank, world_size=WORLD_SIZE
    )
    assert_sparse_elements_remain_zeros(static_topo, use_ddp=True)


def assert_sparse_gradients_remain_zeros_DISTRIBUTED(rank, static_topo):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        BACKEND, init_method=init_method, rank=rank, world_size=WORLD_SIZE
    )
    assert_sparse_gradients_remain_zeros(static_topo, use_ddp=True)


class TestRigLSchedulerDistributed:
    @pytest.mark.slow
    def test_initial_sparsity(self):
        mp.spawn(assert_actual_sparsity_is_valid_DISTRIBUTED, nprocs=WORLD_SIZE)

    @pytest.mark.slow
    def test_sparse_momentum_remain_zeros_STATIC_TOPO(self):
        mp.spawn(
            assert_sparse_momentum_remain_zeros_DISTRIBUTED,
            nprocs=WORLD_SIZE,
            args=(True,),
        )

    @pytest.mark.slow
    def test_sparse_momentum_remain_zeros_RIGL_TOPO(self):
        mp.spawn(
            assert_sparse_momentum_remain_zeros_DISTRIBUTED,
            nprocs=WORLD_SIZE,
            args=(False,),
        )

    @pytest.mark.slow
    def test_sparse_elements_remain_zeros_STATIC_TOPO(self):
        mp.spawn(
            assert_sparse_elements_remain_zeros_DISTRIBUTED,
            nprocs=WORLD_SIZE,
            args=(True,),
        )

    @pytest.mark.slow
    def test_sparse_elements_remain_zeros_RIGL_TOPO(self):
        mp.spawn(
            assert_sparse_elements_remain_zeros_DISTRIBUTED,
            nprocs=WORLD_SIZE,
            args=(False,),
        )

    @pytest.mark.slow
    def test_sparse_gradients_remain_zeros_STATIC_TOPO(self):
        mp.spawn(
            assert_sparse_gradients_remain_zeros_DISTRIBUTED,
            nprocs=WORLD_SIZE,
            args=(True,),
        )

    @pytest.mark.slow
    def test_sparse_gradients_remain_zeros_RIGL_TOPO(self):
        mp.spawn(
            assert_sparse_gradients_remain_zeros_DISTRIBUTED,
            nprocs=WORLD_SIZE,
            args=(False,),
        )
