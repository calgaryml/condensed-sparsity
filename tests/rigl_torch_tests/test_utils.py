import pytest
from rigl_torch.utils import rigl_utils
import torch
from utils.mocks import MNISTNet
from hydra import compose, initialize
import os
import dotenv


@pytest.fixture(scope="module")
def net():
    net = MNISTNet()
    yield net
    del net


@pytest.mark.parametrize(
    "module_param, expected_fan_in, expected_fan_out",
    [
        (torch.nn.Conv2d(3, 32, 3), 3 * 3 * 3, 32 * 3 * 3),
        (
            torch.nn.Conv2d(3, 32, 3)._parameters["weight"],
            3 * 3 * 3,
            32 * 3 * 3,
        ),
        (torch.nn.Linear(in_features=9216, out_features=128), 9216, 128),
        (
            torch.nn.Linear(in_features=9216, out_features=128)._parameters[
                "weight"
            ],
            9216,
            128,
        ),
    ],
    ids=[
        "Conv2D-Module",
        "Conv2D-Parameters",
        "Linear-Module",
        "Linear-Parameters",
    ],
)
def test_calculate_fan_in_and_fan_out_parameters_arg(
    module_param, expected_fan_in, expected_fan_out
):
    fan_in, fan_out = rigl_utils.calculate_fan_in_and_fan_out(module_param)
    assert fan_in == expected_fan_in
    assert fan_out == expected_fan_out


@pytest.mark.parametrize(
    "sparse_tensor, const_fan_in",
    [
        (
            torch.tensor(
                [
                    [
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    ],
                    [
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    ],
                ]
            ),
            9,
        ),
        (
            torch.tensor(
                [
                    [[[1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0]]],
                    [[[1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0]]],
                ]
            ),
            6,
        ),
        (
            torch.tensor(
                [
                    [1, 1, 1, 0, 0, 0],
                ]
            ),
            3,
        ),
    ],
    ids=[
        "structured_sparse_tensor",
        "unstructured_sparse_tensor",
        "dim2_sparse_tensor",
    ],
)
def test_calculate_fan_in_and_fan_out_sparse(sparse_tensor, const_fan_in):
    assert (
        rigl_utils.get_fan_in_tensor(sparse_tensor).unique().item()
        == const_fan_in
    )


def test_calculate_fan_in_and_fan_out_raises_on_dim():
    t = torch.tensor([0, 1, 1])
    with pytest.raises(ValueError) as _:
        rigl_utils.calculate_fan_in_and_fan_out(t)


def test_get_fan_in_tensor_conv(net):
    fan_in_tensor = rigl_utils.get_fan_in_tensor(net.conv1.weight)
    assert len(fan_in_tensor) == net.conv1.out_channels
    assert len(torch.unique(fan_in_tensor)) == 1
    assert (
        torch.unique(fan_in_tensor).item() == 1 * 3 * 3
    )  # 1 in_channel * (3 *3) kernel


def test_get_fan_in_tensor_linear(net):
    fan_in_tensor = rigl_utils.get_fan_in_tensor(net.fc1.weight)
    assert len(fan_in_tensor) == net.fc1.out_features
    assert len(torch.unique(fan_in_tensor)) == 1
    assert torch.unique(fan_in_tensor).item() == 9216


def test_validate_constant_fan_in(net):
    fan_in_tensor = rigl_utils.get_fan_in_tensor(net.fc1.weight)
    assert rigl_utils.validate_constant_fan_in(fan_in_tensor)
    fan_in_tensor = rigl_utils.get_fan_in_tensor(net.conv1.weight)
    assert rigl_utils.validate_constant_fan_in(fan_in_tensor)


def get_test_cfg(cfg_args):
    overrides = [
        "dataset=cifar10",
        "compute.distributed=False",
        "model=resnet18",
        # # "model=skinny_resnet18",
        # "rigl.dense_allocation=0.01",
        # "rigl.delta=2",
        # "rigl.grad_accumulation_n=1"
    ]
    overrides.extend(cfg_args)
    with initialize("../../configs", version_base="1.2.0"):
        cfg = compose(
            "config.yaml",
            overrides=overrides,
        )
        dotenv.load_dotenv("../.env")
        os.environ["IMAGE_NET_PATH"]
    return cfg
