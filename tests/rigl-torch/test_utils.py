import pytest
from rigl_torch import util
import torch
from utils.mocks import MNISTNet


@pytest.fixture(scope="module")
def net():
    net = MNISTNet()
    yield net
    del net


def test_calculate_fan_in_and_fan_out_conv(net):
    fan_in, fan_out = util.calculate_fan_in_and_fan_out(net.conv1)
    assert fan_in == 1 * 3 * 3  # 1 in_channel * (3 *3) kernel
    assert fan_out == 32 * 3 * 3  # 32 out_channel * (3 *3) kernel


def test_calculate_fan_in_and_fan_out_parameters_arg():
    params = torch.nn.Conv2d(3, 32, 3)._parameters
    fan_in, fan_out = util.calculate_fan_in_and_fan_out(params["weight"])
    assert fan_in == 3 * 3 * 3  # 3 in_channel * (3 *3) kernel
    assert fan_out == 32 * 3 * 3  # 32 out_channel * (3 *3) kernel


def test_calculate_fan_in_and_fan_out_linear(net):
    fan_in, fan_out = util.calculate_fan_in_and_fan_out(
        net.fc1
    )  # in 9216, out 128
    assert fan_in == 9216
    assert fan_out == 128


def test_calculate_fan_in_and_fan_out_sparse():
    sparsity = 0.33
    ones = torch.ones(size=(10, 3, 3, 3))
    ones[:, 0] = 0  # Structured sparsity
    assert util.get_fan_in_tensor(ones).unique().item() == round(
        3 * 3 * 3 * (1 - sparsity)
    )
    ones = torch.ones(size=(10, 3, 3, 3))
    for idx, filter in enumerate(ones):  # Unstructured sparsity
        filter = filter.flatten()
        idx_to_drop = torch.randperm(filter.shape[0])
        idx_to_drop = idx_to_drop[: round(filter.shape[0] * sparsity)]
        filter[idx_to_drop] = 0
        ones[idx] = filter.reshape(3, 3, 3)
    assert util.get_fan_in_tensor(ones).unique().item() == round(
        3 * 3 * 3 * (1 - sparsity)
    )


def test_get_fan_in_tensor_conv(net):
    fan_in_tensor = util.get_fan_in_tensor(net.conv1.weight)
    assert len(fan_in_tensor) == net.conv1.out_channels
    assert len(torch.unique(fan_in_tensor)) == 1
    assert (
        torch.unique(fan_in_tensor).item() == 1 * 3 * 3
    )  # 1 in_channel * (3 *3) kernel


def test_get_fan_in_tensor_linear(net):
    fan_in_tensor = util.get_fan_in_tensor(net.fc1.weight)
    assert len(fan_in_tensor) == net.fc1.out_features
    assert len(torch.unique(fan_in_tensor)) == 1
    assert torch.unique(fan_in_tensor).item() == 9216


def test_validate_constant_fan_in(net):
    fan_in_tensor = util.get_fan_in_tensor(net.fc1.weight)
    assert util.validate_constant_fan_in(fan_in_tensor)
    fan_in_tensor = util.get_fan_in_tensor(net.conv1.weight)
    assert util.validate_constant_fan_in(fan_in_tensor)
