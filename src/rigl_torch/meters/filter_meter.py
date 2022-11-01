from __future__ import annotations
import torch
from typing import Callable


class FilterMeter(object):
    def __init__(
        self,
        idx: int,
        weight_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ):
        self.idx = idx
        self.weight_history = []
        self.mask_history = []
        self.grad_history = []
        self._weight_tensor = weight_tensor
        self._mask_tensor = mask_tensor
        self.weight_tensor = weight_tensor  # Call setter to populate history
        self.mask_tensor = mask_tensor
        self._ablated = False
        self._grad = None

    def _check_ablated(fn) -> Callable:
        def wrapper(
            self,
            *fn_args,
            **fn_kwargs,
        ) -> Callable:
            if self._ablated:
                return 0
            else:
                return fn(self, *fn_args, **fn_kwargs)

        return wrapper

    def ablate(self) -> None:
        self._ablated = True

    def reactivate(self) -> None:
        self._ablated = False

    @property
    def weight_tensor(self):
        return self._weight_tensor

    @weight_tensor.setter
    def weight_tensor(self, new_weight_tensor: torch.Tensor) -> None:
        self.weight_history.append(self.weight_tensor)
        self._weight_tensor = new_weight_tensor

    @property
    def mask_tensor(self):
        return self._mask_tensor

    @mask_tensor.setter
    def mask_tensor(self, new_mask_tensor: torch.Tensor) -> None:
        self.mask_history.append(self.mask_tensor)
        self._mask_tensor = new_mask_tensor

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, new_grad: torch.Tensor) -> None:
        if self.grad is not None:
            self.grad_history.append(self.grad)
        self._grad = new_grad

    def update_weight_history(self, weights: torch.Tensor) -> None:
        self.weight_history.append(weights)

    def update_mask_history(self, mask: torch.Tensor) -> None:
        self.mask_history.append(mask)

    @_check_ablated
    def mean_weight(self) -> float:
        return torch.mean(self.weight_tensor).item()

    @_check_ablated
    def median_weight(self) -> float:
        return torch.median(self.weight_tensor).item()

    @_check_ablated
    def max_weight(self) -> float:
        return torch.max(self.weight_tensor).item()

    @_check_ablated
    def min_weight(self) -> float:
        return torch.min(self.weight_tensor).item()

    @_check_ablated
    def fan_in(self) -> int:
        return self.mask_tensor.sum().item()

    @_check_ablated
    def sum_weight(self) -> float:
        return self.weight_tensor.sum().item()

    @_check_ablated
    def weight_magnitude(self) -> float:
        return torch.abs(self.weight_tensor).sum().item()


if __name__ == "__main__":

    def test_member_funcs(filter_meter):
        callables = [
            filter_meter.mean_weight,
            filter_meter.median_weight,
            filter_meter.max_weight,
            filter_meter.min_weight,
            filter_meter.fan_in,
        ]
        for c in callables:
            print(c())

    shape = (16, 3, 3, 3)
    weights = torch.rand(size=shape)
    mask = torch.zeros(size=shape, dtype=torch.bool)
    fan_in = 1
    mask[:fan_in] = True
    filter_meter = FilterMeter(0, weight_tensor=weights, mask_tensor=mask)
    print("Before ablation...")
    test_member_funcs(filter_meter)
    print("After ablation...")
    filter_meter.ablate()
    test_member_funcs(filter_meter)
    weights = torch.rand(size=shape)
    mask = torch.zeros(size=shape, dtype=torch.bool)
    filter_meter.weight_tensor = weights
    filter_meter.mask_tensor = mask
    print(len(filter_meter.weight_history))
    print(len(filter_meter.mask_history))
