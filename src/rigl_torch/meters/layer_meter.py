from __future__ import annotations
import torch
import wandb
from typing import List

from rigl_torch.utils.rigl_utils import calculate_fan_in_and_fan_out


class LayerMeter(object):
    def __init__(
        self, idx: int, weight: torch.Tensor, mask: torch.Tensor, name: str = ""
    ):
        self.idx = idx
        self.weight = weight
        self.mask = mask
        self.name = name
        self._log_neuron_stats = False
        self._append_name = True
        if self._append_name:
            self.name = f"z-{self.name}"  # Place at the end of panels

    def log_to_wandb(
        self,
        dense_grads: List[float],
        max_inactive_weights: List[float],
        step: int,
    ):
        # Populate layerwise data
        active_neuron_count = self.active_neuron_count()
        fan_ins = self.fan_ins()
        mean_abs_weights = self.mean_abs_weights()
        max_abs_weights = self.max_abs_weights()
        min_abs_weights = self.min_abs_weights()
        weights_std = self.weights_std()
        with torch.no_grad():
            mean_abs_grad = [
                torch.mean(torch.abs(grad)).item() for grad in dense_grads
            ]
            max_abs_grad = [
                torch.max(torch.abs(grad)).item() for grad in dense_grads
            ]
            min_abs_grad = [
                torch.min(torch.abs(grad)).item() for grad in dense_grads
            ]
            grads_std = [
                torch.std(torch.abs(grad)).item() for grad in dense_grads
            ]
        max_inactive_grad = self.max_inactive_grad(dense_grads)
        log_data = {}

        # Per neuron data
        if self._log_neuron_stats:
            for idx in range(len(self.mask)):
                filter_name = f"{self.name}_neuron-{idx}"
                log_data.update(
                    {
                        f"{filter_name}-fan_in": fan_ins[idx],
                        f"{filter_name}-mean_abs_weight": mean_abs_weights[idx],
                        f"{filter_name}-max_abs_weights": max_abs_weights[idx],
                        f"{filter_name}-min_abs_weights": min_abs_weights[idx],
                        f"{filter_name}-weights_std": weights_std[idx],
                        f"{filter_name}-mean_abs_grad": mean_abs_grad[idx],
                        f"{filter_name}-max_abs_grad": max_abs_grad[idx],
                        f"{filter_name}-min_abs_grad": min_abs_grad[idx],
                        f"{filter_name}-grad_std": grads_std[idx],
                    }
                )

        # Layerwise statstics
        log_data.update(
            {
                f"{self.name}_total-active-neurons": active_neuron_count,
                f"{self.name}-Fan-Ins": wandb.Histogram(fan_ins),
                f"{self.name}-Weight-Dist": wandb.Histogram(
                    self.weight.flatten().detach().cpu()
                ),
                f"{self.name}-Max-Weight-Per-Filter": wandb.Histogram(
                    max_abs_weights
                ),
                f"{self.name}-Max-Grad-Per-Filter": wandb.Histogram(
                    max_abs_grad
                ),
                f"{self.name}-Grad-Dist": wandb.Histogram(
                    dense_grads.flatten().detach().cpu()
                ),
                f"{self.name}-Max-Inactive_Weight": max_inactive_weights[
                    self.idx
                ],
                f"{self.name}-Max-Inactive_Grad": max_inactive_grad,
            }
        )
        wandb.log(log_data, step=step)

    @torch.no_grad()
    def active_neurons(self) -> List[int]:
        active_neurons = []
        if self.mask is not None:
            for idx, neuron in enumerate(self.mask):
                if neuron.any():
                    active_neurons.append(idx)
        else:
            active_neurons = [i for i in range(len(self.weight))]
        return active_neurons

    @torch.no_grad()
    def active_neuron_count(self) -> int:
        return len(self.active_neurons())

    @torch.no_grad()
    def fan_ins(self) -> List[int]:
        if self.mask is not None:
            fan_ins = [
                torch.sum(self.mask[idx]).item()
                for idx in range(len(self.mask))
            ]
        else:
            fan_in, _ = calculate_fan_in_and_fan_out(self.weight)
            fan_ins = [fan_in for _ in range(len(self.weight))]
        return fan_ins

    @torch.no_grad()
    def mean_abs_weights(self) -> List[float]:
        return [torch.mean(torch.abs(w)).item() for w in self.weight]

    @torch.no_grad()
    def max_abs_weights(self) -> List[float]:
        return [torch.max(torch.abs(w)).item() for w in self.weight]

    @torch.no_grad()
    def max_inactive_grad(self, dense_grads) -> float:
        max_grad = 0.0
        if self.mask is None:
            return 0.0
        for m, g in list(zip(self.mask, dense_grads)):
            if m is None:
                continue
            elif not m.any():  # neuron inactive
                g_max = torch.abs(g).max().item()
                if g_max > max_grad:
                    max_grad = g_max
        return max_grad

    @torch.no_grad()
    def min_abs_weights(self) -> List[float]:
        return [torch.min(torch.abs(w)).item() for w in self.weight]

    @torch.no_grad()
    def weights_std(self) -> List[float]:
        return [torch.std(torch.abs(w)).item() for w in self.weight]
