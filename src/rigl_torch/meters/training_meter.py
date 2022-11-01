from rigl_torch import RiglScheduler
from layer_meter import LayerMeter
from typing import List
import torch


class TrainingMeter(object):
    def __init__(self, pruner: RiglScheduler):
        self.pruner = pruner
        self._layer_meters = self._initalize_meters()

    def update_grads(self, grads: torch.Tensor) -> None:
        for layer_grad, lm in list(zip(grads, self._layer_meters)):
            lm.grad = layer_grad
        return

    def update_pruner(self, updated_pruner: RiglScheduler):
        self.pruner = updated_pruner
        for layer_idx, (layer, mask) in enumerate(
            list(zip(self.pruner.W, self.pruner.backward_masks))
        ):
            assert layer.idx == layer_idx
            layer.update(weight_tensor=layer, mask_tensor=mask)

    def _initalize_meters(self) -> List[LayerMeter]:
        layer_meters = []
        for layer_idx, (layer, mask) in enumerate(
            list(zip(self.pruner.W, self.pruner.backward_masks))
        ):
            layer_meters.append(
                LayerMeter(idx=layer_idx, weight_tensor=layer, mask_tensor=mask)
            )
        return layer_meters
