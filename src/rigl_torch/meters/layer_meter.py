# from __future__ import annotations
# import torch
# from typing import Callable


# class LayerMeter(object):
#     def __init__(
#         self,
#         idx: int,
#         weight_tensor: torch.Tensor,
#         mask_tensor: torch.Tensor,
#     ):
#         self.idx = idx
#         self.weight_history = []
#         self.mask_history = []
#         self.grad_history =
#         self.filter_meters = self._initalize_filter_meters(
#             weight_tensor, mask_tensor
#         )
#         self.ablated_filters = []
#         self.active_filters = [idx]
#         self._weight_tensor = weight_tensor
#         self._mask_tensor = mask_tensor
#         self.weight_tensor = weight_tensor  # Call setter to populate history
#         self.mask_tensor = mask_tensor

#     def _get_active_filter_statistic(fn) -> Callable:
#         def wrapper(
#             self,
#             tensor: torch.Tensor,
#             *fn_args,
#             **fn_kwargs,
#         ) -> Callable:
#             active_elements = tensor.index_select(dim=0, index=)


#             if self._ablated:
#                 return 0
#             else:
#                 return fn(self, *fn_args, **fn_kwargs)

#         return wrapper

#     def ablate(self) -> None:
#         self._ablated = True

#     def reactivate(self) -> None:
#         self._ablated = False

#     @property
#     def weight_tensor(self):
#         return self._weight_tensor

#     @weight_tensor.setter
#     def weight_tensor(self, new_weight_tensor: torch.Tensor) -> None:
#         self.weight_history.append(self.weight_tensor)
#         self._weight_tensor = new_weight_tensor

#     @property
#     def mask_tensor(self):
#         return self._mask_tensor

#     @mask_tensor.setter
#     def mask_tensor(self, new_mask_tensor: torch.Tensor) -> None:
#         self.mask_history.append(self.mask_tensor)
#         self._mask_tensor = new_mask_tensor

#     @property
#     def mask_tensor(self):
#         return self._mask_tensor

#     @mask_tensor.setter
#     def mask_tensor(self, new_mask_tensor: torch.Tensor) -> None:
#         self.mask_history.append(self.mask_tensor)
#         self._mask_tensor = new_mask_tensor

#     def update_weight_history(self, weights: torch.Tensor) -> None:
#         self.weight_history.append(weights)

#     def update_mask_history(self, mask: torch.Tensor) -> None:
#         self.mask_history.append(mask)

#     @_check_ablated
#     def mean_weight(self) -> float:
#         return torch.mean(self.weight_tensor).item()

#     @_check_ablated
#     def median_weight(self) -> float:
#         return torch.median(self.weight_tensor).item()

#     @_check_ablated
#     def max_weight(self) -> float:
#         return torch.max(self.weight_tensor).item()

#     @_check_ablated
#     def min_weight(self) -> float:
#         return torch.min(self.weight_tensor).item()

#     @_check_ablated
#     def get_fan_in(self) -> int:
#         return self.mask_tensor.sum().item()


# if __name__ == "__main__":

#     def test_member_funcs(filter_meter):
#         callables = [
#             filter_meter.mean_weight,
#             filter_meter.median_weight,
#             filter_meter.max_weight,
#             filter_meter.min_weight,
#             filter_meter.get_fan_in,
#         ]
#         for c in callables:
#             print(c())

#     shape = (16, 3, 3, 3)
#     weights = torch.rand(size=shape)
#     mask = torch.zeros(size=shape, dtype=torch.bool)
#     fan_in = 1
#     mask[:fan_in] = True
#     filter_meter = FilterMeter(0, weight_tensor=weights, mask_tensor=mask)
#     print("Before ablation...")
#     test_member_funcs(filter_meter)
#     print("After ablation...")
#     filter_meter.ablate()
#     test_member_funcs(filter_meter)
#     weights = torch.rand(size=shape)
#     mask = torch.zeros(size=shape, dtype=torch.bool)
#     filter_meter.weight_tensor = weights
#     filter_meter.mask_tensor = mask
#     print(len(filter_meter.weight_history))
#     print(len(filter_meter.mask_history))
