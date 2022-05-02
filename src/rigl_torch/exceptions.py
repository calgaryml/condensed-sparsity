import torch


class ConstantFanInException(Exception):
    def __init__(self, fan_in_tensor: torch.Tensor) -> None:
        super().__init__(
            f"Non-constant fan-in detected. Fan-in tensor: {fan_in_tensor}"
        )
