import torch


class ConstantFanInException(Exception):
    def __init__(self, fan_in_tensor: torch.Tensor) -> None:
        super().__init__(
            f"Non-constant fan-in detected. Fan-in tensor: {fan_in_tensor}"
        )


class InvalidAblatedNeuronException(Exception):
    def __init__(self, mask_index: int) -> None:
        super().__init__(
            "Initally ablated neuron detected with elements != False. I found"
            f"an invalid filter with values in mask index = {mask_index}"
        )


class WandbRunNameException(Exception):
    def __init__(self, message, name) -> None:
        super().__init__(f"Wandb run name of {name} is invalid! " f"{message}")
