import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union, List
from omegaconf.listconfig import ListConfig


class StepLrWithLinearWarmUp(_LRScheduler):
    def __init__(
        self,
        optimizer,
        gamma,
        step_size: Union[List[int], int],
        warm_up_steps,
        last_epoch=-1,
        verbose=False,
        lr=0.001,
        init_lr=None,
    ):
        if isinstance(step_size, ListConfig):
            step_size = list(step_size)
        self.step_size = step_size
        self.warm_up_steps = warm_up_steps
        self.gamma = gamma
        if init_lr is None:
            init_lr = 1e-6
        if lr is None:
            lr = optimizer.param_groups[0]["lr"]
        self.lr = lr
        self._linear_warmup_lrs = np.linspace(
            init_lr, self.lr, self.warm_up_steps
        )
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self.last_epoch < self.warm_up_steps:  # Linear warm up phase
            return [self._linear_warmup_lrs[self.last_epoch]] * len(
                self.optimizer.param_groups
            )
        else:  # Step phase
            if isinstance(self.step_size, list):
                if (
                    len(self.step_size) != 0
                    and self.last_epoch == self.step_size[0]
                ):
                    self.step_size.pop(0)
                    return [
                        group["lr"] * self.gamma
                        for group in self.optimizer.param_groups
                    ]
                else:
                    return [
                        group["lr"] for group in self.optimizer.param_groups
                    ]
            else:
                if (self.last_epoch == 0) or (
                    self.last_epoch % self.step_size != 0
                ):
                    return [
                        group["lr"] for group in self.optimizer.param_groups
                    ]
                else:
                    return [
                        group["lr"] for group in self.optimizer.param_groups
                    ]
