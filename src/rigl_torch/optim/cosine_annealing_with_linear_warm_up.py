import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithLinearWarmUp(_LRScheduler):
    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
        lr=0.001,
        warm_up_steps=20,
        init_lr=None,
    ):
        """Cosine annealing lr decay with linear warm up.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.warm_up_steps = warm_up_steps
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

        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            # Cosine annealing phases
            return [
                group["lr"]
                + (self.lr - self.eta_min)
                * (1 - math.cos(math.pi / self.T_max))
                / 2
                for group in self.optimizer.param_groups
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
