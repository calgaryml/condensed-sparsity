from typing import Callable, Any

from rigl_torch.exceptions import WandbRunNameException


class WandbRunName:
    def __init__(self, name: str):
        self.name = name
        self._verify_name()

    def _verify_name(self):
        if " " in self.name:
            raise WandbRunNameException(
                message="No spaces allowed in name", name=self.name
            )
        if len(self.name) > 128:
            raise WandbRunNameException(
                message="Name must be <= 128 chars", name=self.name
            )


def wandb_log_check(fn: Callable, log_to_wandb: bool = True) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        if log_to_wandb:
            return fn(*args, **kwargs)
        else:
            return None
    return wrapper
