from typing import Callable, Any, Dict
import omegaconf
import wandb

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


def _wandb_log_check(fn: Callable, log_to_wandb: bool = True) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        if log_to_wandb:
            return fn(*args, **kwargs)
        else:
            return None

    return wrapper


def init_wandb(cfg: omegaconf.DictConfig, wandb_init_kwargs: Dict[str, Any]):
    # We override logging functions now to avoid any calls
    if not cfg.wandb.log_to_wandb:
        print("No logging to WANDB! See cfg.wandb.log_to_wandb")
        wandb.log = _wandb_log_check(wandb.log, cfg.wandb.log_to_wandb)
        wandb.log_artifact = _wandb_log_check(
            wandb.log_artifact, cfg.wandb.log_to_wandb
        )
        wandb.watch = _wandb_log_check(wandb.watch, cfg.wandb.log_to_wandb)
        return None
    _ = WandbRunName(name=cfg.experiment.name)  # Verify name is OK
    run = wandb.init(
        name=cfg.experiment.name,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg=cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method=cfg.wandb.start_method),
        dir=cfg.paths.logs,
        **wandb_init_kwargs,
    )
    return run
