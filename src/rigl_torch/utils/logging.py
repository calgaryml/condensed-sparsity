import omegaconf
import logging
import pathlib
from typing import Optional
from datetime import date


def get_logger(
    cfg: omegaconf.DictConfig, name: str, rank: Optional[int] = None
) -> logging.Logger:
    log_path = pathlib.Path(cfg.paths.logs)
    if not log_path.is_dir():
        log_path.mkdir()
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)
    current_date = date.today().strftime("%Y-%m-%d")
    # logformat = "[%(levelname)s] %(asctime)s G- %(name)s -%(rank)s -
    # %(funcName)s (%(lineno)d) : %(message)s"
    logformat = (
        "[%(levelname)s] %(asctime)s G- %(name)s - %(funcName)s "
        "(%(lineno)d) : %(message)s"
    )
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.DEBUG,
        format=logformat,
        handlers=[
            logging.FileHandler(log_path / f"processor_{current_date}.log"),
            logging.StreamHandler(),
        ],
    )
    # logger = logging.LoggerAdapter(logger, {"rank": f"rank: {rank}"})
    # logger.info("hell world")
    return logger
