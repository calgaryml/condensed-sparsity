import logging
import pathlib
from datetime import date


def get_logger(log_path: str, name: str, rank: int = -1) -> logging.Logger:
    log_path = pathlib.Path(log_path)
    if not log_path.is_dir():
        log_path.mkdir()
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    current_date = date.today().strftime("%Y-%m-%d")
    # logformat = (
    #     "[%(levelname)s] %(asctime)s %(name)s %(rank)s "
    #     "%(funcName)s (%(lineno)d) : %(message)s"
    # )
    logformat = (
        "[%(levelname)s] %(asctime)s G- %(name)s - %(funcName)s "
        "(%(lineno)d) : %(message)s"
    )
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format=logformat,
        handlers=[
            logging.FileHandler(log_path / f"processor_{current_date}.log"),
            logging.StreamHandler(),
        ],
    )
    # logger = logging.LoggerAdapter(logger, {"rank": f"rank: {rank}"})
    return logger


if __name__ == "__main__":
    logger = get_logger("./test_logging/", __name__, rank=0)
    logger.info("Hello world!")
