import os
import sys

from loguru import logger

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> - "
    "<level>{level}</level> - "
    "<white>{message}</white> - "
    "<blue>{file}:{line}</blue>"
)


def setup_logging() -> None:
    logger.remove()
    level = os.environ.get("PEARL_LOG_LEVEL", "INFO").upper()
    logger.add(
        sys.stderr,
        level=level,
        format=_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )


def get_logger(name: str):
    return logger.bind(name=name)


if os.environ.get("PEARL_LOG_LEVEL", "INFO").upper() != "NONE":
    setup_logging()
