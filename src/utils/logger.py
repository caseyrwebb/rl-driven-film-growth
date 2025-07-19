"""Utility for logging configuration."""

import logging
import os


def setup_logging(
    log_level=None,
    log_to_file=False,
    log_filename="local-log.log",
):
    """
    Configure the logging settings for the project.
    Priority:
    1. LOG_LEVEL environment variable (if set)
    2. log_level argument (if given)
    3. Defaults to INFO
    """
    log_level_env = os.environ.get("LOG_LEVEL", None)
    if log_level_env is not None:
        level = getattr(logging, log_level_env.upper(), logging.INFO)
    elif log_level is not None:
        if isinstance(log_level, str):
            level = getattr(logging, log_level.upper(), logging.INFO)
        else:
            level = log_level
    else:
        level = logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_filename))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name=None):
    """Get a logger with the specified name."""
    return logging.getLogger(name)
