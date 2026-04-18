"""Loguru logging setup with stdlib interception."""
from __future__ import annotations

import logging
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame, depth = sys._getframe(6), 6
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure loguru as the single logging sink.

    Intercepts all stdlib ``logging`` calls (including those from uvicorn,
    httpx, and openai) and routes them through loguru.

    Args:
        level: Minimum log level (e.g. ``"DEBUG"``, ``"INFO"``).
        json_logs: When True, emit newline-delimited JSON for log aggregators.
            When False, emit colorized human-readable output.
    """
    logger.remove()  # Remove default handler

    if json_logs:
        logger.add(sys.stdout, level=level, serialize=True)
    else:
        logger.add(
            sys.stdout,
            level=level,
            colorize=True,
            format=(
                "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
            ),
        )

    # Intercept all stdlib logging (used by existing code + uvicorn + httpx)
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "httpx", "openai"):
        log = logging.getLogger(name)
        log.handlers = [_InterceptHandler()]
        log.propagate = False






print("Logging configured with loguru.")
