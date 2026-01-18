from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    *,
    run_dir: Path,
    logger_name: str = "ig_quant_bot",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_filename: str = "run.log",
) -> logging.Logger:
    """Create a consistent logger used across backtest and live runners.

    - Logs to console (INFO by default)
    - Logs to a per-run file in run_dir (DEBUG by default)

    This function is idempotent: calling it multiple times does not duplicate handlers.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_filename

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path) for h in logger.handlers):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(file_level)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(console_level)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.propagate = False
    return logger
