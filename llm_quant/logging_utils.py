from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def configure_logging(cfg: Dict[str, Any] | None = None, verbose: bool = False) -> None:
    logging_cfg = (cfg or {}).get("logging", {})
    level_name = str(logging_cfg.get("level", "INFO")).upper()
    if verbose:
        level_name = "DEBUG"
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging_cfg.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    datefmt = logging_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    console_enabled = bool(logging_cfg.get("console", True))

    handlers: list[logging.Handler] = []
    if console_enabled:
        handlers.append(logging.StreamHandler())

    file_path = logging_cfg.get("file")
    if file_path:
        log_path = Path(file_path)
        if bool(logging_cfg.get("file_timestamped", True)):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_path.with_name(f"{log_path.stem}_{stamp}{log_path.suffix}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers if handlers else None,
        force=True,
    )
