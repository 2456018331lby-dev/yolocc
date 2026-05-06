"""
Structured logging with rich formatting.
Provides a consistent logger across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ── Theme ────────────────────────────────────────────────────────────────────

THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "debug": "dim",
})

console = Console(theme=THEME, stderr=True)

# ── Logger setup ─────────────────────────────────────────────────────────────

_LOGGERS: dict = {}
_INITIALIZED = False


def _init_root():
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    root = logging.getLogger("yolocc")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    rich_handler.setLevel(logging.INFO)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(rich_handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a named logger under the 'yolocc' namespace."""
    _init_root()
    full_name = f"yolocc.{name}" if not name.startswith("yolocc.") else name

    if full_name in _LOGGERS:
        return _LOGGERS[full_name]

    logger = logging.getLogger(full_name)
    if level is not None:
        logger.setLevel(level)
    _LOGGERS[full_name] = logger
    return logger


def add_file_handler(
    log_file: Union[str, Path],
    level: int = logging.DEBUG,
    name: str = "yolocc",
) -> None:
    """Add a file handler to the root logger."""
    _init_root()
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger(name).addHandler(fh)


# ── Convenience functions ────────────────────────────────────────────────────

def log_section(title: str, logger: Optional[logging.Logger] = None):
    """Print a section header."""
    log = logger or get_logger("main")
    log.info("─" * 60)
    log.info(title)
    log.info("─" * 60)


def log_kv(key: str, value, logger: Optional[logging.Logger] = None):
    """Log a key-value pair aligned."""
    log = logger or get_logger("main")
    log.info(f"  {key:20s}: {value}")


def log_table(headers: list, rows: list, logger: Optional[logging.Logger] = None):
    """Log a simple table."""
    log = logger or get_logger("main")
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    log.info(f"  {header_line}")
    log.info("  " + "-+-".join("-" * w for w in widths))

    # Rows
    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        log.info(f"  {line}")


def log_success(msg: str, logger: Optional[logging.Logger] = None):
    """Log a success message."""
    log = logger or get_logger("main")
    console.print(f"[success]{msg}[/success]")


def log_error(msg: str, logger: Optional[logging.Logger] = None):
    """Log an error message."""
    log = logger or get_logger("main")
    log.error(msg)
