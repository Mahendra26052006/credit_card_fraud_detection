"""Shared utilities: seeding, logging, IO, timing."""

from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, RAW_DATA_DIR


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix Python, NumPy, and (when present) backend RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str = "fraud", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def save_json(payload: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None) -> Iterator[Dict[str, float]]:
    log = logger or get_logger()
    started = time.perf_counter()
    result: Dict[str, float] = {"seconds": 0.0}
    log.info("Starting: %s", label)
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - started
        log.info("Finished: %s (%.2fs)", label, result["seconds"])


def format_pct(value: float, digits: int = 4) -> str:
    return f"{100.0 * value:.{digits}f}%"
