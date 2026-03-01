"""
Shared utilities for Project EXODUS.
Config loading, caching, logging.
"""

import math
import os
import json
import hashlib
import logging
import yaml
from pathlib import Path
from datetime import datetime

# ── Project root detection ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# ── Logging ─────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a module."""
    logger = logging.getLogger(f"exodus.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# ── Config ──────────────────────────────────────────────────────────
_config = None

def get_config() -> dict:
    """Load and cache the YAML configuration."""
    global _config
    if _config is None:
        with open(CONFIG_PATH, "r") as f:
            _config = yaml.safe_load(f)
    return _config

# ── Caching helpers ─────────────────────────────────────────────────
def _cache_dir() -> Path:
    cfg = get_config()
    d = PROJECT_ROOT / cfg["project"]["cache_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d

def _results_dir() -> Path:
    cfg = get_config()
    d = PROJECT_ROOT / cfg["project"]["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d

def cache_key(*parts) -> str:
    """Create a deterministic cache key from arbitrary parts."""
    raw = "_".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def load_cache(key: str, subfolder: str = ""):
    """Load a cached JSON/CSV file. Returns None if not found."""
    base = _cache_dir() / subfolder if subfolder else _cache_dir()
    json_path = base / f"{key}.json"
    csv_path = base / f"{key}.csv"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    if csv_path.exists():
        import pandas as pd
        return pd.read_csv(csv_path)
    return None

def save_cache(key: str, data, subfolder: str = "", fmt: str = "json"):
    """Save data to cache. Supports 'json' and 'csv' formats."""
    base = _cache_dir() / subfolder if subfolder else _cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path = base / f"{key}.json"
        with open(path, "w") as f:
            safe_json_dump(data, f, indent=2)
    elif fmt == "csv":
        import pandas as pd
        path = base / f"{key}.csv"
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            raise ValueError("CSV format requires a pandas DataFrame")
    return path

def save_result(name: str, data: dict):
    """Save a result to the results directory as JSON."""
    d = _results_dir()
    path = d / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        safe_json_dump(data, f, indent=2)
    return path


# ── NaN-safe JSON serialization ────────────────────────────────────

class _NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to null for RFC 8259 compliance."""

    def default(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)

    def encode(self, o):
        return super().encode(_sanitize_for_json(o))


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def safe_json_dump(data, fp, **kwargs):
    """Write JSON to file with NaN/Inf converted to null.

    Drop-in replacement for ``json.dump()`` that produces valid JSON
    (RFC 8259) by sanitizing non-finite floats.  Also applies
    ``default=str`` for non-serializable types (datetime, Path, etc.).
    """
    kwargs.setdefault("default", str)
    sanitized = _sanitize_for_json(data)
    json.dump(sanitized, fp, **kwargs)


def safe_json_dumps(data, **kwargs) -> str:
    """Return JSON string with NaN/Inf converted to null."""
    kwargs.setdefault("default", str)
    sanitized = _sanitize_for_json(data)
    return json.dumps(sanitized, **kwargs)
