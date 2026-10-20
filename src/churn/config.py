"""Configuration + logging helpers."""
import logging
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger(name)
