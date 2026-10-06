import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    override = os.environ.get("AIE_LOCAL_LLM_CONFIG_DIR")
    return Path(override) if override else REPO_ROOT / "config"


@lru_cache
def load_yaml(name: str) -> dict:
    path = config_dir() / f"{name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f) or {}


def clear_config_cache() -> None:
    load_yaml.cache_clear()
