"""Загрузка настроек проекта из config/settings.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Путь из конфига → абсолютный путь от корня репозитория."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
