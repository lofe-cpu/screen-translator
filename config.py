from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "ScreenTranslator"
APP_DIR = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
SETTINGS_FILE = APP_DIR / "settings.json"


@dataclass
class Settings:
    src: str = "auto"
    dst: str = "ru"
    ocr: str = "rapid"
    translator: str = "google"
    gpu: bool = True
    monitor: int = 0
    region: tuple[int, int, int, int] | None = None
    interval: float = 0.03
    settle: float = 0.35
    min_score: float = 0.6
    min_height: int = 8
    merge_lines: bool = True
    opacity: float = 0.92
    autostart: bool = False
    hotkey_toggle: str = "ctrl+alt+t"
    hotkey_region: str = "ctrl+alt+r"
    hotkey_full: str = "ctrl+alt+f"
    hotkey_peek: str = "ctrl+alt+h"
    extra: dict = field(default_factory=dict)


def load_settings() -> Settings:
    s = Settings()
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return s
    names = {f.name for f in fields(Settings)}
    for k, v in data.items():
        if k in names:
            setattr(s, k, v)
    if s.region is not None:
        s.region = tuple(s.region)
    return s


def save_settings(s: Settings):
    APP_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(s)
    data.pop("extra", None)
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
