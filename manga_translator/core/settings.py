"""App settings stored in settings.json at the project root."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .paths import ROOT, SETTINGS_PATH
from .project import atomic_write_text


@dataclass
class Settings:
    work_dir: str = str(ROOT / "work")
    source_lang: str = "ja"
    target_lang: str = "th"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_keep_alive: str = "5m"
    font_family: str = ""  # empty = per-language default
    text_color: str = "#000000"
    outline_color: str = ""
    use_gpu: bool = True
    glossary: dict[str, str] = field(default_factory=dict)
    story_notes: str = ""

    @classmethod
    def load(cls) -> "Settings":
        if not SETTINGS_PATH.exists():
            return cls()
        data = json.loads(SETTINGS_PATH.read_text("utf-8"))
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        atomic_write_text(SETTINGS_PATH, json.dumps(asdict(self), ensure_ascii=False, indent=2))
