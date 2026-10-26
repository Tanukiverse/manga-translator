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
    ollama_model: str = "hf.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
    ollama_keep_alive: str = "5m"
    font_family: str = ""  # empty = per-language default
    text_color: str = "#000000"
    outline_color: str = ""
    use_gpu: bool = True
    ui_font_size: int = 11  # pt; buttons and menus scale with it
    glossary: dict[str, str] = field(default_factory=dict)
    story_notes: str = ""
    # page list
    page_sort: str = "name"  # name | mtime | size | manual
    page_sort_desc: bool = False
    skip_done: bool = True  # batch runs skip pages that finished every step
    # saving (preset for auto-save and the Export dialog); {ตอน} = chapter (source folder) name
    save_dir: str = ""
    autosave: bool = False
    save_prefix: str = "{ตอน}_"
    save_start: int = 1
    save_digits: int = 3
    save_format: str = "png"  # png | jpg | webp
    save_quality: int = 92
    export_dir: str = ""  # last folder used by the Export dialog

    @classmethod
    def load(cls) -> "Settings":
        if not SETTINGS_PATH.exists():
            return cls()
        data = json.loads(SETTINGS_PATH.read_text("utf-8"))
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        atomic_write_text(SETTINGS_PATH, json.dumps(asdict(self), ensure_ascii=False, indent=2))
