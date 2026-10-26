"""Page document: the editable state of one manga page, stored as page.json beside its image layers.

Layout of a page folder:
    original.png  - source image (never modified)
    mask.png      - text mask (white = remove), editable by the user
    cleaned.png   - original with text removed
    final.png     - cleaned + typeset translation
    page.json     - regions and settings
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Region:
    id: int
    bbox: list[int]  # x, y, w, h in page pixels
    source_text: str = ""
    translation: str = ""
    vertical: bool = False
    # typesetting
    font_family: str = ""  # empty = project default
    font_size: int = 0  # 0 = auto-fit
    color: str = "#000000"
    outline: str = ""  # empty = none, e.g. "#ffffff"
    align: str = "center"
    wrap: str = "auto"  # "auto": wrap to fit + Enter breaks; "manual": break only where the user pressed Enter
    locked: bool = False  # skip in automatic steps; user edited it
    text_rect: list[int] | None = None  # x, y, w, h where the user put the translation; None = auto

    @property
    def rect(self) -> tuple[int, int, int, int]:
        x, y, w, h = self.bbox
        return x, y, w, h


@dataclass
class PageDoc:
    folder: Path
    source_lang: str = "ja"
    target_lang: str = "th"
    regions: list[Region] = field(default_factory=list)
    # which pipeline steps have produced output since the last edit
    done: dict[str, bool] = field(default_factory=dict)

    # ---- paths -------------------------------------------------------
    @property
    def original_path(self) -> Path:
        return self.folder / "original.png"

    @property
    def mask_path(self) -> Path:
        return self.folder / "mask.png"

    @property
    def cleaned_path(self) -> Path:
        return self.folder / "cleaned.png"

    @property
    def final_path(self) -> Path:
        return self.folder / "final.png"

    @property
    def json_path(self) -> Path:
        return self.folder / "page.json"

    # ---- images ------------------------------------------------------
    def load_original(self) -> np.ndarray:
        return imread(self.original_path)

    def load_layer(self, name: str) -> np.ndarray | None:
        p = self.folder / f"{name}.png"
        if not p.exists():
            return None
        return imread(p, gray=(name == "mask"))

    def save_layer(self, name: str, img: np.ndarray) -> None:
        imwrite(self.folder / f"{name}.png", img)

    # ---- regions -----------------------------------------------------
    def next_region_id(self) -> int:
        return max((r.id for r in self.regions), default=0) + 1

    def region(self, rid: int) -> Region | None:
        return next((r for r in self.regions if r.id == rid), None)

    # ---- persistence -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "done": self.done,
            "regions": [asdict(r) for r in self.regions],
        }

    def save(self) -> None:
        atomic_write_text(self.json_path, json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, folder: Path) -> "PageDoc":
        folder = Path(folder)
        doc = cls(folder=folder)
        if doc.json_path.exists():
            data = json.loads(doc.json_path.read_text("utf-8"))
            doc.source_lang = data.get("source_lang", doc.source_lang)
            doc.target_lang = data.get("target_lang", doc.target_lang)
            doc.done = data.get("done", {})
            known = Region.__dataclass_fields__.keys()
            doc.regions = [Region(**{k: v for k, v in r.items() if k in known}) for r in data.get("regions", [])]
        return doc

    @classmethod
    def create(cls, image_path: Path, work_dir: Path, source_lang: str = "ja", target_lang: str = "th") -> "PageDoc":
        """Create (or reopen) the page folder for an image inside work_dir."""
        image_path = Path(image_path)
        folder = Path(work_dir) / image_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        doc = cls.load(folder) if (folder / "page.json").exists() else cls(folder, source_lang, target_lang)
        if not doc.original_path.exists():
            img = imread(image_path)
            doc.save_layer("original", img)
            doc.save()
        return doc


# ---- file helpers (Windows-safe) -------------------------------------

def _replace_with_retry(tmp: Path, dest: Path) -> None:
    # Windows refuses to replace a file another thread/process has open for a moment.
    for attempt in range(40):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, "utf-8")
    _replace_with_retry(tmp, path)


def imread(path: Path, gray: bool = False) -> np.ndarray:
    # cv2.imread cannot open non-ASCII (e.g. Thai/Japanese) paths on Windows
    data = np.fromfile(str(path), dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    tmp = path.with_name(path.name + ".tmp")
    buf.tofile(str(tmp))
    _replace_with_retry(tmp, path)
