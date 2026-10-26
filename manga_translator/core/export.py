"""Saving translated pages: file names (prefix + number), PNG/JPG/WEBP encoding, and auto-save
that keeps <save folder>/<chapter>/ in step with each page's result."""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .project import _replace_with_retry, atomic_write_text, imread

FORMATS = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}
CHAPTER_TOKEN = "{ตอน}"
EXPORTED_FILE = "exported.json"
_INVALID = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class ExportError(RuntimeError):
    pass


def clean_part(text: str) -> str:
    """Remove characters Windows forbids in file names."""
    return _INVALID.sub("", text).rstrip(" .")


def make_prefix(template: str, chapter: str) -> str:
    return clean_part(template.replace(CHAPTER_TOKEN, chapter))


def output_name(prefix: str, number: int, digits: int, fmt: str) -> str:
    if fmt not in FORMATS:
        raise ExportError(f"ไม่รู้จักชนิดไฟล์ {fmt}")
    return f"{prefix}{number:0{max(1, digits)}d}{FORMATS[fmt]}"


def plan_names(count: int, prefix: str, start: int, digits: int, fmt: str) -> list[str]:
    return [output_name(prefix, start + i, digits, fmt) for i in range(count)]


def encode(img: np.ndarray, fmt: str, quality: int) -> np.ndarray:
    params = []
    if fmt == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif fmt == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    ok, buf = cv2.imencode(FORMATS[fmt], img, params)
    if not ok:
        raise ExportError(f"แปลงภาพเป็น {fmt} ไม่ได้")
    return buf


def write_image(src_final: Path, dest: Path, fmt: str, quality: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = encode(imread(src_final), fmt, quality)
    tmp = dest.with_name(dest.name + ".tmp")
    buf.tofile(str(tmp))
    _replace_with_retry(tmp, dest)


def export_pages(items: list[tuple[Path, Path]], fmt: str, quality: int, overwrite: bool) -> tuple[list[Path], list[Path]]:
    """items: (final.png, destination). Returns (written, skipped because they already exist)."""
    written, skipped = [], []
    for src, dest in items:
        if dest.exists() and not overwrite:
            skipped.append(dest)
            continue
        write_image(src, dest, fmt, quality)
        written.append(dest)
    return written, skipped


# ---- auto-save ----------------------------------------------------------------------------

def _stat(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_size, st.st_mtime_ns


@dataclass
class AutoSaveStatus:
    folder: str = ""
    saved: int = 0
    last_file: str = ""
    last_error: str = ""
    conflicts: list[str] = field(default_factory=list)


class AutoSaver:
    """Writes a page's final image to <save_dir>/<chapter>/<prefix><number>.<ext>.

    work_dir/exported.json remembers which file each page was written to (name, size, mtime).
    When a page's number changes (pages reordered) the old file is removed, but only if it is
    still exactly the file this app wrote. Files the app did not write are never overwritten.
    """

    def __init__(self, settings):
        self.settings = settings
        self.status = AutoSaveStatus()
        self._lock = threading.Lock()  # the AI thread and the UI (after edits) both save

    @property
    def enabled(self) -> bool:
        return bool(self.settings.autosave and self.settings.save_dir)

    def chapter_dir(self, chapter: str) -> Path:
        return Path(self.settings.save_dir) / clean_part(chapter)

    def name_for(self, chapter: str, index: int) -> str:
        s = self.settings
        return output_name(make_prefix(s.save_prefix, chapter), s.save_start + index, s.save_digits, s.save_format)

    def _load(self, work_dir: Path) -> dict:
        path = Path(work_dir) / EXPORTED_FILE
        try:
            return json.loads(path.read_text("utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            return {}

    def _matches(self, path: Path, rec: dict) -> bool:
        return path.exists() and list(_stat(path)) == [rec.get("size"), rec.get("mtime_ns")]

    def save(self, work_dir: Path, chapter: str, page: str, index: int, final: Path) -> str | None:
        """Save one page. Returns the written file name, or None when skipped (status says why)."""
        if not self.enabled or not final.exists():
            return None
        with self._lock:
            return self._save(work_dir, chapter, page, index, final)

    def _save(self, work_dir: Path, chapter: str, page: str, index: int, final: Path) -> str | None:
        folder = self.chapter_dir(chapter)
        name = self.name_for(chapter, index)
        dest = folder / name
        manifest = self._load(work_dir)
        self.status.folder = str(folder)
        ours = [p for p, rec in manifest.items() if rec.get("file") == name and self._matches(dest, rec)]
        if dest.exists() and not ours:
            msg = f"ไม่เขียนทับ {name}: มีไฟล์นี้อยู่แล้วและไม่ได้มาจากการเซฟอัตโนมัติ"
            if name not in self.status.conflicts:
                self.status.conflicts.append(name)
            self.status.last_error = msg
            return None
        old = manifest.get(page)
        if old and old.get("file") != name:
            old_path = folder / old["file"]
            if self._matches(old_path, old):
                old_path.unlink()
        try:
            write_image(final, dest, self.settings.save_format, self.settings.save_quality)
        except (OSError, ExportError) as e:
            self.status.last_error = f"เซฟ {name} ไม่ได้: {e}"
            return None
        for p in ours:  # another page's old record pointed at this name; that file is now ours
            if p != page:
                manifest.pop(p, None)
        size, mtime = _stat(dest)
        manifest[page] = {"file": name, "size": size, "mtime_ns": mtime}
        atomic_write_text(Path(work_dir) / EXPORTED_FILE, json.dumps(manifest, ensure_ascii=False, indent=1))
        if name in self.status.conflicts:
            self.status.conflicts.remove(name)
        self.status.saved = len(manifest)
        self.status.last_file = name
        self.status.last_error = ""
        return name

    def saved_pages(self, work_dir: Path) -> set[str]:
        folder_records = self._load(work_dir)
        return set(folder_records)


def open_folder(path: Path) -> None:
    if Path(path).exists():
        os.startfile(str(path))
