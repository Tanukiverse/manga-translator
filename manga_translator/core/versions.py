"""Page history: snapshots of a page's state (page.json, mask, cleaned, final) before each run,
reset or restore, so a worse result can be undone.

Snapshots hard-link the files. Layers are written by replacing the file (atomic write), so a
link keeps the old picture while the page moves on: a snapshot only costs disk space for
pictures that actually changed afterwards.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .project import PageDoc, atomic_write_text

FILES = ["page.json", "mask.png", "cleaned.png", "final.png"]
DIR = "versions"
DEFAULT_KEEP = 10


@dataclass
class Version:
    path: Path
    label: str
    time: float
    regions: int
    has_final: bool

    @property
    def preview(self) -> Path | None:
        for name in ("final.png", "cleaned.png"):
            if (self.path / name).exists():
                return self.path / name
        return None


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:  # other drive, FAT, no permission: fall back to a real copy
        shutil.copy2(src, dst)


def _state_key(folder: Path) -> list:
    """What a snapshot would contain: unchanged state means no new snapshot is needed."""
    out = []
    for name in FILES:
        p = folder / name
        out.append([name, p.stat().st_size, p.stat().st_mtime_ns] if p.exists() else [name, None, None])
    return out


def _has_work(folder: Path) -> bool:
    if any((folder / n).exists() for n in FILES[1:]):
        return True
    doc = PageDoc.load(folder)
    return bool(doc.regions)


def snapshot(doc: PageDoc, label: str, keep: int = DEFAULT_KEEP) -> Path | None:
    """Save the page's current state (unless there is nothing yet or it equals the last one)."""
    folder = doc.folder
    if not _has_work(folder):
        return None
    key = _state_key(folder)
    versions = list_versions(doc)
    if versions:
        meta = json.loads((versions[0].path / "meta.json").read_text("utf-8"))
        if meta.get("state") == key:
            return None
    root = folder / DIR
    root.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = root / stamp
    n = 1
    while dest.exists():
        n += 1
        dest = root / f"{stamp}-{n}"
    dest.mkdir()
    for name in FILES:
        if (folder / name).exists():
            _link_or_copy(folder / name, dest / name)
    regions = len(PageDoc.load(folder).regions)
    atomic_write_text(dest / "meta.json", json.dumps(
        {"label": label, "time": time.time(), "regions": regions, "state": key}, ensure_ascii=False))
    prune(doc, keep)
    return dest


def list_versions(doc: PageDoc) -> list[Version]:
    """Newest first."""
    root = doc.folder / DIR
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        out.append(Version(d, meta.get("label", ""), meta.get("time", 0.0), meta.get("regions", 0),
                           (d / "final.png").exists()))
    return sorted(out, key=lambda v: v.time, reverse=True)


def prune(doc: PageDoc, keep: int) -> None:
    for v in list_versions(doc)[keep:]:
        shutil.rmtree(v.path, ignore_errors=True)


def restore(doc: PageDoc, version: Version, keep: int = DEFAULT_KEEP) -> PageDoc:
    """Put a saved version back. The current state is saved first, so this can be undone too."""
    snapshot(doc, "ก่อนคืนค่าเวอร์ชัน", keep)
    folder = doc.folder
    for name in FILES:
        src, dst = version.path / name, folder / name
        if src.exists():
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copy2(src, tmp)  # a copy, so the version's own link stays untouched
            os.replace(tmp, dst)
        elif dst.exists() and name != "page.json":
            dst.unlink()  # that layer did not exist in the version yet
    return PageDoc.load(folder)


def reset(doc: PageDoc, keep: int = DEFAULT_KEEP) -> PageDoc:
    """Back to the untouched original: no boxes, no removal, no translation. Saved first."""
    snapshot(doc, "ก่อนรีเซ็ต", keep)
    for name in FILES[1:]:
        p = doc.folder / name
        if p.exists():
            p.unlink()
    fresh = PageDoc(doc.folder, doc.source_lang, doc.target_lang)
    fresh.save()
    return fresh
