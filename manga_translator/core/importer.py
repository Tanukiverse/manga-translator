"""Bring pages in from archives (ZIP/CBZ, RAR/CBR, 7Z/CB7) or loose image files.

Everything is copied into the work folder: the user's own files and archives are never
changed. Imported chapters live in <work>/_imported/<name>/ and open like any folder.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .pages import natural_key

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ZIP_EXT = {".zip", ".cbz"}
TAR_EXT = {".rar", ".cbr", ".7z", ".cb7"}  # read by Windows' own tar.exe (libarchive)
ARCHIVE_EXT = ZIP_EXT | TAR_EXT
IMPORTED = "_imported"
ADDED = "_added"  # images dropped onto an open chapter: <work>/<chapter>/_added/
MAX_IMAGES = 2000


class ImportFailed(RuntimeError):
    pass


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXT


def is_archive(p: Path) -> bool:
    return p.suffix.lower() in ARCHIVE_EXT


def _unique(path: Path) -> Path:
    """path, or path with (2), (3)… so nothing existing is overwritten."""
    if not path.exists():
        return path
    for n in range(2, 10000):
        cand = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not cand.exists():
            return cand
    raise ImportFailed(f"ตั้งชื่อไม่ได้: {path}")


def _bsdtar() -> str:
    # Git for Windows puts GNU tar first on PATH; only Windows' bsdtar reads RAR and 7z
    path = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"
    if not path.exists():
        raise ImportFailed("ไม่พบ tar.exe ของ Windows (ต้องใช้ Windows 10 เวอร์ชัน 1803 ขึ้นไป) จึงเปิด RAR/7Z ไม่ได้")
    return str(path)


def import_archive(archive: Path, work_root: Path) -> Path:
    """Extract the archive's images (in natural order of their inner paths) into a new chapter
    folder and return it. Names are numbered so pages from sub-folders keep their order."""
    archive = Path(archive)
    Path(work_root).mkdir(parents=True, exist_ok=True)
    dest = _unique(Path(work_root) / IMPORTED / archive.stem)
    tmp = Path(tempfile.mkdtemp(prefix="mt_import_", dir=Path(work_root)))
    try:
        entries: list[tuple[str, Path]] = []  # (path inside archive, extracted file)
        if archive.suffix.lower() in ZIP_EXT:
            try:
                with zipfile.ZipFile(archive) as zf:
                    names = [n for n in zf.namelist() if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXT]
                    for i, name in enumerate(names[:MAX_IMAGES]):
                        out = tmp / f"{i}{Path(name).suffix.lower()}"  # never trust inner paths (zip-slip)
                        with zf.open(name) as src, out.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        entries.append((name, out))
            except zipfile.BadZipFile as e:
                raise ImportFailed(f"เปิดไฟล์ {archive.name} ไม่ได้: ไฟล์เสียหรือไม่ใช่ ZIP") from e
        else:
            r = subprocess.run([_bsdtar(), "-xf", str(archive), "-C", str(tmp)], capture_output=True, text=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode != 0:
                raise ImportFailed(f"แตกไฟล์ {archive.name} ไม่ได้: {(r.stderr or r.stdout).strip()[:200]}")
            for f in tmp.rglob("*"):
                if f.is_file() and is_image(f):
                    entries.append((f.relative_to(tmp).as_posix(), f))
        if not entries:
            raise ImportFailed(f"ไม่พบภาพใน {archive.name}")
        entries.sort(key=lambda e: natural_key(e[0]))
        dest.mkdir(parents=True)
        digits = max(3, len(str(len(entries))))
        for n, (name, f) in enumerate(entries[:MAX_IMAGES], 1):
            shutil.move(str(f), dest / f"{n:0{digits}d}_{Path(name).stem}{Path(name).suffix.lower()}")
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def copy_images(images: list[Path], dest: Path) -> list[Path]:
    """Copy images into dest (created if needed) without overwriting; returns the copies."""
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for p in images:
        target = _unique(dest / p.name)
        shutil.copy2(p, target)
        out.append(target)
    return out


def new_chapter_from_images(images: list[Path], work_root: Path) -> Path:
    """A new imported chapter holding copies of loose images (named after their folder)."""
    name = images[0].parent.name or "ภาพที่ลากมา"
    dest = _unique(Path(work_root) / IMPORTED / name)
    copy_images(images, dest)
    return dest


def classify(paths: list[Path]) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """Split dropped paths into (folders, archives, images, other)."""
    folders, archives, images, other = [], [], [], []
    for p in map(Path, paths):
        if p.is_dir():
            folders.append(p)
        elif is_archive(p):
            archives.append(p)
        elif is_image(p):
            images.append(p)
        else:
            other.append(p)
    return folders, archives, images, other
