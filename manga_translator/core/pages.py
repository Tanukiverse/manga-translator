"""Page order: natural name sort, date, size, or the user's own drag order (saved per chapter)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .project import atomic_write_text

SORT_KEYS = ["name", "mtime", "size", "manual"]
SORT_LABELS = {"name": "ชื่อไฟล์", "mtime": "วันที่แก้ไข", "size": "ขนาดไฟล์", "manual": "ลากเรียงเอง"}
ORDER_FILE = "order.json"


def natural_key(name: str) -> tuple:
    """Compare digit runs as numbers: p2 < p10, ตอน9 < ตอน10."""
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part)
                 for part in re.split(r"(\d+)", name.lower()) if part)


def load_order(work_dir: Path) -> list[str]:
    path = Path(work_dir) / ORDER_FILE
    if not path.exists():
        return []
    try:
        names = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return [n for n in names if isinstance(n, str)]


def save_order(work_dir: Path, paths: list[Path]) -> None:
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_text(Path(work_dir) / ORDER_FILE, json.dumps([p.name for p in paths], ensure_ascii=False, indent=1))


def sort_pages(paths: list[Path], key: str, reverse: bool = False, manual: list[str] | None = None) -> list[Path]:
    by_name = sorted(paths, key=lambda p: natural_key(p.name))
    if key == "manual":
        # saved order first; files added since then go after it in name order; vanished names drop out
        rank = {name: i for i, name in enumerate(manual or [])}
        return sorted(by_name, key=lambda p: (0, rank[p.name]) if p.name in rank else (1, 0))
    if key == "mtime":
        out = sorted(by_name, key=lambda p: p.stat().st_mtime)
    elif key == "size":
        out = sorted(by_name, key=lambda p: p.stat().st_size)
    else:
        out = by_name
    return out[::-1] if reverse else out
