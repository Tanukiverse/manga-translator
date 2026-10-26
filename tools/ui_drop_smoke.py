"""Drop files on the real window the way Explorer does: an archive, loose images, a folder, junk.

Usage: .venv\\Scripts\\python tools\\ui_drop_smoke.py <scratch dir>
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manga_translator import crashlog  # noqa: E402

crashlog.install()
from _isolate import isolate_settings  # noqa: E402

isolate_settings()
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QTimer, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from manga_translator.core import importer  # noqa: E402
from manga_translator.core.project import imwrite  # noqa: E402
from manga_translator.ui.main_window import MainWindow  # noqa: E402

boxes: list[str] = []


def close_boxes():
    w = QApplication.activeModalWidget()
    if isinstance(w, QMessageBox):
        boxes.append(w.text())
        w.accept()


def mime(paths):
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


def drop(win, paths):
    data = mime(paths)  # the event only points at the data: keep it alive until the drop is handled
    win.dropEvent(QDropEvent(QPointF(400, 300), Qt.DropAction.CopyAction, data,
                             Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    QApplication.processEvents()


def main() -> int:
    root = Path(sys.argv[1])
    shutil.rmtree(root, ignore_errors=True)
    src = root / "src"
    loose = root / "loose"
    src.mkdir(parents=True)
    loose.mkdir()
    for i in (1, 2, 10):
        imwrite(src / f"p{i}.png", np.full((300, 200, 3), 20 * i, np.uint8))
    arc = root / "ตอนที่ 7.cbz"
    with zipfile.ZipFile(arc, "w") as zf:
        for i in (10, 2, 1):
            zf.write(src / f"p{i}.png", f"chapter/p{i}.png")
    for name in ("extra_a.png", "extra_b.png"):
        imwrite(loose / name, np.full((300, 200, 3), 200, np.uint8))
    (root / "notes.txt").write_text("not a page")

    app = QApplication(sys.argv)
    t = QTimer()
    t.timeout.connect(close_boxes)
    t.start(100)
    win = MainWindow()
    win.settings.work_dir = str(root / "work")
    win.show()

    # the window accepts a file drag from Explorer
    data = mime([arc])  # kept alive while the event is used
    enter = QDragEnterEvent(QPoint(400, 300), Qt.DropAction.CopyAction, data,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    win.dragEnterEvent(enter)
    assert enter.isAccepted(), "drag not accepted"

    # 1. an archive becomes a chapter, pages in natural order
    drop(win, [arc])
    assert win.folder == root / "work" / importer.IMPORTED / "ตอนที่ 7", win.folder
    assert len(win.images) == 3 and win.doc is not None
    assert [p.name.split("_", 1)[1] for p in win.images] == ["p1.png", "p2.png", "p10.png"], [p.name for p in win.images]
    print("1. archive ok:", [p.name for p in win.images])

    # 2. loose images join the open chapter as copies
    drop(win, [loose / "extra_a.png", loose / "extra_b.png"])
    assert len(win.images) == 5, [p.name for p in win.images]
    assert (win.work_dir / importer.ADDED / "extra_a.png").exists() and (loose / "extra_a.png").exists()
    print("2. images added ok:", len(win.images), "pages")

    # 3. a folder opens as its own chapter (source folder untouched)
    before = sorted(p.name for p in src.iterdir())
    drop(win, [src])
    assert win.folder == src and len(win.images) == 3
    assert sorted(p.name for p in src.iterdir()) == before
    print("3. folder ok")

    # 4. something else is refused with a message
    drop(win, [root / "notes.txt"])
    app.processEvents()
    assert any("เปิดไฟล์นี้ไม่ได้" in b or "ลากโฟลเดอร์" in b for b in boxes), boxes
    print("4. unsupported file refused ok")

    win.grab().save(str(root / "after_drop.png"))
    win.close()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
