"""Drive the real window: sorting, drag order, batch run that survives a failing page,
auto-save (incl. re-numbering after a reorder) and the multi-page export dialog.

Usage: .venv\\Scripts\\python tools\\ui_batch_smoke.py <scratch dir>
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from manga_translator import crashlog  # noqa: E402

crashlog.install()
from _isolate import isolate_settings  # noqa: E402

isolate_settings()
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import synth  # noqa: E402
from manga_translator.core.project import imread, imwrite  # noqa: E402
from manga_translator.ui.main_window import MainWindow  # noqa: E402

boxes: list[str] = []


def auto_close_message_boxes():
    """Answer modal message boxes the way a user would ("ปิด"), recording their text."""
    w = QApplication.activeModalWidget()
    if isinstance(w, QMessageBox):
        boxes.append(w.text())
        for b in w.buttons():
            if b.text() == "ปิด":
                b.click()
                return
        w.reject()


def wait(cond, timeout=600):
    loop = QEventLoop()
    t = QTimer()
    t.timeout.connect(lambda: cond() and loop.quit())
    t.start(100)
    QTimer.singleShot(int(timeout * 1000), loop.quit)
    loop.exec()
    t.stop()
    return cond()


def pause(seconds):
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def names(win):
    return [p.name for p in win.images]


def main() -> int:
    root = Path(sys.argv[1])
    shutil.rmtree(root, ignore_errors=True)
    src = root / "ตอนทดสอบ"
    src.mkdir(parents=True)
    page, _ = synth.english_page()
    for name, scale in [("p1.png", 1.0), ("p2.png", 0.8), ("p10.png", 1.2)]:
        imwrite(src / name, cv2.resize(page, None, fx=scale, fy=scale))
    (src / "p3.png").write_bytes(b"not an image")
    out, export_to = root / "out", root / "export"

    app = QApplication(sys.argv)
    app.setFont(QFont("Leelawadee UI", 10))
    closer = QTimer()
    closer.timeout.connect(auto_close_message_boxes)
    closer.start(200)

    win = MainWindow()
    s = win.settings
    s.work_dir, s.source_lang, s.target_lang = str(root / "work"), "en", "th"
    s.page_sort, s.page_sort_desc = "name", False
    s.autosave, s.save_dir, s.save_prefix, s.save_format = True, str(out), "{ตอน}_", "png"
    win.sort_combo.setCurrentIndex(0)
    win.sort_dir.setChecked(False)
    win.src_combo.setCurrentIndex(win.src_combo.findData("en"))
    win.resize(1500, 950)
    win.show()
    win.open_folder(str(src))

    # 1. natural name order, then size (largest first), then a drag-style move
    assert names(win) == ["p1.png", "p2.png", "p3.png", "p10.png"], names(win)
    win.sort_combo.setCurrentIndex(win.sort_combo.findData("size"))
    win.sort_dir.setChecked(True)
    by_size = sorted(win.images, key=lambda p: p.stat().st_size, reverse=True)
    assert win.images == by_size, names(win)
    win.sort_combo.setCurrentIndex(win.sort_combo.findData("name"))
    win.sort_dir.setChecked(False)
    win.pages.model().moveRow(win.pages.rootIndex(), 3, win.pages.rootIndex(), 0)  # what a drag does
    app.processEvents()
    assert names(win) == ["p10.png", "p1.png", "p2.png", "p3.png"], names(win)
    assert win.sort_combo.currentData() == "manual"
    assert (win.work_dir / "order.json").exists()
    win.sort_combo.setCurrentIndex(win.sort_combo.findData("name"))
    print("1. sorting ok:", names(win))

    # 2. batch run: p2's work copy is removed so its first step fails; p3 cannot be opened at all
    win.pages.setCurrentRow(1)  # open p2 so its work folder exists
    app.processEvents()
    (win.work_dir / "p2" / "original.png").unlink()
    win.pages.selectAll()
    win.run_selected()
    assert wait(lambda: not win.busy), "batch did not finish"
    assert set(win.failed) == {"p2", "p3"}, win.failed
    for stem in ("p1", "p10"):
        assert (win.work_dir / stem / "final.png").exists(), stem
    assert any("ทำไม่สำเร็จ" in b or "พัง" in b for b in boxes), boxes
    print("2. batch ok: failed =", sorted(win.failed))

    # 3. auto-save: numbers follow the list order (p1=1st, p10=4th)
    chapter = out / "ตอนทดสอบ"
    assert sorted(p.name for p in chapter.iterdir()) == ["ตอนทดสอบ_001.png", "ตอนทดสอบ_004.png"], list(chapter.iterdir())
    # edit p1's translation: the saved file follows after the debounce
    win.pages.clearSelection()
    win.pages.setCurrentRow(0)
    app.processEvents()
    before = (chapter / "ตอนทดสอบ_001.png").stat().st_mtime_ns
    r = win.doc.regions[0]
    win.panel.select(r.id)
    win.panel.translation.setPlainText("แก้แล้วเซฟเอง")
    win.panel._commit_pending()
    assert wait(lambda: (chapter / "ตอนทดสอบ_001.png").stat().st_mtime_ns != before, 10), "auto-save after edit"
    # reorder: p10 to the top -> p10 becomes 001, p1 becomes 002, old 004 removed
    win.pages.model().moveRow(win.pages.rootIndex(), 3, win.pages.rootIndex(), 0)
    assert wait(lambda: sorted(p.name for p in chapter.iterdir()) == ["ตอนทดสอบ_001.png", "ตอนทดสอบ_002.png"], 30), \
        list(chapter.iterdir())
    p10 = imread(win.work_dir / "p10" / "final.png")
    assert imread(chapter / "ตอนทดสอบ_001.png").shape == p10.shape  # p10 (larger image) is now 001
    print("3. auto-save ok:", sorted(p.name for p in chapter.iterdir()))

    # 4. export dialog: JPG, own prefix, numbers 1..n over the saved pages only
    from manga_translator.ui.export_dialog import ExportDialog

    dlg = ExportDialog(win)
    dlg.prefix.setText("ทดสอบ_")
    dlg.renumber.setChecked(True)
    dlg.fmt.setCurrentIndex(dlg.fmt.findData("jpg"))
    dlg.folder.setText(str(export_to))
    dlg.show()
    app.processEvents()
    dlg.grab().save(str(root / "export_dialog.png"))
    dlg._save()
    files = sorted(p.name for p in export_to.iterdir())
    assert files == ["ทดสอบ_001.jpg", "ทดสอบ_002.jpg"], files
    assert (export_to / "ทดสอบ_001.jpg").read_bytes()[:2] == b"\xff\xd8"  # really JPEG
    print("4. export ok:", files)

    win.show_layer("final", fit=True)
    app.processEvents()
    win.grab().save(str(root / "main_window.png"))
    win.close()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    t = time.time()
    rc = main()
    print(f"took {time.time() - t:.0f}s")
    raise SystemExit(rc)
