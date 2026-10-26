"""Drive the real window: open a folder, run every step, edit a translation, undo, screenshot.

Usage: .venv\\Scripts\\python tools\\ui_smoke.py <image folder> <work dir> <screenshot.png>
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manga_translator import crashlog  # noqa: E402
crashlog.install()
from _isolate import isolate_settings  # noqa: E402

isolate_settings()  # the window saves settings (sort order, languages): never touch the user's file
from manga_translator.core.pipeline import STEPS  # noqa: E402
from manga_translator.ui.main_window import MainWindow  # noqa: E402


def wait_until(app, cond, timeout_s=600):
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: cond() and loop.quit())
    timer.start(100)
    QTimer.singleShot(int(timeout_s * 1000), loop.quit)
    loop.exec()
    timer.stop()


def main() -> int:
    folder, work, shot = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    app = QApplication(sys.argv)
    app.setFont(QFont("Leelawadee UI", 10))
    win = MainWindow()
    win.settings.work_dir = str(work)  # not saved: keeps the user's settings.json untouched
    win.resize(1500, 950)
    win.show()
    win.open_folder(str(folder))
    assert win.doc is not None, "no page opened"

    errors = []
    win.ai.runner.finished.connect(lambda ok, err: errors.append(err) if not ok else None)
    # one step per click, like a user: separate runs used to crash onnxruntime
    for step in STEPS:
        win.run_steps([step])
        wait_until(app, lambda: not win.busy)
        assert not errors, (step, errors)
    assert not errors, errors
    assert win.doc.final_path.exists(), "final.png missing"
    assert all(win.doc.done.get(s) for s in STEPS), win.doc.done
    print("regions:", [(r.id, r.source_text, r.translation) for r in win.doc.regions])

    # edit the first translation through the panel, like a user would
    first = win.doc.regions[0]
    win.panel.select(first.id)
    win.panel.translation.setPlainText("ทดสอบแก้คำแปล")
    win.panel._commit_pending()
    assert win.doc.region(first.id).translation == "ทดสอบแก้คำแปล"
    assert win.doc.region(first.id).locked
    win.undo()
    assert win.doc.region(first.id).translation != "ทดสอบแก้คำแปล", "undo failed"
    win.redo()
    assert win.doc.region(first.id).translation == "ทดสอบแก้คำแปล", "redo failed"

    # move a translation by dragging its text box with the mouse, like a user would
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from manga_translator.ui.canvas import MODE_TEXT

    win.canvas.fit()
    win.set_mode(MODE_TEXT)
    app.processEvents()
    rid, item = next(iter(win.canvas.items_by_id.items()))
    assert item.kind == "text"
    before = item.bbox()
    vp = win.canvas.viewport()
    start = win.canvas.mapFromScene(item.sceneBoundingRect().center())
    end = start + QPoint(-60, 80)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for i in range(1, 11):
        QTest.mouseMove(vp, start + (end - start) * (i / 10))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    app.processEvents()
    moved = win.doc.region(rid).text_rect
    assert moved and moved[:2] != before[:2], (before, moved)
    print("text moved:", before, "->", moved)
    win.undo()
    assert win.doc.region(rid).text_rect is None, "undo of text move failed"
    win.redo()
    assert win.doc.region(rid).text_rect == moved

    # line breaks: Enter in the translation box, then "break only at Enter"
    from manga_translator.core.typeset import place

    second = win.doc.regions[1]
    win.panel.select(second.id)
    win.panel.translation.setPlainText("ไปกันเถอะ\nทุกคนรออยู่")
    win.panel._commit_pending()
    win.panel.wrap.setCurrentIndex(win.panel.wrap.findData("manual"))
    app.processEvents()
    r2 = win.doc.region(second.id)
    assert r2.wrap == "manual" and win.doc.load_layer("final") is not None
    lines = place(win.doc.load_layer("cleaned"), r2, win.doc.target_lang)[1].lines
    assert lines == ["ไปกันเถอะ", "ทุกคนรออยู่"], lines
    print("manual lines:", lines)

    win.show_layer("final", fit=True)
    app.processEvents()
    win.grab().save(str(shot))
    print("screenshot:", shot)
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
