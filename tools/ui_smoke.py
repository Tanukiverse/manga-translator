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
    win.run_steps(STEPS)
    win.worker.finished.connect(lambda ok, err: errors.append(err) if not ok else None)
    wait_until(app, lambda: win.thread is None)
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

    win.show_layer("final", fit=True)
    app.processEvents()
    win.grab().save(str(shot))
    print("screenshot:", shot)
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
