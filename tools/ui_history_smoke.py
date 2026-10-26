"""Real window: run a page, re-run it, restore the first result from history, reset, restore again.

Usage: .venv\\Scripts\\python tools\\ui_history_smoke.py <folder with page images> <scratch dir>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manga_translator import crashlog  # noqa: E402

crashlog.install()
from _isolate import isolate_settings  # noqa: E402

isolate_settings()
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from manga_translator.core import versions  # noqa: E402
from manga_translator.core.pipeline import STEPS  # noqa: E402
from manga_translator.ui.history_dialog import HistoryDialog  # noqa: E402
from manga_translator.ui.main_window import MainWindow  # noqa: E402


def answer_yes():
    w = QApplication.activeModalWidget()
    if isinstance(w, QMessageBox):
        yes = w.button(QMessageBox.StandardButton.Yes)
        (yes or w.buttons()[0]).click()


def wait(win):
    loop = QEventLoop()
    t = QTimer()
    t.timeout.connect(lambda: (not win.busy) and loop.quit())
    t.start(100)
    loop.exec()


def state(doc):
    return [(r.id, r.translation) for r in doc.regions], doc.final_path.read_bytes() if doc.final_path.exists() else None


def main() -> int:
    src, work = Path(sys.argv[1]), Path(sys.argv[2])
    shutil.rmtree(work, ignore_errors=True)
    app = QApplication(sys.argv)
    t = QTimer()
    t.timeout.connect(answer_yes)
    t.start(150)
    win = MainWindow()
    win.settings.work_dir = str(work)
    win.src_combo.setCurrentIndex(win.src_combo.findData("en"))
    win.show()
    win.open_folder(str(src))
    win.pages.setCurrentRow(win.images.index(next(p for p in win.images if p.stem == "002")))
    app.processEvents()

    win.run_steps(STEPS)
    wait(win)
    first = state(win.doc)
    assert first[1] is not None and versions.list_versions(win.doc) == []  # nothing to keep before the first run

    # a second run of two steps: the first result is kept in the history
    win.doc.regions[0].translation = "แก้ด้วยมือก่อนรันซ้ำ"
    win.doc.regions[0].locked = False
    win.doc.save()
    win.run_steps(["translate", "typeset"])
    wait(win)
    hist = versions.list_versions(win.doc)
    assert len(hist) == 1 and hist[0].label.startswith("ก่อน: แปล"), [v.label for v in hist]

    # restore it through the history window, as a user would
    dlg = HistoryDialog(win, win.doc)
    dlg.list.setCurrentRow(0)
    QTimer.singleShot(0, dlg._restore)
    dlg.exec()
    assert dlg.restored is not None
    win._reload_page(win._current_path, dlg.restored)
    restored = state(win.doc)
    assert restored[0][0][1] == "แก้ด้วยมือก่อนรันซ้ำ", restored[0][:1]  # the state right before the re-run
    print("history restore ok:", len(versions.list_versions(win.doc)), "versions")

    # reset to the original, then bring the page back from the history
    before_reset = state(win.doc)
    win.reset_pages()
    app.processEvents()
    assert win.doc.regions == [] and not win.doc.final_path.exists() and win.doc.original_path.exists()
    assert win.layer == "original"
    newest = versions.list_versions(win.doc)[0]
    assert newest.label == "ก่อนรีเซ็ต"
    win._reload_page(win._current_path, versions.restore(win.doc, newest))
    assert state(win.doc) == before_reset
    print("reset + restore ok")

    win.grab().save(str(work / "after_history.png"))
    win.close()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
