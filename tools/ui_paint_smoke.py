"""Colour brush in the real window: paint with the mouse, pick a colour, undo, wipe paint off.

Runs the full pipeline on one page first, so the paint is tested on a finished page.
Usage: .venv\\Scripts\\python tools\\ui_paint_smoke.py <folder with page images> <scratch dir>
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
from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from manga_translator.core.pipeline import STEPS  # noqa: E402
from manga_translator.ui.canvas import MODE_COLOR, MODE_UNCOLOR  # noqa: E402
from manga_translator.ui.main_window import MainWindow  # noqa: E402


def wait(win):
    loop = QEventLoop()
    t = QTimer()
    t.timeout.connect(lambda: (not win.busy) and loop.quit())
    t.start(100)
    loop.exec()


def view_point(win, x, y) -> QPoint:
    return win.canvas.mapFromScene(x, y)


def drag(win, a, b, modifier=Qt.KeyboardModifier.NoModifier):
    vp = win.canvas.viewport()
    pa, pb = view_point(win, *a), view_point(win, *b)
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, modifier, pa)
    for i in range(1, 11):
        QTest.mouseMove(vp, pa + (pb - pa) * i / 10)
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, modifier, pb)
    QTest.qWait(50)


def main() -> int:
    src, work = Path(sys.argv[1]), Path(sys.argv[2])
    shutil.rmtree(work, ignore_errors=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.settings.work_dir = str(work)
    win.resize(1500, 950)
    win.show()
    win.open_folder(str(src))
    QTest.qWait(200)
    win.run_steps(STEPS)
    wait(win)
    doc = win.doc
    assert doc.final_path.exists()
    win.canvas.fit()
    QTest.qWait(100)

    # the colour tool opens on the finished page with its colour box
    win.set_mode(MODE_COLOR)
    assert win.layer == "final" and win.color_box.isVisible() and win.brush_box.isVisible()
    h, w = doc.load_original().shape[:2]
    win.brush.setValue(20)
    win._set_paint_color(QColor("#ff0000"))
    y = int(h * 0.97)  # a strip at the bottom, below the text
    drag(win, (w * 0.1, y), (w * 0.9, y))
    final = doc.load_layer("final")
    assert tuple(final[y, w // 2]) == (0, 0, 255), final[y, w // 2]
    assert doc.paint_path.exists()
    print("paint ok")

    # Alt+click takes the colour under the cursor (the red just painted)
    win._set_paint_color(QColor("#00ff00"))
    vp = win.canvas.viewport()
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, view_point(win, w // 2, y))
    assert win.settings.paint_color == "#ff0000", win.settings.paint_color
    print("pick ok")

    # undo takes the stroke away, redo brings it back
    win.undo()
    assert not doc.paint_path.exists() and tuple(doc.load_layer("final")[y, w // 2]) != (0, 0, 255)
    win.redo()
    assert tuple(doc.load_layer("final")[y, w // 2]) == (0, 0, 255)
    print("undo/redo ok")

    # re-running the steps keeps the paint
    win.run_steps(["clean", "typeset"])
    wait(win)
    assert tuple(win.doc.load_layer("final")[y, w // 2]) == (0, 0, 255)
    print("re-run keeps paint ok")

    win.grab().save(str(work / "after_paint.png"))
    # wiping the paint off restores the page under it
    win.set_mode(MODE_UNCOLOR)
    win.brush.setValue(32)  # wider than the stroke, so its soft edge goes too
    drag(win, (w * 0.05, y), (w * 0.95, y))
    assert not win.doc.paint_path.exists()
    assert tuple(win.doc.load_layer("final")[y, w // 2]) != (0, 0, 255)
    print("uncolor ok")
    win.close()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
