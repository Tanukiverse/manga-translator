"""Runs pipeline steps off the UI thread."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal

from ..core.pipeline import RUNNERS, STEP_LABELS, Models
from ..core.project import PageDoc


class StepWorker(QObject):
    progress = Signal(str)
    step_done = Signal(str)
    finished = Signal(bool, str)  # ok, error message

    def __init__(self, models: Models, docs: list[PageDoc], steps: list[str]):
        super().__init__()
        self.models, self.docs, self.steps = models, docs, steps
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            for doc in self.docs:
                for step in self.steps:
                    if self._cancel:
                        self.finished.emit(False, "ยกเลิกแล้ว")
                        return
                    self.progress.emit(f"{doc.folder.name}: {STEP_LABELS[step]} …")
                    RUNNERS[step](doc, self.models, lambda m, d=doc: self.progress.emit(f"{d.folder.name}: {m}"))
                    self.step_done.emit(step)
            self.finished.emit(True, "")
        except Exception as e:  # surface every failure in the UI instead of crashing
            traceback.print_exc()
            self.finished.emit(False, str(e))


def start_worker(parent, worker: StepWorker) -> QThread:
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
