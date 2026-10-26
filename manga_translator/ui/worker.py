"""One long-lived AI thread that runs every pipeline step, so all models load and run on it.

Connect the runner's signals only to bound methods of QObjects (e.g. the main window), never
to lambdas: a lambda has no receiver, so Qt runs it on the AI thread, and touching widgets
there crashed the app inside Qt6Core.dll.
"""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..core.pipeline import RUNNERS, STEP_LABELS, Models


class StepRunner(QObject):
    progress = Signal(str)
    step_done = Signal(str)
    finished = Signal(bool, str)  # ok, error message

    def __init__(self, models: Models):
        super().__init__()
        self.models = models
        self._cancel = False

    def cancel(self):
        self._cancel = True  # read by the AI thread between steps

    @Slot(list, list)
    def run(self, docs: list, steps: list):
        self._cancel = False
        try:
            for doc in docs:
                for step in steps:
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


class AIWorker(QObject):
    _submit = Signal(list, list)

    def __init__(self, models: Models):
        super().__init__()
        self.thread = QThread()
        self.thread.setObjectName("ai")
        self.runner = StepRunner(models)
        self.runner.moveToThread(self.thread)
        self._submit.connect(self.runner.run)  # queued: runs on the AI thread
        self.thread.start()

    def submit(self, docs: list, steps: list):
        self._submit.emit(docs, steps)

    def cancel(self):
        self.runner.cancel()

    def stop(self, timeout_ms: int = 5000):
        self.runner.cancel()
        self.thread.quit()
        self.thread.wait(timeout_ms)
