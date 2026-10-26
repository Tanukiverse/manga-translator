"""One long-lived AI thread that runs every pipeline step, so all models load and run on it.

Connect the runner's signals only to bound methods of QObjects (e.g. the main window), never
to lambdas: a lambda has no receiver, so Qt runs it on the AI thread, and touching widgets
there crashed the app inside Qt6Core.dll.
"""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..core import versions
from ..core.pipeline import RUNNERS, STEP_LABELS, Models


class StepRunner(QObject):
    progress = Signal(str)
    step_done = Signal(str, str)  # step, page folder
    page_started = Signal(int, int, str)  # index (1-based), total, page name
    page_failed = Signal(str, str)  # page name, error
    page_saved = Signal(str, str)  # page name, file name written by auto-save
    finished = Signal(bool, str)  # ok (nothing failed or cancelled), message

    def __init__(self, models: Models):
        super().__init__()
        self.models = models
        self._cancel = False
        # set from the UI thread before submit: called after "typeset" with the page's doc,
        # returns the auto-saved file name or None. Must not touch widgets.
        self.save_hook = None

    def cancel(self):
        self._cancel = True  # read by the AI thread between steps

    @Slot(list, list)
    def run(self, docs: list, steps: list):
        """Run steps on each page in order. A failing page is reported and skipped; the rest go on."""
        self._cancel = False
        failed = []
        what = "ทำทั้งหมด" if list(steps) == list(RUNNERS) else ", ".join(STEP_LABELS[s] for s in steps)
        for n, doc in enumerate(docs, 1):
            name = doc.folder.name
            self.page_started.emit(n, len(docs), name)
            # keep the page as it was, so a worse new result can be undone from its history;
            # a failed snapshot (disk full, locked file) must not stop the translation itself
            try:
                keep = getattr(getattr(self.models, "settings", None), "versions_keep", versions.DEFAULT_KEEP)
                versions.snapshot(doc, f"ก่อน: {what}", keep)
            except Exception:
                traceback.print_exc()
            try:
                for step in steps:
                    if self._cancel:
                        self.finished.emit(False, "ยกเลิกแล้ว")
                        return
                    prefix = f"หน้า {n}/{len(docs)} · {name}" if len(docs) > 1 else name
                    self.progress.emit(f"{prefix}: {STEP_LABELS[step]} …")
                    RUNNERS[step](doc, self.models, lambda m, p=prefix: self.progress.emit(f"{p}: {m}"))
                    if step == "typeset" and self.save_hook is not None:
                        saved = self.save_hook(doc)
                        if saved:
                            self.page_saved.emit(name, saved)
                    self.step_done.emit(step, str(doc.folder))
            except Exception as e:  # report and move on to the next page
                traceback.print_exc()
                failed.append(name)
                self.page_failed.emit(name, str(e) or type(e).__name__)
        if failed:
            self.finished.emit(False, f"เสร็จ {len(docs) - len(failed)}/{len(docs)} หน้า · พัง {len(failed)} หน้า")
        else:
            self.finished.emit(True, f"เสร็จ {len(docs)} หน้า" if len(docs) > 1 else "")

    @Slot(object)
    def call(self, fn):
        """Run a plain function on the AI thread (e.g. re-saving pages after a reorder)."""
        try:
            fn()
        except Exception:
            traceback.print_exc()


class AIWorker(QObject):
    _submit = Signal(list, list)
    _call = Signal(object)

    def __init__(self, models: Models):
        super().__init__()
        self.thread = QThread()
        self.thread.setObjectName("ai")
        self.runner = StepRunner(models)
        self.runner.moveToThread(self.thread)
        self._submit.connect(self.runner.run)  # queued: runs on the AI thread
        self._call.connect(self.runner.call)
        self.thread.start()

    def submit(self, docs: list, steps: list):
        self._submit.emit(docs, steps)

    def run_call(self, fn):
        self._call.emit(fn)

    def cancel(self):
        self.runner.cancel()

    def stop(self, timeout_ms: int = 5000):
        self.runner.cancel()
        self.thread.quit()
        self.thread.wait(timeout_ms)
