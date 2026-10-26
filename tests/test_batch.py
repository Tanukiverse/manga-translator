from pathlib import Path
from types import SimpleNamespace

from manga_translator.core.typeset import ensure_qt
from manga_translator.ui import worker


def _runner(monkeypatch, fail_on: str):
    ensure_qt()
    calls = []

    def step(doc, models, progress):
        calls.append(doc.folder.name)
        if doc.folder.name == fail_on:
            raise RuntimeError("Ollama หลุด")

    monkeypatch.setattr(worker, "RUNNERS", {"detect": step, "typeset": step})
    monkeypatch.setattr(worker, "STEP_LABELS", {"detect": "ตรวจจับ", "typeset": "ใส่คำแปล"})
    runner = worker.StepRunner(models=None)
    events = {"started": [], "failed": [], "finished": [], "saved": []}
    runner.page_started.connect(lambda n, t, name: events["started"].append((n, t, name)))
    runner.page_failed.connect(lambda name, err: events["failed"].append((name, err)))
    runner.finished.connect(lambda ok, msg: events["finished"].append((ok, msg)))
    runner.page_saved.connect(lambda name, f: events["saved"].append((name, f)))
    return runner, calls, events


def _docs(*names):
    return [SimpleNamespace(folder=Path("work") / n, final_path=Path("x")) for n in names]


def test_failing_page_is_skipped_and_reported(monkeypatch):
    runner, calls, events = _runner(monkeypatch, fail_on="p2")
    runner.run(_docs("p1", "p2", "p3"), ["detect", "typeset"])
    assert calls == ["p1", "p1", "p2", "p3", "p3"]  # p2 stops after its failing step, p3 still runs
    assert [s[0] for s in events["started"]] == [1, 2, 3]
    assert events["failed"] == [("p2", "Ollama หลุด")]
    ok, msg = events["finished"][0]
    assert not ok and "2/3" in msg


def test_save_hook_runs_after_typeset(monkeypatch):
    runner, _calls, events = _runner(monkeypatch, fail_on="none")
    runner.save_hook = lambda doc: f"{doc.folder.name}.png"
    runner.run(_docs("p1", "p2"), ["detect", "typeset"])
    assert events["saved"] == [("p1", "p1.png"), ("p2", "p2.png")]
    assert events["finished"] == [(True, "เสร็จ 2 หน้า")]


def test_cancel_stops_between_steps(monkeypatch):
    runner, calls, events = _runner(monkeypatch, fail_on="none")
    worker.RUNNERS["detect"] = lambda doc, m, p: (calls.append(doc.folder.name), runner.cancel())
    runner.run(_docs("p1", "p2"), ["detect", "typeset"])
    assert calls == ["p1"] and events["finished"] == [(False, "ยกเลิกแล้ว")]
