"""Write every error and hard crash to logs/app.log.

The launcher uses pythonw.exe, which has no console: sys.stdout/sys.stderr are None, so any
library that prints or logs (tqdm, loguru, traceback) would raise, and a native crash would
vanish without a trace.
"""
from __future__ import annotations

import faulthandler
import sys
import threading
import traceback
from datetime import datetime

from .core.paths import ROOT

LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "app.log"
_log_file = None


def install() -> None:
    global _log_file
    if _log_file is not None:
        return
    LOG_DIR.mkdir(exist_ok=True)
    _log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    _log_file.write(f"\n===== start {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    if sys.stdout is None:
        sys.stdout = _log_file
    if sys.stderr is None:
        sys.stderr = _log_file
    faulthandler.enable(_log_file, all_threads=True)

    def hook(exc_type, exc, tb):
        _log_file.write("".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = hook
    threading.excepthook = lambda args: hook(args.exc_type, args.exc_value, args.exc_traceback)


def log(msg: str) -> None:
    if _log_file is not None:
        _log_file.write(f"{datetime.now():%H:%M:%S} {msg}\n")
