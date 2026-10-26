"""For UI test scripts: point settings at a temporary copy so the user's settings.json is never written."""
import shutil
import tempfile
from pathlib import Path

import manga_translator.core.settings as settings_mod
from manga_translator.core.paths import SETTINGS_PATH


def isolate_settings() -> Path:
    tmp = Path(tempfile.mkdtemp()) / "settings.json"
    if SETTINGS_PATH.exists():
        shutil.copyfile(SETTINGS_PATH, tmp)  # start from the user's choices (model, languages)
    settings_mod.SETTINGS_PATH = tmp
    return tmp
