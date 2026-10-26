"""Command line: python -m manga_translator.cli page.png [--from ja] [--to th] [--steps detect,ocr]"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .core.pipeline import RUNNERS, STEPS, Models
from .core.project import PageDoc
from .core.settings import Settings


def main() -> int:
    ap = argparse.ArgumentParser(description="Translate a manga page")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--from", dest="src", default=None, help="source language (ja/en)")
    ap.add_argument("--to", dest="tgt", default=None, help="target language (th/en/...)")
    ap.add_argument("--work", type=Path, default=None, help="work folder (default from settings)")
    ap.add_argument("--steps", default=",".join(STEPS), help="comma separated steps to run")
    ap.add_argument("--model", default=None, help="Ollama model")
    args = ap.parse_args()

    settings = Settings.load()
    if args.model:
        settings.ollama_model = args.model
    work = args.work or Path(settings.work_dir)
    models = Models(settings)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for img in args.images:
        doc = PageDoc.create(img, work, args.src or settings.source_lang, args.tgt or settings.target_lang)
        if args.src:
            doc.source_lang = args.src
        if args.tgt:
            doc.target_lang = args.tgt
        print(f"== {img.name} -> {doc.folder}")
        for step in steps:
            t = time.perf_counter()
            RUNNERS[step](doc, models, lambda m: print("  ", m))
            print(f"  {step}: {time.perf_counter() - t:.1f}s")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"peak VRAM (torch): {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
