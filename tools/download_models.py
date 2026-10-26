"""Download the AI models into models/ and record their SHA-256 in models/manifest.json.

The first download pins the hash; later runs verify files against it.
Run:  .venv\\Scripts\\python tools\\download_models.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MANIFEST = MODELS / "manifest.json"

FILES = {
    "comictextdetector.pt.onnx": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
    "big-lama.pt": "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
}
HF_REPOS = {"manga-ocr-base": "kha-white/manga-ocr-base"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text("utf-8")) if MANIFEST.exists() else {}
    ok = True
    for name, url in FILES.items():
        dest = MODELS / name
        print(name)
        if not dest.exists():
            download(url, dest)
        digest = sha256(dest)
        if name in manifest and manifest[name] != digest:
            print(f"  HASH MISMATCH: expected {manifest[name]}, got {digest}")
            ok = False
            continue
        manifest[name] = digest
        print(f"  ok {digest[:16]}…")

    from huggingface_hub import snapshot_download

    for name, repo in HF_REPOS.items():
        print(name)
        snapshot_download(repo, local_dir=MODELS / name)
        print("  ok")

    MANIFEST.write_text(json.dumps(manifest, indent=2), "utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
