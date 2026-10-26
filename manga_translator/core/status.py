"""System status for the info page: Ollama, GPU, model files, loaded models, versions.

Every probe is quick and never raises: a missing tool or a stopped server is reported as data.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata

import requests

from .paths import MODELS_DIR

# files the pipeline needs, with what each is for (shown on the info page)
MODEL_FILES = [
    ("comictextdetector.pt.onnx", "ตรวจจับข้อความ (comic-text-detector)"),
    ("manga-ocr-base", "อ่านข้อความญี่ปุ่น (manga-ocr)"),
    ("en_PP-OCRv3_rec_infer.onnx", "อ่านข้อความอังกฤษ (PP-OCRv3 English)"),
    ("big-lama.pt", "ลบคำบนพื้นที่มีลาย (LaMa)"),
]
PACKAGES = ["PySide6", "torch", "onnxruntime", "rapidocr-onnxruntime", "manga-ocr", "pythainlp", "opencv-python-headless"]


@dataclass
class OllamaStatus:
    ok: bool
    version: str = ""
    error: str = ""
    installed: list[dict] = field(default_factory=list)  # name, size_gb
    running: list[dict] = field(default_factory=list)  # name, vram_gb, expires


@dataclass
class GpuStatus:
    ok: bool
    name: str = ""
    used_mb: int = 0
    total_mb: int = 0
    util: int = 0
    temp: int = 0
    error: str = ""


def ollama_status(url: str, timeout: float = 1.5) -> OllamaStatus:
    try:
        version = requests.get(f"{url}/api/version", timeout=timeout).json().get("version", "")
        tags = requests.get(f"{url}/api/tags", timeout=timeout).json().get("models", [])
        ps = requests.get(f"{url}/api/ps", timeout=timeout).json().get("models", [])
    except (requests.RequestException, ValueError) as e:
        return OllamaStatus(ok=False, error=type(e).__name__)
    return OllamaStatus(
        ok=True,
        version=version,
        installed=[{"name": m["name"], "size_gb": m.get("size", 0) / 1e9} for m in tags],
        running=[{"name": m["name"], "vram_gb": m.get("size_vram", 0) / 1e9, "expires": m.get("expires_at", "")}
                 for m in ps],
    )


def gpu_status() -> GpuStatus:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console flash under pythonw
        )
        name, used, total, util, temp = [p.strip() for p in out.stdout.splitlines()[0].split(",")]
        return GpuStatus(True, name, int(used), int(total), int(util), int(temp))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as e:
        return GpuStatus(ok=False, error=type(e).__name__)


def _size(path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def model_files() -> list[dict]:
    out = []
    for name, purpose in MODEL_FILES:
        path = MODELS_DIR / name
        out.append({"name": name, "purpose": purpose, "present": path.exists(),
                    "size_mb": _size(path) / 1e6 if path.exists() else 0.0})
    return out


def versions() -> dict[str, str]:
    v = {"Python": platform.python_version(), "Windows": platform.version()}
    for pkg in PACKAGES:
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = "ไม่ได้ติดตั้ง"
    torch = sys.modules.get("torch")  # report CUDA only if torch is already loaded; importing it is slow
    if torch is not None:
        v["CUDA (torch)"] = f"{torch.version.cuda} ({'ใช้ได้' if torch.cuda.is_available() else 'ใช้ไม่ได้'})"
    return v
