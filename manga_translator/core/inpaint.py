"""Remove text: flat fill for plain bubbles, LaMa (big-lama TorchScript on GPU) for textured backgrounds.

Only pixels inside the mask are ever changed, so the artwork outside the text stays identical.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .paths import MODELS_DIR

FLAT_STD = 10.0  # max colour std of the surrounding ring to treat a blob as a plain bubble
RING = 6
LAMA_MAX_SIDE = 1024
CROP_MARGIN = 48


class Lama:
    def __init__(self, model_path: Path | None = None, device: str | None = None):
        import torch

        path = model_path or MODELS_DIR / "big-lama.pt"
        if not Path(path).exists():
            raise FileNotFoundError(f"missing model {path} - run tools/download_models.py")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(path), map_location=self.device).eval()

    def __call__(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = img_bgr.shape[:2]
        scale = min(1.0, LAMA_MAX_SIDE / max(h, w))
        img, m = img_bgr, mask
        if scale < 1.0:
            size = (round(w * scale), round(h * scale))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
        ph, pw = img.shape[:2]
        H, W = (ph + 7) // 8 * 8, (pw + 7) // 8 * 8
        img = cv2.copyMakeBorder(img, 0, H - ph, 0, W - pw, cv2.BORDER_REFLECT)
        m = cv2.copyMakeBorder(m, 0, H - ph, 0, W - pw, cv2.BORDER_CONSTANT, value=0)

        t_img = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(self.device)
        t_mask = torch.from_numpy((m > 127).astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            out = self.model(t_img, t_mask)
        res = (out[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy())[:ph, :pw, ::-1]
        if scale < 1.0:
            res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(res)

    def release(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _ring(blob: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING + 1, 2 * RING + 1))
    return cv2.dilate(blob, k) & ~blob


def clean(img_bgr: np.ndarray, mask: np.ndarray, lama: Lama | None = None) -> tuple[np.ndarray, dict]:
    """Return (cleaned image, stats). mask: uint8, 255 = remove."""
    out = img_bgr.copy()
    mask = (mask > 127).astype(np.uint8) * 255
    stats = {"flat": 0, "lama": 0, "telea": 0}
    n, labels, boxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    hard = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, _area = boxes[i]
        x0, y0 = max(0, x - RING - 1), max(0, y - RING - 1)
        x1, y1 = min(mask.shape[1], x + w + RING + 1), min(mask.shape[0], y + h + RING + 1)
        blob = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        ring = _ring(blob) & ~mask[y0:y1, x0:x1]
        pixels = img_bgr[y0:y1, x0:x1][ring > 0]
        if len(pixels) >= 8 and pixels.std(axis=0).max() <= FLAT_STD:
            colour = np.median(pixels, axis=0).astype(np.uint8)
            out[y0:y1, x0:x1][blob > 0] = colour
            stats["flat"] += 1
        else:
            hard[y0:y1, x0:x1] |= blob

    if hard.any():
        n2, labels2, boxes2, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
        for i in range(1, n2):
            x, y, w, h, _ = boxes2[i]
            x0, y0 = max(0, x - CROP_MARGIN), max(0, y - CROP_MARGIN)
            x1, y1 = min(mask.shape[1], x + w + CROP_MARGIN), min(mask.shape[0], y + h + CROP_MARGIN)
            # include every hard blob inside the crop so neighbours are not used as context
            m = hard[y0:y1, x0:x1]
            if lama is not None:
                patch = lama(img_bgr[y0:y1, x0:x1], m)
                stats["lama"] += 1
            else:
                patch = cv2.inpaint(img_bgr[y0:y1, x0:x1], m, 5, cv2.INPAINT_TELEA)
                stats["telea"] += 1
            sel = (labels2[y0:y1, x0:x1] == i)
            out[y0:y1, x0:x1][sel] = patch[sel]
    return out, stats
