"""The colour paint layer: what the user paints by hand over the cleaned page.

It is a BGRA picture stored premultiplied (colour already multiplied by alpha). That is what
OpenCV's anti-aliased drawing produces when it blends a stroke into the layer, so painting over
paint and erasing paint both come out right without any extra maths, and laying the layer over
the page is a single "over" step: out = page * (1 - alpha) + colour.
"""
from __future__ import annotations

import cv2
import numpy as np


def blank(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.zeros((h, w, 4), np.uint8)


def stroke(paint: np.ndarray, a, b, radius: int, color_bgr=(0, 0, 0), erase: bool = False) -> None:
    """Draw (or with erase, remove) a round brush stroke from point a to point b, in place."""
    value = (0, 0, 0, 0) if erase else (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]), 255)
    pa, pb = (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))
    thickness = max(1, int(radius) * 2)
    cv2.line(paint, pa, pb, value, thickness, cv2.LINE_AA)
    # a zero-length line can draw nothing at small sizes; a dot makes every click leave a mark
    cv2.circle(paint, pb, max(1, int(radius)), value, -1, cv2.LINE_AA)


def composite(base_bgr: np.ndarray, paint: np.ndarray | None) -> np.ndarray:
    """The page with the paint on top. Returns base unchanged when there is no paint."""
    if paint is None or paint.shape[:2] != base_bgr.shape[:2] or not paint[..., 3].any():
        return base_bgr
    alpha = paint[..., 3:4].astype(np.float32) / 255.0
    out = base_bgr.astype(np.float32) * (1.0 - alpha) + paint[..., :3].astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def pick(img_bgr: np.ndarray, paint: np.ndarray | None, x: int, y: int) -> tuple[int, int, int] | None:
    """The colour (BGR) seen at a point of the page with its paint, or None outside the page."""
    h, w = img_bgr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    px = img_bgr[y:y + 1, x:x + 1]
    if paint is not None and paint.shape[:2] == img_bgr.shape[:2]:
        px = composite(px, paint[y:y + 1, x:x + 1])
    b, g, r = (int(v) for v in px[0, 0])
    return b, g, r
