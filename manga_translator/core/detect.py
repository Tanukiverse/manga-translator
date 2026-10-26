"""Text detection with comic-text-detector (ONNX, CPU via onnxruntime).

The model returns YOLO text-block boxes plus a per-pixel text segmentation map.
We keep the boxes as editable regions and the segmentation (limited to the boxes)
as the removal mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .paths import MODELS_DIR

INPUT_SIZE = 1024
CONF_THRESH = 0.25  # 0.4 missed a real "KOU!" bubble (score 0.31); 0.2 added nothing false on that page
NMS_THRESH = 0.35
MASK_THRESH = 0.3


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float


class TextDetector:
    def __init__(self, model_path: Path | None = None):
        import onnxruntime as ort

        path = model_path or MODELS_DIR / "comictextdetector.pt.onnx"
        if not Path(path).exists():
            raise FileNotFoundError(f"missing model {path} - run tools/download_models.py")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, img_bgr: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """Return (detections, mask) where mask is uint8 0/255 at page resolution."""
        h, w = img_bgr.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        canvas[:nh, :nw] = resized  # pad right/bottom
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self.session.run(None, {self.input_name: blob})
        blks = seg = None
        for out in outputs:
            if out.ndim == 3:
                blks = out
            elif out.ndim == 4 and out.shape[1] == 1:
                seg = out
        if blks is None or seg is None:
            raise RuntimeError(f"unexpected detector outputs: {[o.shape for o in outputs]}")

        dets = resolve_overlaps(self._boxes(blks[0], scale, w, h))
        prob = seg[0, 0, :nh, :nw]
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = build_mask(prob, dets)
        return dets, mask

    @staticmethod
    def _boxes(pred: np.ndarray, scale: float, w: int, h: int) -> list[Detection]:
        conf = pred[:, 4] * pred[:, 5:].max(axis=1)
        keep = conf > CONF_THRESH
        pred, conf = pred[keep], conf[keep]
        if len(pred) == 0:
            return []
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1) / scale
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), CONF_THRESH, NMS_THRESH)
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw_, bh_ = boxes[i]
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(np.ceil(x + bw_))), min(h, int(np.ceil(y + bh_)))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                dets.append(Detection((x0, y0, x1 - x0, y1 - y0), float(conf[i])))
        return dets


def resolve_overlaps(dets: list[Detection], inside: float = 0.6) -> list[Detection]:
    """A box that spans two neighbouring bubbles often comes with a second box for one of them,
    and both translations would then be drawn on top of each other. When a smaller box lies
    mostly inside a bigger one, cut the bigger box back to the side the smaller one is not on;
    if the smaller one sits in the middle, it is a duplicate and is dropped."""
    out = sorted(dets, key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)  # biggest first
    i = 0
    while i < len(out):
        j = i + 1
        while j < len(out):
            bx, by, bw, bh = out[i].bbox
            sx, sy, sw, sh = out[j].bbox
            ix = max(0, min(bx + bw, sx + sw) - max(bx, sx))
            iy = max(0, min(by + bh, sy + sh) - max(by, sy))
            if ix * iy < inside * sw * sh:
                j += 1
                continue
            # free strip beside the small box on each side, as a share of the big box's size
            sides = {
                "left": (sx - bx) / bw, "right": (bx + bw - sx - sw) / bw,
                "top": (sy - by) / bh, "bottom": (by + bh - sy - sh) / bh,
            }
            side = max(sides, key=sides.get)
            if sides[side] < 0.3:
                del out[j]  # it sits in the middle: a duplicate of the big box
                continue
            trimmed = {
                "left": (bx, by, sx - bx, bh), "right": (sx + sw, by, bx + bw - sx - sw, bh),
                "top": (bx, by, bw, sy - by), "bottom": (bx, sy + sh, bw, by + bh - sy - sh),
            }[side]
            out[i] = Detection(trimmed, out[i].score)
            j += 1
        i += 1
    return [d for d in out if d.bbox[2] >= 4 and d.bbox[3] >= 4]


def _flag_runs(flags, glue: int = 0) -> list[tuple[int, int]]:
    """[start, end) runs of True values; runs separated by at most `glue` False values are joined
    (faint lettering breaks into slivers that would otherwise read as tiny lines)."""
    runs, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    joined: list[tuple[int, int]] = []
    for a, b in runs:
        if joined and a - joined[-1][1] <= glue:
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    return joined


def split_side_by_side(bbox, mask: np.ndarray, gray: np.ndarray | None = None) -> list[tuple[int, int, int, int]]:
    """Split a box holding two horizontal text blocks next to each other (two speech beats in one
    bubble, or two neighbouring bubbles) at the empty column between them. Only for horizontal
    text: vertical Japanese columns are separated by gaps by design."""
    x, y, w, h = bbox
    ink = mask[y:y + h, x:x + w] > 0
    if gray is not None:
        # the mask is dilated, which joins lines and narrows gaps: measure the dark lettering itself
        ink &= gray[y:y + h, x:x + w] < 140
    if not ink.any():
        return [tuple(bbox)]
    pieces = _flag_runs(ink.any(0))  # column runs between any empty columns (words, blocks)
    # line height measured inside each piece: blocks starting at different heights would
    # otherwise merge into one tall run of rows
    heights = [b - a for c0, c1 in pieces for a, b in _flag_runs(ink[:, c0:c1].any(1), glue=3)]
    # upper quartile, not median: in mixed-case fonts most letters are only x-height tall
    line_h = sorted(heights)[(3 * len(heights)) // 4]
    # word spaces are well under half a line high; a wider empty column separates blocks
    min_gap = max(8, int(0.6 * line_h))
    blocks = [list(pieces[0])]
    for a, b in pieces[1:]:
        if a - blocks[-1][1] < min_gap:
            blocks[-1][1] = b
        else:
            blocks.append([a, b])
    if len(blocks) < 2 or any(b - a < 2 * line_h for a, b in blocks):
        return [tuple(bbox)]
    # one paragraph whose word spaces happen to line up: every line of both parts sits at the
    # same height. Separate speech beats start at different heights or have other line counts.
    def lines(a, b):  # rows of ink, with slivers (accents, faint strokes) glued to their line
        out = []
        for r0, r1 in _flag_runs(ink[:, a:b].any(1)):
            if out and r0 - out[-1][1] <= max(3, 0.25 * line_h):
                out[-1] = (out[-1][0], r1)
            else:
                out.append((r0, r1))
        return out

    rows = [lines(a, b) for a, b in blocks]
    tol = 0.4 * line_h
    if all(len(r) == len(rows[0]) for r in rows) and all(
            abs(r[k][0] - rows[0][k][0]) <= tol and abs(r[k][1] - rows[0][k][1]) <= tol
            for r in rows[1:] for k in range(len(r))):
        return [tuple(bbox)]
    out = []
    for a, b in blocks:
        ys = np.nonzero(ink[:, a:b].any(1))[0]
        out.append((x + a, y + int(ys.min()), b - a, int(ys.max()) + 1 - int(ys.min())))
    return out


def is_light_text(gray: np.ndarray, mask: np.ndarray, bbox) -> bool:
    """White lettering on a dark background (captions over hair, night scenes, black boxes):
    the text pixels are clearly brighter than the pixels just around them, which are dark."""
    x, y, w, h = bbox
    m = mask[y:y + h, x:x + w] > 0
    if m.sum() < 20:
        return False
    g = gray[y:y + h, x:x + w]
    ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
    if ring.sum() < 20:
        return False
    text, around = float(g[m].mean()), float(g[ring].mean())
    return around < 110 and text > around + 40


def grow_mask_in(mask: np.ndarray, bbox, px: int, pad: int = 8) -> None:
    """Dilate the mask by `px` inside one box (in place). Light lettering has a bright halo past
    the detected strokes; left in, LaMa smears it into a white blotch over the art."""
    x, y, w, h = bbox
    sl = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    mask[sl] = cv2.dilate(mask[sl], k)


def build_mask(prob: np.ndarray, dets: list[Detection], pad: int = 4, dilate: int = 3) -> np.ndarray:
    """Text mask from the segmentation map, kept only inside (slightly padded) detected boxes."""
    h, w = prob.shape
    inside = np.zeros((h, w), np.uint8)
    for d in dets:
        x, y, bw, bh = d.bbox
        inside[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad] = 1
    mask = ((prob > MASK_THRESH) & (inside > 0)).astype(np.uint8) * 255
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, k)
    return mask


def sort_reading_order(boxes: list[tuple[int, int, int, int]], right_to_left: bool) -> list[int]:
    """Indices of boxes in reading order: rows top to bottom, then by x (RTL for manga)."""
    if not boxes:
        return []
    heights = sorted(b[3] for b in boxes)
    row_tol = max(8, heights[len(heights) // 2] * 0.5)
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1])
    rows: list[list[int]] = []
    for i in order:
        if rows and abs(boxes[i][1] - boxes[rows[-1][0]][1]) <= row_tol:
            rows[-1].append(i)
        else:
            rows.append([i])
    result = []
    for row in rows:
        row.sort(key=lambda i: boxes[i][0] + boxes[i][2] / 2, reverse=right_to_left)
        result.extend(row)
    return result
