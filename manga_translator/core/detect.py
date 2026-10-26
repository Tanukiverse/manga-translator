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
CONF_THRESH = 0.4
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

        dets = self._boxes(blks[0], scale, w, h)
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
