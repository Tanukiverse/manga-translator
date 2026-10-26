"""Synthetic manga pages with known text boxes, so tests do not depend on copyrighted pages."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

JA_FONT = "C:/Windows/Fonts/YuGothM.ttc"
EN_FONT = "C:/Windows/Fonts/comicbd.ttf"


@dataclass
class TruthBox:
    bbox: tuple[int, int, int, int]
    text: str
    bubble: tuple[int, int, int, int]  # ellipse bounding box


def _screentone(h: int, w: int) -> np.ndarray:
    img = np.full((h, w), 235, np.uint8)
    for y in range(0, h, 6):
        for x in range((y // 6 % 2) * 3, w, 6):
            cv2.circle(img, (x, y), 1, 150, -1)
    # some "artwork" lines
    rng = np.random.default_rng(0)
    for _ in range(40):
        p1 = tuple(int(v) for v in rng.integers(0, [w, h]))
        p2 = tuple(int(v) for v in rng.integers(0, [w, h]))
        cv2.line(img, p1, p2, 40, 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _bubble(img: np.ndarray, cx: int, cy: int, rx: int, ry: int) -> tuple[int, int, int, int]:
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (0, 0, 0), 3, cv2.LINE_AA)
    return (cx - rx, cy - ry, 2 * rx, 2 * ry)


def _ink_bbox(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int, int]:
    diff = np.any(before != after, axis=2)
    ys, xs = np.nonzero(diff)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def japanese_page(w: int = 900, h: int = 1300) -> tuple[np.ndarray, list[TruthBox]]:
    img = _screentone(h, w)
    font = ImageFont.truetype(JA_FONT, 34)
    specs = [
        (680, 300, 110, 200, ["おはよう", "ございます"]),
        (260, 330, 120, 210, ["今日は", "いい天気だね"]),
        (470, 900, 120, 220, ["本当に", "そう思う？"]),
    ]
    truth = []
    for cx, cy, rx, ry, cols in specs:
        bubble = _bubble(img, cx, cy, rx, ry)
        before = img.copy()
        pil = Image.fromarray(img[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        step = 40
        total_w = step * len(cols)
        x0 = cx + total_w // 2 - step
        for ci, col in enumerate(cols):  # vertical text, columns right to left
            y = cy - len(col) * 38 // 2
            for ch in col:
                d.text((x0 - ci * step, y), ch, font=font, fill=(0, 0, 0))
                y += 38
        img = np.array(pil)[:, :, ::-1].copy()
        truth.append(TruthBox(_ink_bbox(before, img), "".join(cols), bubble))
    return img, truth


def english_page(w: int = 900, h: int = 1300) -> tuple[np.ndarray, list[TruthBox]]:
    img = _screentone(h, w)
    font = ImageFont.truetype(EN_FONT, 30)
    specs = [
        (250, 260, 190, 110, ["GOOD MORNING!", "DID YOU SLEEP", "WELL?"]),
        (640, 640, 190, 110, ["I STAYED UP", "ALL NIGHT..."]),
        (300, 1050, 200, 110, ["THEN LET'S", "GET COFFEE!"]),
    ]
    truth = []
    for cx, cy, rx, ry, lines in specs:
        bubble = _bubble(img, cx, cy, rx, ry)
        before = img.copy()
        pil = Image.fromarray(img[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        lh = 36
        y = cy - len(lines) * lh // 2
        for line in lines:
            tw = d.textlength(line, font=font)
            d.text((cx - tw / 2, y), line, font=font, fill=(0, 0, 0))
            y += lh
        img = np.array(pil)[:, :, ::-1].copy()
        truth.append(TruthBox(_ink_bbox(before, img), " ".join(lines), bubble))
    return img, truth


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein / len(ref)), ignoring spaces."""
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(1, len(ref))
