import cv2
import numpy as np

from manga_translator.core.detect import grow_mask_in, is_light_text
from manga_translator.core.ocr import readable
from manga_translator.core.project import Region
from manga_translator.core.typeset import render


def _caption(light: bool):
    """A caption box: white letters on black (light=True) or black letters on white."""
    bg, fg = (0, 255) if light else (255, 0)
    img = np.full((200, 400, 3), bg, np.uint8)
    cv2.putText(img, "WELL THEN", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (fg, fg, fg), 5)
    mask = np.zeros((200, 400), np.uint8)
    cv2.putText(mask, "WELL THEN", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 7)
    return img, mask, (30, 60, 340, 70)


def test_light_text_on_dark_is_recognised():
    img, mask, box = _caption(light=True)
    assert is_light_text(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask, box)


def test_dark_text_on_paper_is_not_light():
    img, mask, box = _caption(light=False)
    assert not is_light_text(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask, box)


def test_readable_inverts_only_light_captions():
    img, mask, box = _caption(light=True)
    out = readable(img, box, mask)
    x, y, w, h = box
    assert out[y:y + h, x:x + w].mean() > 150  # now dark letters on light paper
    assert out[0, 0].tolist() == img[0, 0].tolist()  # outside the box is untouched
    dark, dmask, dbox = _caption(light=False)
    assert readable(dark, dbox, dmask) is dark


def test_grow_mask_only_inside_the_box():
    mask = np.zeros((100, 200), np.uint8)
    mask[40:50, 40:60] = 255
    mask[40:50, 150:160] = 255  # another box far away
    grow_mask_in(mask, (40, 40, 20, 10), 6)
    assert mask[34:56, 34:66].mean() > mask[40:50, 40:60].mean() * 0.5  # grew around the first box
    assert mask[40:50, 143:149].max() == 0  # the other box did not grow


def test_white_translation_is_drawn_white():
    img = np.zeros((1200, 900, 3), np.uint8)  # page-sized, so the font is not capped tiny
    out, _ = render(img, [Region(1, [100, 100, 500, 200], translation="สวัสดี", color="#ffffff", outline="#000000")], "th")
    assert (out.min(axis=2) > 200).sum() > 100  # white glyph pixels on the black page
