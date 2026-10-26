import cv2
import numpy as np

from manga_translator.core.ocr import normalize_english, text_lines, uppercase_page
from manga_translator.core.translate import TranslatorConfig, build_messages


def _draw(lines):
    img = np.full((160, 320), 255, np.uint8)
    for text, (x, y) in lines:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    return img


def test_lines_split_side_by_side_blocks():
    # left block of three lines, a word beside it offset by half a line
    img = _draw([("THE", (40, 30)), ("SOLDIERS", (20, 52)), ("ARE BACK!", (20, 74)), ("CAPTAIN!", (200, 42))])
    lines = text_lines(img, np.full_like(img, 255))
    assert len(lines) == 4
    xs = sorted(l[2] for l in lines)
    assert xs[-1] > 150  # "CAPTAIN!" read as its own line, not glued to the left block


def test_lines_only_inside_mask():
    img = _draw([("HELLO", (20, 40)), ("NOISE", (20, 120))])
    mask = np.zeros_like(img)
    mask[:70] = 255
    assert len(text_lines(img, mask)) == 1


def test_normalize_english():
    assert normalize_english("TAKEDA，2OOO- STRONG") == "TAKEDA,2000-STRONG"
    assert normalize_english("SER- GE- ANT!") == "SER-GE-ANT!"
    assert normalize_english("Hello  there") == "Hello there"  # mixed case is left alone


def test_uppercase_page():
    page = ["THE CAVALRY IS COMING!", "MOu-uu!", "STAY BEHIND ME..."]
    assert uppercase_page(page)[1] == "MOU-UU!"
    assert uppercase_page(["Hello", "there"]) == ["Hello", "there"]


def test_short_bubbles_marked_for_translator():
    msgs = build_messages([(1, "HM?"), (2, "THE BRIDGE COLLAPSED BEFORE WE COULD CROSS IT...")], "en", "th", TranslatorConfig())
    assert '"short": true' in msgs[1]["content"]
    assert msgs[1]["content"].count('"short"') == 1
