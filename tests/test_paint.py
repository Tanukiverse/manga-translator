import numpy as np

from manga_translator.core import paint, versions
from manga_translator.core.pipeline import run_typeset
from manga_translator.core.project import PageDoc, Region, imwrite


def test_stroke_paints_solid_colour_and_erase_removes_it():
    layer = paint.blank((50, 60))
    paint.stroke(layer, (10, 25), (50, 25), 5, color_bgr=(0, 0, 255))
    base = np.full((50, 60, 3), 200, np.uint8)
    out = paint.composite(base, layer)
    assert tuple(out[25, 30]) == (0, 0, 255)  # centre of the stroke: exactly the colour
    assert tuple(out[2, 2]) == (200, 200, 200)  # away from it: untouched
    paint.stroke(layer, (10, 25), (50, 25), 9, erase=True)
    assert not layer[..., 3].any()
    assert paint.composite(base, layer) is base


def test_edges_blend_without_dark_fringe():
    # premultiplied storage: a white stroke on white must stay white at its soft edge
    layer = paint.blank((40, 40))
    paint.stroke(layer, (20, 5), (20, 35), 4, color_bgr=(255, 255, 255))
    out = paint.composite(np.full((40, 40, 3), 255, np.uint8), layer)
    assert out.min() >= 254


def test_painting_over_paint_replaces_the_colour():
    layer = paint.blank((30, 30))
    paint.stroke(layer, (15, 15), (15, 15), 8, color_bgr=(255, 0, 0))
    paint.stroke(layer, (15, 15), (15, 15), 8, color_bgr=(0, 255, 0))
    assert tuple(paint.composite(np.zeros((30, 30, 3), np.uint8), layer)[15, 15]) == (0, 255, 0)


def test_pick_reads_the_painted_colour():
    base = np.full((20, 20, 3), 30, np.uint8)
    layer = paint.blank((20, 20))
    paint.stroke(layer, (10, 10), (10, 10), 4, color_bgr=(1, 2, 3))
    assert paint.pick(base, layer, 10, 10) == (1, 2, 3)
    assert paint.pick(base, None, 10, 10) == (30, 30, 30)
    assert paint.pick(base, layer, 99, 0) is None


def _page(tmp_path):
    src = tmp_path / "p1.png"
    imwrite(src, np.full((120, 160, 3), 90, np.uint8))
    doc = PageDoc.create(src, tmp_path / "work", target_lang="th")
    doc.save_layer("cleaned", np.full((120, 160, 3), 240, np.uint8))
    doc.regions = [Region(1, [40, 30, 80, 60], translation="ทดสอบ")]
    doc.save()
    return doc


class _Models:
    class settings:
        font_family = ""


def test_paint_sits_under_the_translation_and_survives_typeset(tmp_path):
    doc = _page(tmp_path)
    layer = paint.blank((120, 160))
    paint.stroke(layer, (2, 110), (150, 110), 6, color_bgr=(0, 200, 0))  # a band away from the text
    paint.stroke(layer, (80, 60), (80, 60), 30, color_bgr=(0, 200, 0))  # a blob under the text
    doc.save_paint(layer)
    run_typeset(doc, _Models)
    final = doc.load_layer("final")
    assert tuple(final[110, 80]) == (0, 200, 0)  # the paint is in the finished page
    blob = final[45:75, 55:105].reshape(-1, 3)
    assert (blob == (0, 200, 0)).all(axis=1).any()  # paint shows around the letters ...
    assert not (blob == (0, 200, 0)).all(axis=1).all()  # ... and the text is drawn on top of it
    assert doc.load_layer("cleaned")[110, 80, 0] == 240  # the cleaned layer itself is untouched
    run_typeset(doc, _Models)  # running the step again keeps the paint
    assert tuple(doc.load_layer("final")[110, 80]) == (0, 200, 0)


def test_empty_paint_removes_the_file(tmp_path):
    doc = _page(tmp_path)
    layer = paint.blank((120, 160))
    paint.stroke(layer, (10, 10), (20, 20), 3, color_bgr=(5, 5, 5))
    doc.save_paint(layer)
    assert doc.paint_path.exists() and doc.load_paint().shape == (120, 160, 4)
    doc.save_paint(paint.blank((120, 160)))
    assert not doc.paint_path.exists() and doc.load_paint() is None


def test_history_keeps_and_restores_paint_and_reset_clears_it(tmp_path):
    doc = _page(tmp_path)
    layer = paint.blank((120, 160))
    paint.stroke(layer, (10, 10), (30, 10), 3, color_bgr=(0, 0, 255))
    doc.save_paint(layer)
    versions.snapshot(doc, "ก่อน: ใส่คำแปล")
    doc.save_paint(None)
    restored = versions.restore(doc, versions.list_versions(doc)[0])
    assert restored.paint_path.exists() and restored.load_paint()[10, 20, 3] == 255
    fresh = versions.reset(restored)
    assert not fresh.paint_path.exists()
