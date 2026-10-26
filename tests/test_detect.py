import synth

from manga_translator.core.detect import sort_reading_order


def _match(dets, truth):
    for t in truth:
        best = max((synth.iou(d.bbox, t.bbox) for d in dets), default=0)
        assert best > 0.6, f"missed {t.text}: best IoU {best:.2f}"


def test_detect_japanese(detector, ja_page):
    img, truth = ja_page
    dets, mask = detector(img)
    assert len(dets) == len(truth)
    _match(dets, truth)
    # mask covers the ink of every box and stays near the boxes
    for t in truth:
        x, y, w, h = t.bbox
        assert mask[y:y + h, x:x + w].mean() > 20
    assert (mask > 0).mean() < 0.1


def test_detect_english(detector, en_page):
    img, truth = en_page
    dets, _ = detector(img)
    assert len(dets) == len(truth)
    _match(dets, truth)


def test_reading_order_rtl_and_ltr():
    boxes = [(10, 10, 50, 50), (200, 15, 50, 50), (100, 300, 50, 50)]
    assert sort_reading_order(boxes, right_to_left=True) == [1, 0, 2]
    assert sort_reading_order(boxes, right_to_left=False) == [0, 1, 2]
