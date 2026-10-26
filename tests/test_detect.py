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


def test_box_spanning_two_bubbles_is_trimmed():
    from manga_translator.core.detect import Detection, resolve_overlaps

    # the shape seen on a real page: one box over two side-by-side bubbles + a box for the right one
    big = Detection((1001, 178, 288, 160), 0.9)
    right = Detection((1176, 190, 124, 93), 0.8)
    out = {d.score: d.bbox for d in resolve_overlaps([big, right])}
    assert out[0.8] == (1176, 190, 124, 93)
    assert out[0.9] == (1001, 178, 175, 160)  # now ends where the right bubble's box starts


def test_duplicate_box_in_the_middle_is_dropped():
    from manga_translator.core.detect import Detection, resolve_overlaps

    out = resolve_overlaps([Detection((0, 0, 100, 100), 0.9), Detection((10, 10, 80, 80), 0.5)])
    assert [d.bbox for d in out] == [(0, 0, 100, 100)]


def test_separate_boxes_untouched():
    from manga_translator.core.detect import Detection, resolve_overlaps

    a, b = Detection((0, 0, 50, 50), 0.9), Detection((100, 0, 50, 50), 0.9)
    assert sorted(d.bbox for d in resolve_overlaps([a, b])) == [(0, 0, 50, 50), (100, 0, 50, 50)]
