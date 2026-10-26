import os
import time

from manga_translator.core.pages import load_order, natural_key, save_order, sort_pages


def _files(tmp_path, specs):
    out = []
    for name, size in specs:
        p = tmp_path / name
        p.write_bytes(b"x" * size)
        out.append(p)
        time.sleep(0.01)
    return out


def test_natural_name_order(tmp_path):
    paths = _files(tmp_path, [("p10.png", 1), ("p2.png", 1), ("p1.png", 1), ("ตอน9.png", 1), ("ตอน10.png", 1)])
    names = [p.name for p in sort_pages(paths, "name")]
    assert names.index("p1.png") < names.index("p2.png") < names.index("p10.png")
    assert names.index("ตอน9.png") < names.index("ตอน10.png")
    assert [p.name for p in sort_pages(paths, "name", reverse=True)] == names[::-1]


def test_mtime_and_size(tmp_path):
    a, b, c = _files(tmp_path, [("a.png", 30), ("b.png", 10), ("c.png", 20)])
    now = time.time()
    os.utime(a, (now, now - 100))
    os.utime(b, (now, now - 50))
    os.utime(c, (now, now - 10))
    assert [p.name for p in sort_pages([c, a, b], "mtime")] == ["a.png", "b.png", "c.png"]
    assert [p.name for p in sort_pages([c, a, b], "size", reverse=True)] == ["a.png", "c.png", "b.png"]


def test_manual_order_survives_added_and_removed_files(tmp_path):
    paths = _files(tmp_path, [("p1.png", 1), ("p2.png", 1), ("p3.png", 1)])
    save_order(tmp_path / "work", [paths[2], paths[0], paths[1]])
    order = load_order(tmp_path / "work")
    assert order == ["p3.png", "p1.png", "p2.png"]
    paths[1].unlink()
    new = _files(tmp_path, [("p0.png", 1)])[0]
    now = [paths[0], paths[2], new]
    assert [p.name for p in sort_pages(now, "manual", manual=order)] == ["p3.png", "p1.png", "p0.png"]


def test_natural_key_mixed_types_do_not_crash():
    assert sorted(["1a", "a1", "a", "10"], key=natural_key)
