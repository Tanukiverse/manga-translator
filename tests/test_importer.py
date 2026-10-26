import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from manga_translator.core import importer
from manga_translator.core.project import imread, imwrite


def _png(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(path, np.full((20, 10, 3), value, np.uint8))
    return path


def test_zip_pages_keep_their_order_and_stay_inside(tmp_path):
    src = tmp_path / "src"
    a, b, c = _png(src / "a.png", 10), _png(src / "b.png", 20), _png(src / "c.png", 30)
    arc = tmp_path / "ตอน 5.cbz"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.write(c, "ch5/p10.png")
        zf.write(a, "ch5/p2.png")
        zf.write(b, "../../evil.png")  # zip-slip attempt: must land inside the chapter folder
        zf.writestr("ch5/notes.txt", "not an image")
    chapter = importer.import_archive(arc, tmp_path / "work")
    assert chapter == tmp_path / "work" / importer.IMPORTED / "ตอน 5"
    files = sorted(p.name for p in chapter.iterdir())
    assert len(files) == 3 and not (tmp_path / "evil.png").exists()
    # natural order of inner paths: ../../evil, ch5/p2, ch5/p10
    assert [imread(chapter / f)[0, 0, 0] for f in files] == [20, 10, 30]
    assert arc.exists()  # the archive itself is untouched


def test_second_import_does_not_overwrite(tmp_path):
    arc = tmp_path / "c.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.write(_png(tmp_path / "x.png", 1), "x.png")
    first = importer.import_archive(arc, tmp_path / "work")
    second = importer.import_archive(arc, tmp_path / "work")
    assert first != second and second.name == "c (2)"


def test_7z_through_windows_tar(tmp_path):
    tar = Path(r"C:\Windows\System32\tar.exe")
    if not tar.exists():
        pytest.skip("Windows tar.exe not available")
    src = tmp_path / "src"
    _png(src / "p1.png", 50)
    _png(src / "sub" / "p2.png", 60)
    arc = tmp_path / "chapter.7z"
    subprocess.run([str(tar), "--format", "7zip", "-cf", str(arc), "-C", str(src), "p1.png", "sub"], check=True)
    chapter = importer.import_archive(arc, tmp_path / "work")
    assert [imread(p)[0, 0, 0] for p in sorted(chapter.iterdir())] == [50, 60]


def test_archive_without_images_is_refused(tmp_path):
    arc = tmp_path / "empty.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("readme.txt", "hi")
    with pytest.raises(importer.ImportFailed):
        importer.import_archive(arc, tmp_path / "work")
    assert not (tmp_path / "work" / importer.IMPORTED / "empty").exists()


def test_broken_zip_is_refused(tmp_path):
    arc = tmp_path / "bad.cbz"
    arc.write_bytes(b"not a zip")
    with pytest.raises(importer.ImportFailed):
        importer.import_archive(arc, tmp_path / "work")


def test_classify_and_copy(tmp_path):
    img = _png(tmp_path / "in" / "p1.png", 5)
    (tmp_path / "in" / "notes.txt").write_text("x")
    folders, archives, images, other = importer.classify(
        [tmp_path / "in", img, tmp_path / "in" / "notes.txt", tmp_path / "a.cbr"])
    assert folders == [tmp_path / "in"] and images == [img] and archives == [tmp_path / "a.cbr"]
    assert other == [tmp_path / "in" / "notes.txt"]
    dest = tmp_path / "out"
    importer.copy_images([img], dest)
    importer.copy_images([img], dest)  # same name again: kept as a second copy
    assert sorted(p.name for p in dest.iterdir()) == ["p1 (2).png", "p1.png"]
    assert img.exists()
