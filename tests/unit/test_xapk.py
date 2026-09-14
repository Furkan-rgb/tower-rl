from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tower_rl.xapk import XapkInspectionError, _safe_apk_entries, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"tower-rl")

    assert sha256_file(path) == "e8ca2e8cba3b9b218315dae388cc1880c3ed5e46f60818af7f529fc7d8f157a8"


def test_archive_requires_an_apk(tmp_path: Path) -> None:
    path = tmp_path / "empty.xapk"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")

    with zipfile.ZipFile(path) as archive, pytest.raises(
        XapkInspectionError, match="contains no APK"
    ):
        _safe_apk_entries(archive)


def test_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.xapk"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../base.apk", b"not-an-apk")

    with zipfile.ZipFile(path) as archive, pytest.raises(
        XapkInspectionError, match="unsafe archive path"
    ):
        _safe_apk_entries(archive)
