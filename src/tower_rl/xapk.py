"""Safe, metadata-only inspection of locally supplied XAPK archives."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree


class XapkInspectionError(ValueError):
    """Raised when an XAPK cannot be safely or coherently inspected."""


@dataclass(frozen=True)
class ApkMetadata:
    """Public-safe manifest and archive metadata for one APK split."""

    name: str
    size_bytes: int
    sha256: str
    package_name: str
    version_code: str
    version_name: str | None
    min_sdk: str | None
    target_sdk: str | None
    split_name: str | None
    config_for_split: str | None
    native_abis: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class XapkMetadata:
    """Metadata for a complete split-APK archive."""

    archive_name: str
    archive_size_bytes: int
    archive_sha256: str
    package_name: str
    version_code: str
    version_name: str
    min_sdk: str | None
    target_sdk: str | None
    native_abis: tuple[str, ...]
    apks: tuple[ApkMetadata, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "apks": [apk.to_dict() for apk in self.apks],
        }


def _sha256_stream(chunks: Iterator[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest without loading the whole file into memory."""

    with path.open("rb") as stream:
        return _sha256_stream(iter(lambda: stream.read(1024 * 1024), b""))


def _safe_apk_entries(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    entries: list[zipfile.ZipInfo] = []
    basenames: set[str] = set()
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or ".." in path.parts:
            raise XapkInspectionError(f"unsafe archive path: {entry.filename}")
        if entry.is_dir() or path.suffix.lower() != ".apk":
            continue
        if path.name in basenames:
            raise XapkInspectionError(f"duplicate APK basename: {path.name}")
        basenames.add(path.name)
        entries.append(entry)
    if not entries:
        raise XapkInspectionError("archive contains no APK files")
    return tuple(entries)


ANDROID_NAMESPACE = "{http://schemas.android.com/apk/res/android}"


def _read_manifest(tool: Path, apk_path: Path) -> ElementTree.Element:
    result = subprocess.run(
        [str(tool), "manifest", "print", str(apk_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise XapkInspectionError(f"apkanalyzer failed for {apk_path.name}: {detail}")
    try:
        return ElementTree.fromstring(result.stdout)
    except ElementTree.ParseError as error:
        raise XapkInspectionError(f"invalid decoded manifest for {apk_path.name}") from error


def _android_attribute(element: ElementTree.Element | None, name: str) -> str | None:
    return element.get(f"{ANDROID_NAMESPACE}{name}") if element is not None else None


def _native_abis(apk_path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(apk_path) as apk:
        abis = {
            PurePosixPath(name).parts[1]
            for name in apk.namelist()
            if len(PurePosixPath(name).parts) >= 3
            and PurePosixPath(name).parts[0] == "lib"
            and name.endswith(".so")
        }
    return tuple(sorted(abis))


def inspect_xapk(path: Path, apkanalyzer: Path) -> XapkMetadata:
    """Inspect XAPK metadata using Android's manifest analyzer.

    APK contents are copied only to an operating-system temporary directory and
    removed when inspection ends. No extracted game data enters the repository.
    """

    if not path.is_file():
        raise XapkInspectionError(f"XAPK not found: {path}")
    if not apkanalyzer.is_file():
        raise XapkInspectionError(f"apkanalyzer not found: {apkanalyzer}")

    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as error:
        raise XapkInspectionError(f"invalid XAPK zip archive: {path}") from error

    with archive, tempfile.TemporaryDirectory(prefix="tower-rl-xapk-") as directory:
        entries = _safe_apk_entries(archive)
        extracted: list[tuple[zipfile.ZipInfo, Path]] = []
        temporary_root = Path(directory)
        for entry in entries:
            destination = temporary_root / PurePosixPath(entry.filename).name
            with archive.open(entry) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append((entry, destination))

        apk_reports: list[ApkMetadata] = []
        for entry, apk_path in extracted:
            manifest = _read_manifest(apkanalyzer, apk_path)
            uses_sdk = manifest.find("uses-sdk")
            apk_reports.append(
                ApkMetadata(
                    name=apk_path.name,
                    size_bytes=entry.file_size,
                    sha256=sha256_file(apk_path),
                    package_name=manifest.get("package", ""),
                    version_code=_android_attribute(manifest, "versionCode") or "",
                    version_name=_android_attribute(manifest, "versionName"),
                    min_sdk=_android_attribute(uses_sdk, "minSdkVersion"),
                    target_sdk=_android_attribute(uses_sdk, "targetSdkVersion"),
                    split_name=manifest.get("split"),
                    config_for_split=manifest.get("configForSplit"),
                    native_abis=_native_abis(apk_path),
                )
            )

    bases = [apk for apk in apk_reports if apk.version_name]
    if len(bases) != 1:
        raise XapkInspectionError(
            f"expected exactly one base APK with a version name, found {len(bases)}"
        )
    base = bases[0]
    identities = {(apk.package_name, apk.version_code) for apk in apk_reports}
    if identities != {(base.package_name, base.version_code)}:
        raise XapkInspectionError(f"split identity mismatch: {sorted(identities)}")

    native_abis = tuple(sorted({abi for apk in apk_reports for abi in apk.native_abis}))
    return XapkMetadata(
        archive_name=path.name,
        archive_size_bytes=path.stat().st_size,
        archive_sha256=sha256_file(path),
        package_name=base.package_name,
        version_code=base.version_code,
        version_name=base.version_name or "",
        min_sdk=base.min_sdk,
        target_sdk=base.target_sdk,
        native_abis=native_abis,
        apks=tuple(sorted(apk_reports, key=lambda apk: apk.name)),
    )
