"""Admit one immutable Python distribution artifact family."""

from __future__ import annotations

import hashlib
import re
import stat
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path

from python_candidate_common import CandidateError

FAMILY_MEMBER_BYTES_LIMIT = 16_777_216
FAMILY_TOTAL_BYTES_LIMIT = 67_108_864


@dataclass(frozen=True)
class CandidateFamily:
    sdist: Path
    wheel: Path
    version: str
    inventory: tuple[dict[str, object], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _basename(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise CandidateError(f"unsafe artifact basename: {value!r}")
    return value


def family_inventory(directory: Path) -> tuple[dict[str, object], ...]:
    """Hash an exact flat family of regular, singly-linked files."""

    if not directory.is_dir() or directory.is_symlink():
        raise CandidateError("artifact family must be one directory")
    paths = tuple(sorted(directory.iterdir(), key=lambda item: item.name.encode()))
    total_bytes = 0
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise CandidateError(f"artifact family member is not one regular file: {path.name}")
        size = path.stat().st_size
        if size > FAMILY_MEMBER_BYTES_LIMIT:
            raise CandidateError("artifact family member exceeds the raw byte ceiling")
        total_bytes += size
        if total_bytes > FAMILY_TOTAL_BYTES_LIMIT:
            raise CandidateError("artifact family exceeds the raw byte ceiling")
    return tuple(
        {
            "filename": _basename(path.name),
            "kind": "sdist" if path.name.endswith(".tar.gz") else "wheel",
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    )


def require_exact_maturin_wheel(path: Path, *, version: str) -> None:
    expected_name = (
        f"eqiora-{version}-cp311-abi3-"
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    if path.name != expected_name:
        raise CandidateError(f"wheel identity drifted: {path.name}")
    expected_tags = (
        "cp311-abi3-manylinux_2_17_x86_64",
        "cp311-abi3-manylinux2014_x86_64",
    )
    expected_wheel_member = f"eqiora-{version}.dist-info/WHEEL"
    expected_wheel_payload = (
        "Wheel-Version: 1.0\n"
        "Generator: maturin (1.15.0)\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {expected_tags[0]}\n"
        f"Tag: {expected_tags[1]}\n"
    ).encode()
    expected_metadata_member = f"eqiora-{version}.dist-info/METADATA"
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            wheel_members = tuple(
                member
                for member in archive.infolist()
                if member.filename.endswith(".dist-info/WHEEL")
            )
            metadata_members = tuple(
                member
                for member in archive.infolist()
                if member.filename.endswith(".dist-info/METADATA")
            )
            if len(wheel_members) != 1 or len(metadata_members) != 1:
                raise CandidateError(f"wheel metadata is ambiguous: {path.name}")
            wheel_member = wheel_members[0]
            metadata_member = metadata_members[0]
            if (
                wheel_member.filename != expected_wheel_member
                or not stat.S_ISREG(wheel_member.external_attr >> 16)
                or metadata_member.filename != expected_metadata_member
                or not stat.S_ISREG(metadata_member.external_attr >> 16)
            ):
                raise CandidateError(f"wheel metadata ownership drifted: {path.name}")
            wheel_payload = archive.read(wheel_member)
            metadata = BytesParser().parsebytes(archive.read(metadata_member))
    except (OSError, NotImplementedError, zipfile.BadZipFile) as error:
        raise CandidateError(f"wheel archive is invalid: {path.name}") from error
    if wheel_payload != expected_wheel_payload:
        raise CandidateError(f"wheel build metadata drifted: {path.name}")
    if metadata.get("Name") != "eqiora" or metadata.get("Version") != version:
        raise CandidateError(f"wheel distribution identity drifted: {path.name}")


def admit_candidate_family(directory: Path) -> CandidateFamily:
    """Admit exactly one sdist and one CPython 3.11 stable-ABI wheel."""

    inventory = family_inventory(directory)
    if len(inventory) != 2 or any(int(item["size"]) <= 0 for item in inventory):
        raise CandidateError("candidate requires one nonempty sdist and one stable-ABI wheel")
    paths = {path.name: path for path in directory.iterdir()}
    sdists = [path for path in paths.values() if path.name.endswith(".tar.gz")]
    if len(sdists) != 1:
        raise CandidateError("candidate requires exactly one source distribution")
    match = re.fullmatch(r"eqiora-(.+)\.tar\.gz", sdists[0].name)
    if match is None:
        raise CandidateError("source distribution identity drifted")
    version = match.group(1)
    if re.fullmatch(
        r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
        r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?",
        version,
    ) is None:
        raise CandidateError("source distribution version is not normalized")
    wheels = [path for path in paths.values() if path.name.endswith(".whl")]
    if len(wheels) != 1 or len(paths) != 2:
        raise CandidateError("candidate family contains a non-distribution file")
    wheel = wheels[0]
    require_exact_maturin_wheel(wheel, version=version)
    if family_inventory(directory) != inventory:
        raise CandidateError("candidate family changed during admission")
    return CandidateFamily(
        sdists[0],
        wheel,
        version,
        inventory,
    )
