#!/usr/bin/env python3
"""Validate one complete Eqiora Python candidate artifact set.

The manifest is the authority for filenames and hashes. This checker is used
both after an Actions artifact download and immediately before production
publication; it never rebuilds a candidate.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import sys
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python_candidate_common import CandidateError, python_distribution_version


MANIFEST_FORMAT = "eqiora.python-distribution-candidate/v4"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIRED_PROFILES = ("base", "jax", "matplotlib", "torch", "typing")
FAMILY_MEMBER_BYTES_LIMIT = 16_777_216
FAMILY_TOTAL_BYTES_LIMIT = 67_108_864
MANIFEST_BYTES_LIMIT = 1_048_576
ARCHIVE_MEMBER_COUNT_LIMIT = 4_096
# The reviewed ordinary-GIL manylinux extensions are under 33 MiB uncompressed.
# Keep a bounded 40 MiB per-member envelope distinct from the tighter raw-family
# and expanded-archive ceilings so one native module fits without widening the
# whole-family admission boundary.
ARCHIVE_MEMBER_BYTES_LIMIT = 41_943_040
ARCHIVE_TOTAL_BYTES_LIMIT = 67_108_864


def _base_checks() -> frozenset[str]:
    checks = {
        "generated-public-api",
        "sdist-to-wheel-rebuild",
        "twine-strict",
        "cp312:numpy-2.1.0-floor",
    }
    for python in ("311", "312", "313", "314"):
        checks.update(
            {
                f"cp{python}:installed-wheel",
                f"cp{python}:base-and-numpy",
                f"cp{python}:packaged-mixed-boundary-elasticity-demo",
                f"cp{python}:async-and-cancellation",
                f"cp{python}:public-smoke-base",
                f"cp{python}:matplotlib-free-base",
            }
        )
    return frozenset(checks)


PROFILE_CHECKS = {
    "base": _base_checks(),
    "jax": frozenset({"cp313:jax", "cp313:public-smoke-jax"}),
    "matplotlib": frozenset(
        {
            "cp313:matplotlib",
            "cp313:packaged-exact-cylinder-pressure-demo",
            "cp313:packaged-mixed-boundary-displacement-demo",
        }
    ),
    "torch": frozenset({"cp313:torch", "cp313:public-smoke-torch"}),
    "typing": frozenset(
        {
            "cp311:strict-base-typing",
            "cp312:strict-base-typing",
            "cp313:strict-base-typing",
            "cp314:strict-base-typing",
            "cp313:complete-public-typing",
        }
    ),
}


class ManifestError(RuntimeError):
    """The retained candidate is malformed or differs from its manifest."""


@dataclass(frozen=True)
class Artifact:
    filename: str
    kind: str
    size: int
    sha256: str
    python: str | None


@dataclass(frozen=True)
class Candidate:
    version: str
    commit: str
    expected_tag: str
    artifacts: tuple[Artifact, ...]
    checks: frozenset[str]


def require_candidate_profile(candidate: Candidate, profile: str) -> None:
    """Require one closed profile from an already parsed candidate manifest."""

    try:
        required = PROFILE_CHECKS[profile]
    except KeyError as error:
        raise ManifestError(f"unknown candidate profile {profile!r}") from error
    missing = sorted(required - candidate.checks)
    if missing:
        raise ManifestError(
            f"candidate {profile} profile omits required check {missing[0]!r}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{location} must be an object")
    return value


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{location} must be a nonempty string")
    return value


def _integer(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestError(f"{location} must be a nonnegative integer")
    return value


def _exact_keys(value: Any, keys: set[str], location: str) -> dict[str, Any]:
    item = _object(value, location)
    if set(item) != keys:
        raise ManifestError(
            f"{location} schema has an unexpected key or missing member"
        )
    return item


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{location} must be an array")
    return value


def _positive_integer(value: Any, location: str) -> int:
    result = _integer(value, location)
    if result == 0:
        raise ManifestError(f"{location} must be a positive integer")
    return result


def _sha(value: Any, location: str) -> str:
    result = _text(value, location)
    if SHA256.fullmatch(result) is None:
        raise ManifestError(f"{location} must be a lowercase SHA-256")
    return result


def _git_sha(value: Any, location: str) -> str:
    result = _text(value, location)
    if FULL_SHA.fullmatch(result) is None:
        raise ManifestError(f"{location} must be a full lowercase Git SHA")
    return result


def _basename(value: Any, location: str) -> str:
    result = _text(value, location)
    if (
        result in {".", ".."}
        or "/" in result
        or "\\" in result
        or "\0" in result
        or Path(result).name != result
    ):
        raise ManifestError(f"{location} must be a safe basename")
    return result


def _relative_path(value: Any, location: str) -> str:
    result = _text(value, location)
    parts = result.split("/")
    if (
        result.startswith("/")
        or "\\" in result
        or "\0" in result
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ManifestError(f"{location} must be a safe relative path")
    return result


def _safe_archive_name(name: str, location: str) -> str:
    if name.endswith("/"):
        name = name[:-1]
    return _relative_path(name, location)


@dataclass(frozen=True)
class _FamilyScan:
    sdist_members: dict[str, bytes]
    wheel_members: tuple[dict[str, bytes], ...]
    wheel_metadata: tuple[bytes, ...]


def _scan_family(document: dict[str, Any], artifacts: Path) -> _FamilyScan:
    records = document.get("artifacts")
    if not isinstance(records, list) or len(records) != 5:
        raise ManifestError("candidate family must declare exactly five artifacts")
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise ManifestError("candidate artifact family must be one directory")
    sdist_members: dict[str, bytes] = {}
    wheel_members: list[dict[str, bytes]] = []
    wheel_metadata: list[bytes] = []
    total_bytes = 0
    filenames: set[str] = set()
    for index, raw in enumerate(records):
        record = _object(raw, f"artifacts[{index}]")
        filename = _basename(record.get("filename"), f"artifacts[{index}].filename")
        if filename in filenames:
            raise ManifestError("candidate manifest has duplicate artifact filenames")
        filenames.add(filename)
        path = artifacts / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or path.stat().st_size <= 0
            or path.stat().st_size > FAMILY_MEMBER_BYTES_LIMIT
        ):
            raise ManifestError(
                f"candidate family member is not one bounded regular file: {filename}"
            )
        total_bytes += path.stat().st_size
        if total_bytes > FAMILY_TOTAL_BYTES_LIMIT:
            raise ManifestError("candidate family exceeds the raw byte ceiling")
        kind = record.get("kind")
        if kind == "sdist":
            archive_member_count = 0
            archive_total_bytes = 0
            try:
                with tarfile.open(path, mode="r:*") as archive:
                    for member in archive:
                        name = _safe_archive_name(member.name, "sdist member")
                        if member.isdir():
                            continue
                        if not member.isfile():
                            raise ManifestError(
                                "sdist has an unsafe non-regular member"
                            )
                        archive_member_count += 1
                        archive_total_bytes += member.size
                        if (
                            archive_member_count > ARCHIVE_MEMBER_COUNT_LIMIT
                            or member.size > ARCHIVE_MEMBER_BYTES_LIMIT
                            or archive_total_bytes > ARCHIVE_TOTAL_BYTES_LIMIT
                        ):
                            raise ManifestError(
                                "candidate archive exceeds its expanded bounds"
                            )
                        source = archive.extractfile(member)
                        if source is None:
                            raise ManifestError("sdist regular member cannot be read")
                        payload = source.read(member.size + 1)
                        if len(payload) != member.size:
                            raise ManifestError("sdist member size differs from its header")
                        if name in sdist_members:
                            raise ManifestError("sdist has duplicate member names")
                        sdist_members[name] = payload
            except (OSError, tarfile.TarError) as error:
                raise ManifestError(f"cannot safely scan sdist: {error}") from error
        elif kind == "wheel":
            archive_member_count = 0
            archive_total_bytes = 0
            members: dict[str, bytes] = {}
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        name = _safe_archive_name(member.filename, "wheel member")
                        if member.is_dir():
                            continue
                        mode = member.external_attr >> 16
                        if mode and (mode & 0o170000) not in {0, 0o100000}:
                            raise ManifestError(
                                "wheel has an unsafe non-regular member"
                            )
                        archive_member_count += 1
                        archive_total_bytes += member.file_size
                        if (
                            archive_member_count > ARCHIVE_MEMBER_COUNT_LIMIT
                            or member.file_size > ARCHIVE_MEMBER_BYTES_LIMIT
                            or archive_total_bytes > ARCHIVE_TOTAL_BYTES_LIMIT
                        ):
                            raise ManifestError(
                                "candidate archive exceeds its expanded bounds"
                            )
                        if name in members:
                            raise ManifestError("wheel has duplicate member names")
                        payload = archive.read(member)
                        if len(payload) != member.file_size:
                            raise ManifestError(
                                "wheel member size differs from its header"
                            )
                        members[name] = payload
            except (OSError, zipfile.BadZipFile) as error:
                raise ManifestError(f"cannot safely scan wheel: {error}") from error
            metadata = [
                payload
                for name, payload in members.items()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata) != 1:
                raise ManifestError("wheel has an ambiguous METADATA member")
            wheel_members.append(members)
            wheel_metadata.append(metadata[0])
    if not sdist_members or len(wheel_members) != 4:
        raise ManifestError("candidate family must contain one sdist and four wheels")
    return _FamilyScan(sdist_members, tuple(wheel_members), tuple(wheel_metadata))


def _retained_distribution_identity(scan: _FamilyScan) -> tuple[str, str]:
    roots = {name.split("/", maxsplit=1)[0] for name in scan.sdist_members}
    if len(roots) != 1:
        raise ManifestError("sdist has an ambiguous source root")
    root = roots.pop()

    def member(relative: str) -> bytes:
        try:
            return scan.sdist_members[f"{root}/{relative}"]
        except KeyError as error:
            raise ManifestError(
                f"candidate sdist omits retained identity member {relative}"
            ) from error

    try:
        cargo = tomllib.loads(member("Cargo.toml").decode("utf-8"))
        pyproject = tomllib.loads(member("pyproject.toml").decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ManifestError(
            f"retained source identity cannot be parsed: {error}"
        ) from error

    workspace = cargo.get("workspace")
    workspace_package = (
        workspace.get("package") if isinstance(workspace, dict) else None
    )
    raw_version = (
        workspace_package.get("version")
        if isinstance(workspace_package, dict)
        else None
    )
    if not isinstance(raw_version, str) or not raw_version:
        raise ManifestError("retained Cargo version is not one nonempty string")

    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ManifestError("retained Python project table is unavailable")
    if "version" in project or project.get("dynamic") != ["version"]:
        raise ManifestError("retained Python version must be exactly dynamic")

    try:
        return raw_version, python_distribution_version(raw_version)
    except CandidateError as error:
        raise ManifestError(str(error)) from error


def _candidate_closed(document: dict[str, Any], scan: _FamilyScan) -> Candidate:
    _exact_keys(
        document,
        {
            "acceptance",
            "artifacts",
            "build",
            "checks",
            "format",
            "nonclaims",
            "project",
            "source",
            "version",
        },
        "manifest",
    )
    if document["format"] != MANIFEST_FORMAT:
        raise ManifestError("candidate manifest has an unsupported format")
    if document.get("project") != "eqiora" or document.get("acceptance") != "complete":
        raise ManifestError("only a complete Eqiora candidate is supported")
    _raw_version, retained_version = _retained_distribution_identity(scan)
    version = _text(document.get("version"), "version")
    if version != retained_version:
        raise ManifestError("candidate version differs from retained Cargo authority")
    source = _exact_keys(
        document.get("source"),
        {"commit", "expected_tag", "tags", "tree"},
        "source",
    )
    commit = _git_sha(source["commit"], "source.commit")
    expected_tag = _text(source["expected_tag"], "source.expected_tag")
    if source["tree"] != "clean" or expected_tag != f"v{version}":
        raise ManifestError("candidate source identity drifted")
    tags = _array(source["tags"], "source.tags")
    if (
        any(not isinstance(tag, str) or not tag for tag in tags)
        or tags != sorted(tags)
        or len(tags) != len(set(tags))
    ):
        raise ManifestError("source.tags must be canonically sorted unique strings")
    build = _exact_keys(
        document.get("build"),
        {"dependency_profiles", "sdist_rebuilt", "tools", "wheel_family"},
        "build",
    )
    if build["sdist_rebuilt"] is not True:
        raise ManifestError("candidate wheels were not recorded as sdist rebuilds")
    family = _object(build["wheel_family"], "build.wheel_family")
    if family != {
        "implementation": "CPython",
        "ordinary_gil": True,
        "versions": ["3.11", "3.12", "3.13", "3.14"],
        "platform": "manylinux_2_17_x86_64",
        "abi3": False,
    }:
        raise ManifestError("candidate wheel family drifted")
    tools = _exact_keys(
        build["tools"],
        {"cargo", "maturin", "mypy", "pytest", "rustc", "twine", "uv"},
        "build.tools",
    )
    if any(not isinstance(value, str) or not value for value in tools.values()):
        raise ManifestError("candidate build tools must be nonempty strings")
    dependency_profiles = _exact_keys(
        build["dependency_profiles"], {"numpy_floor"}, "build.dependency_profiles"
    )
    numpy_floor = _object(
        dependency_profiles["numpy_floor"],
        "build.dependency_profiles.numpy_floor",
    )
    if numpy_floor != {
        "python": "3.12",
        "requirement": "numpy==2.1.0",
        "observed": "2.1.0",
        "profile": "cp312:numpy-2.1.0-floor",
    }:
        raise ManifestError("candidate NumPy floor profile drifted")
    artifacts: list[Artifact] = []
    for index, raw in enumerate(_array(document.get("artifacts"), "artifacts")):
        entry = _object(raw, f"artifacts[{index}]")
        expected_keys = {"filename", "kind", "sha256", "size"}
        if entry.get("kind") == "wheel":
            expected_keys.update({"abi", "platform", "python"})
        _exact_keys(entry, expected_keys, f"artifacts[{index}]")
        filename = _basename(entry.get("filename"), f"artifacts[{index}].filename")
        kind = _text(entry.get("kind"), f"artifacts[{index}].kind")
        size = _positive_integer(entry.get("size"), f"artifacts[{index}].size")
        digest = _sha(entry.get("sha256"), f"artifacts[{index}].sha256")
        python = entry.get("python")
        if kind == "wheel":
            python = _text(python, f"artifacts[{index}].python")
            compact = python.replace(".", "")
            if (
                entry.get("abi") != f"cp{compact}"
                or entry.get("platform") != "manylinux_2_17_x86_64"
            ):
                raise ManifestError("candidate wheel ABI/platform record drifted")
        elif kind == "sdist" and python is not None:
            raise ManifestError("sdist must not declare a Python interpreter")
        elif kind != "sdist":
            raise ManifestError("candidate has an unsupported artifact kind")
        artifacts.append(Artifact(filename, kind, size, digest, python))
    sdists = [artifact for artifact in artifacts if artifact.kind == "sdist"]
    wheels = [artifact for artifact in artifacts if artifact.kind == "wheel"]
    if (
        len(artifacts) != 5
        or len(sdists) != 1
        or sdists[0].filename != f"eqiora-{version}.tar.gz"
        or sorted(artifact.python for artifact in wheels)
        != ["3.11", "3.12", "3.13", "3.14"]
    ):
        raise ManifestError("candidate artifact family drifted")
    by_python = {artifact.python: artifact for artifact in wheels}
    for python in ("3.11", "3.12", "3.13", "3.14"):
        compact = python.replace(".", "")
        artifact = by_python[python]
        expected_name = (
            f"eqiora-{version}-cp{compact}-cp{compact}-"
            "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        )
        if artifact.filename != expected_name:
            raise ManifestError("candidate wheel filename/family drifted")

    expected_metadata_member = f"eqiora-{version}.dist-info/METADATA"
    for members, payload in zip(scan.wheel_members, scan.wheel_metadata, strict=True):
        metadata_members = [
            name for name in members if name.endswith(".dist-info/METADATA")
        ]
        if metadata_members != [expected_metadata_member]:
            raise ManifestError(
                "wheel METADATA ownership differs from candidate version"
            )
        metadata = email.parser.BytesParser().parsebytes(payload)
        if metadata.get("Name") != "eqiora" or metadata.get("Version") != version:
            raise ManifestError(
                "wheel metadata version differs from retained Cargo authority"
            )
    checks_raw = _array(document.get("checks"), "checks")
    if (
        any(not isinstance(value, str) or not value for value in checks_raw)
        or checks_raw != sorted(checks_raw)
        or len(checks_raw) != len(set(checks_raw))
    ):
        raise ManifestError("checks must be canonically sorted unique nonempty strings")
    nonclaims = _array(document.get("nonclaims"), "nonclaims")
    if nonclaims != [
        "reproducible-build-certification",
        "artifact-signature",
        "macos-or-windows",
        "abi3",
        "free-threaded-cpython",
        "production-pypi-publication",
    ]:
        raise ManifestError("candidate nonclaims drifted")
    candidate = Candidate(
        version, commit, expected_tag, tuple(artifacts), frozenset(checks_raw)
    )
    for profile in REQUIRED_PROFILES:
        require_candidate_profile(candidate, profile)
    return candidate


def load_candidate_family(
    manifest: Path,
    artifacts: Path,
    *,
    requested_profiles: tuple[str, ...] = (),
) -> Candidate:
    """Admit the current manifest only after scanning its retained family."""
    try:
        info = manifest.lstat()
        if manifest.is_symlink() or not manifest.is_file() or info.st_nlink != 1:
            raise ManifestError("candidate manifest must be one regular file")
        if info.st_size > MANIFEST_BYTES_LIMIT:
            raise ManifestError("candidate manifest exceeds its byte ceiling")
        document = _object(json.loads(manifest.read_text(encoding="utf-8")), "manifest")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read candidate manifest: {error}") from error
    if document.get("format") != MANIFEST_FORMAT:
        raise ManifestError("candidate manifest has an unsupported format")
    scan = _scan_family(document, artifacts)
    candidate = _candidate_closed(document, scan)
    expected_family = {artifact.filename for artifact in candidate.artifacts}
    actual_family = {path.name for path in artifacts.iterdir()}
    if actual_family != expected_family:
        raise ManifestError("candidate artifact directory is not the exact family")
    verify_artifacts(candidate, artifacts)
    for profile in requested_profiles:
        require_candidate_profile(candidate, profile)
    return candidate


def verify_artifacts(candidate: Candidate, directory: Path) -> None:
    """Require exactly the manifested distribution files and matching bytes."""
    if not directory.is_dir():
        raise ManifestError(f"artifact directory does not exist: {directory}")
    expected = {artifact.filename for artifact in candidate.artifacts}
    actual = {
        path.name for path in directory.iterdir() if path.name not in {".DS_Store"}
    }
    if actual != expected:
        raise ManifestError(
            f"artifact directory differs: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    by_name = {artifact.filename: artifact for artifact in candidate.artifacts}
    for filename in sorted(expected):
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise ManifestError(f"artifact is not one regular file: {filename}")
        artifact = by_name[filename]
        if path.stat().st_size != artifact.size:
            raise ManifestError(f"artifact size differs: {filename}")
        if file_sha256(path) != artifact.sha256:
            raise ManifestError(f"artifact SHA-256 differs: {filename}")


def verify_manifest_hash(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    if SHA256.fullmatch(expected) is None:
        raise ManifestError("expected manifest SHA-256 is malformed")
    if file_sha256(path) != expected:
        raise ManifestError("candidate manifest SHA-256 differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-tag")
    arguments = parser.parse_args()

    try:
        verify_manifest_hash(arguments.manifest, arguments.manifest_sha256)
        candidate = load_candidate_family(
            arguments.manifest,
            arguments.artifacts,
        )
        if (
            arguments.expected_commit is not None
            and candidate.commit != arguments.expected_commit
        ):
            raise ManifestError("candidate source commit differs from the request")
        if (
            arguments.expected_tag is not None
            and candidate.expected_tag != arguments.expected_tag
        ):
            raise ManifestError("candidate release tag differs from the request")
    except (ManifestError, OSError) as error:
        print(f"candidate verification failed: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "commit": candidate.commit,
                "expected_tag": candidate.expected_tag,
                "version": candidate.version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
