from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools/release"))

import candidate_manifest as candidate_manifest_module  # noqa: E402
import testpypi_replay as testpypi_replay_module  # noqa: E402
from candidate_manifest import (  # noqa: E402
    MANIFEST_FORMAT,
    ManifestError,
    PROFILE_CHECKS,
    file_sha256,
    load_candidate_family,
    require_candidate_profile,
    verify_artifacts,
)
from testpypi_replay import release_files  # noqa: E402


VERSION = "0.1.0"
RAW_VERSION = "0.1.0"
NONCLAIMS = [
    "reproducible-build-certification",
    "artifact-signature",
    "macos-or-windows",
    "abi3",
    "free-threaded-cpython",
    "production-pypi-publication",
]
METADATA = f"""\
Metadata-Version: 2.4
Name: eqiora
Version: {VERSION}
Requires-Python: <3.15,>=3.11
""".encode()


def _write_regular_tar_member(
    archive: tarfile.TarFile, name: str, payload: bytes
) -> None:
    member = tarfile.TarInfo(name)
    member.mode = 0o644
    member.mtime = 0
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _write_sdist(path: Path) -> None:
    prefix = f"eqiora-{VERSION}"
    with tarfile.open(path, mode="w:gz") as archive:
        _write_regular_tar_member(
            archive,
            f"{prefix}/Cargo.toml",
            f'[workspace.package]\nversion = "{RAW_VERSION}"\n'.encode(),
        )
        _write_regular_tar_member(
            archive,
            f"{prefix}/pyproject.toml",
            b'[project]\nname = "eqiora"\ndynamic = ["version"]\n',
        )
        _write_regular_tar_member(archive, f"{prefix}/PKG-INFO", METADATA)


def _write_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("eqiora/__init__.py", b"")
        archive.writestr(f"eqiora-{VERSION}.dist-info/METADATA", METADATA)


def candidate_fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    artifacts = root / "artifacts"
    artifacts.mkdir()
    records: list[dict[str, object]] = []
    sdist = artifacts / f"eqiora-{VERSION}.tar.gz"
    _write_sdist(sdist)
    records.append(
        {
            "filename": sdist.name,
            "kind": "sdist",
            "size": sdist.stat().st_size,
            "sha256": hashlib.sha256(sdist.read_bytes()).hexdigest(),
        }
    )
    for python in ("3.11", "3.12", "3.13", "3.14"):
        compact = python.replace(".", "")
        wheel = artifacts / (
            f"eqiora-{VERSION}-cp{compact}-cp{compact}-"
            "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        )
        _write_wheel(wheel)
        records.append(
            {
                "abi": f"cp{compact}",
                "filename": wheel.name,
                "kind": "wheel",
                "platform": "manylinux_2_17_x86_64",
                "python": python,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "size": wheel.stat().st_size,
            }
        )
    checks = sorted(set().union(*PROFILE_CHECKS.values()))
    document: dict[str, object] = {
        "acceptance": "complete",
        "artifacts": records,
        "build": {
            "dependency_profiles": {
                "numpy_floor": {
                    "observed": "2.1.0",
                    "profile": "cp312:numpy-2.1.0-floor",
                    "python": "3.12",
                    "requirement": "numpy==2.1.0",
                }
            },
            "sdist_rebuilt": True,
            "tools": {
                "cargo": "cargo 1.97.1",
                "maturin": "maturin 1.15.0",
                "mypy": "mypy==2.3.0",
                "pytest": "pytest==9.1.1",
                "rustc": "rustc 1.97.1",
                "twine": "twine==7.0.0",
                "uv": "uv 0.12.1",
            },
            "wheel_family": {
                "abi3": False,
                "implementation": "CPython",
                "ordinary_gil": True,
                "platform": "manylinux_2_17_x86_64",
                "versions": ["3.11", "3.12", "3.13", "3.14"],
            },
        },
        "checks": checks,
        "format": MANIFEST_FORMAT,
        "nonclaims": NONCLAIMS,
        "project": "eqiora",
        "source": {
            "commit": "1" * 40,
            "expected_tag": f"v{VERSION}",
            "tags": [],
            "tree": "clean",
        },
        "version": VERSION,
    }
    manifest = root / f"eqiora-{VERSION}-python-candidate.json"
    manifest.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return manifest, artifacts, document


def _write_manifest(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


class CandidateManifestTests(unittest.TestCase):
    def test_current_manifest_admits_one_exact_family_and_all_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            candidate = load_candidate_family(
                manifest,
                artifacts,
                requested_profiles=tuple(PROFILE_CHECKS),
            )
            verify_artifacts(candidate, artifacts)

        self.assertEqual(candidate.version, VERSION)
        self.assertEqual(candidate.commit, "1" * 40)
        self.assertEqual(len(candidate.artifacts), 5)

    def test_only_current_closed_schema_is_accepted(self) -> None:
        mutations = (
            ("format", lambda document: document.__setitem__("format", "obsolete")),
            ("manifest", lambda document: document.__setitem__("legacy", True)),
            (
                "build",
                lambda document: document["build"].__setitem__(  # type: ignore[union-attr]
                    "host_runtime", {}
                ),
            ),
            (
                "artifact",
                lambda document: document["artifacts"][0].__setitem__(  # type: ignore[index,union-attr]
                    "legacy", True
                ),
            ),
        )
        for location, mutate in mutations:
            with (
                self.subTest(location=location),
                tempfile.TemporaryDirectory() as temporary,
            ):
                manifest, artifacts, document = candidate_fixture(Path(temporary))
                mutate(document)
                _write_manifest(manifest, document)
                with self.assertRaisesRegex(ManifestError, "schema|unsupported"):
                    load_candidate_family(manifest, artifacts)

    def test_retained_cargo_version_and_dynamic_python_version_are_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, document = candidate_fixture(root)
            document["version"] = "0.1.0a2"
            _write_manifest(manifest, document)
            with self.assertRaisesRegex(ManifestError, "retained Cargo authority"):
                load_candidate_family(manifest, artifacts)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, _ = candidate_fixture(root)
            sdist = artifacts / f"eqiora-{VERSION}.tar.gz"
            with tarfile.open(sdist, mode="w:gz") as archive:
                _write_regular_tar_member(
                    archive,
                    f"eqiora-{VERSION}/Cargo.toml",
                    f'[workspace.package]\nversion = "{RAW_VERSION}"\n'.encode(),
                )
                _write_regular_tar_member(
                    archive,
                    f"eqiora-{VERSION}/pyproject.toml",
                    f'[project]\nname = "eqiora"\nversion = "{VERSION}"\n'.encode(),
                )
            with self.assertRaisesRegex(ManifestError, "exactly dynamic"):
                load_candidate_family(manifest, artifacts)

    def test_every_required_profile_fails_when_one_check_is_absent(self) -> None:
        for profile, required in PROFILE_CHECKS.items():
            with (
                self.subTest(profile=profile),
                tempfile.TemporaryDirectory() as temporary,
            ):
                manifest, artifacts, document = candidate_fixture(Path(temporary))
                omitted = min(required)
                document["checks"].remove(omitted)  # type: ignore[union-attr]
                _write_manifest(manifest, document)
                with self.assertRaisesRegex(
                    ManifestError,
                    rf"candidate {profile} profile omits required check {omitted!r}",
                ):
                    load_candidate_family(manifest, artifacts)

    def test_profile_projection_rejects_an_unsuccessful_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            candidate = load_candidate_family(manifest, artifacts)
        failed = min(PROFILE_CHECKS["torch"])
        unsuccessful = replace(candidate, checks=candidate.checks - {failed})
        with self.assertRaisesRegex(ManifestError, "candidate torch profile"):
            require_candidate_profile(unsuccessful, "torch")

    def test_substitution_extra_member_and_manifest_hash_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            candidate = load_candidate_family(manifest, artifacts)
            first = artifacts / candidate.artifacts[0].filename
            first.write_bytes(b"substituted")
            with self.assertRaisesRegex(ManifestError, "size differs"):
                verify_artifacts(candidate, artifacts)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            (artifacts / "unreviewed.whl").write_bytes(b"x")
            with self.assertRaisesRegex(ManifestError, "exact family"):
                load_candidate_family(manifest, artifacts)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            argv = [
                "candidate_manifest.py",
                "--manifest",
                str(manifest),
                "--artifacts",
                str(artifacts),
                "--manifest-sha256",
                "0" * 64,
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(candidate_manifest_module.main(), 2)

    def test_manifest_and_expanded_archive_bounds_are_per_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            with (
                mock.patch.object(
                    candidate_manifest_module,
                    "MANIFEST_BYTES_LIMIT",
                    manifest.stat().st_size - 1,
                ),
                self.assertRaisesRegex(ManifestError, "manifest exceeds"),
            ):
                load_candidate_family(manifest, artifacts)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            with mock.patch.object(
                candidate_manifest_module,
                "ARCHIVE_MEMBER_COUNT_LIMIT",
                3,
            ):
                load_candidate_family(manifest, artifacts)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            with (
                mock.patch.object(
                    candidate_manifest_module,
                    "ARCHIVE_TOTAL_BYTES_LIMIT",
                    1,
                ),
                self.assertRaisesRegex(ManifestError, "expanded bounds"),
            ):
                load_candidate_family(manifest, artifacts)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            with (
                mock.patch.object(
                    candidate_manifest_module,
                    "ARCHIVE_MEMBER_BYTES_LIMIT",
                    1,
                ),
                self.assertRaisesRegex(ManifestError, "expanded bounds"),
            ):
                load_candidate_family(manifest, artifacts)

        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = candidate_fixture(Path(temporary))
            with (
                mock.patch.object(
                    candidate_manifest_module,
                    "ARCHIVE_MEMBER_COUNT_LIMIT",
                    1,
                ),
                self.assertRaisesRegex(ManifestError, "expanded bounds"),
            ):
                load_candidate_family(manifest, artifacts)

    def test_testpypi_metadata_is_bound_to_the_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, document = candidate_fixture(Path(temporary))
            candidate = load_candidate_family(manifest, artifacts)
            payload = {
                "info": {"name": "eqiora", "version": VERSION},
                "urls": [
                    {
                        "digests": {"sha256": record["sha256"]},
                        "filename": record["filename"],
                        "size": record["size"],
                        "url": (
                            f"https://test-files.pythonhosted.org/{record['filename']}"
                        ),
                    }
                    for record in document["artifacts"]  # type: ignore[union-attr]
                ],
            }
            self.assertEqual(
                set(release_files(payload, candidate)),
                {artifact.filename for artifact in candidate.artifacts},
            )
            payload["urls"][0]["digests"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ManifestError, "metadata hash"):
                release_files(payload, candidate)
            payload["urls"][0]["digests"]["sha256"] = candidate.artifacts[0].sha256
            payload["urls"][0]["url"] = "https://example.invalid/file"
            with self.assertRaisesRegex(ManifestError, "unexpected host"):
                release_files(payload, candidate)

    def test_testpypi_replay_uses_the_same_current_manifest_and_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, document = candidate_fixture(root)
            output = root / "replay"
            retained = {path.name: path.read_bytes() for path in artifacts.iterdir()}
            payload = {
                "info": {"name": "eqiora", "version": VERSION},
                "urls": [
                    {
                        "digests": {"sha256": record["sha256"]},
                        "filename": record["filename"],
                        "size": record["size"],
                        "url": (
                            f"https://test-files.pythonhosted.org/{record['filename']}"
                        ),
                    }
                    for record in document["artifacts"]  # type: ignore[union-attr]
                ],
            }

            def download(candidate: object, urls: dict[str, str], target: Path) -> None:
                target.mkdir()
                for filename in urls:
                    (target / filename).write_bytes(retained[filename])
                verify_artifacts(candidate, target)  # type: ignore[arg-type]

            argv = [
                "testpypi_replay.py",
                "--manifest",
                str(manifest),
                "--manifest-sha256",
                file_sha256(manifest),
                "--artifacts",
                str(artifacts),
                "--out",
                str(output),
                "--attempts",
                "1",
                "--wait-seconds",
                "0",
            ]
            with (
                mock.patch.object(
                    testpypi_replay_module, "fetch_json", return_value=payload
                ),
                mock.patch.object(
                    testpypi_replay_module, "download_files", side_effect=download
                ),
                mock.patch.object(sys, "argv", argv),
            ):
                self.assertEqual(testpypi_replay_module.main(), 0)

            self.assertEqual(
                {path.name: path.read_bytes() for path in output.iterdir()}, retained
            )


if __name__ == "__main__":
    unittest.main()
