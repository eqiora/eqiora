from __future__ import annotations

import base64
import builtins
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
import types
import unittest
import warnings
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from packaging.utils import parse_wheel_filename


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools/release"))

import python_candidate as python_candidate_module  # noqa: E402

from python_candidate import (  # noqa: E402
    PYTHON_TEST_FIXTURES,
    PYTHON_TEST_RESOURCES,
    CandidateError,
    DistributionConfig,
    SourceIdentity,
    ensure_exact_uv,
    exact_uv_version,
    inspect_wheel,
    prepare_base_consumer_tree,
    python_distribution_version,
    require_exact_uv,
    require_expected_tag,
    run_public_smoke,
    safe_extract_sdist,
)


EXACT_WHEEL_INTERPRETERS = ("311", "312", "313", "314")
EXACT_PHYSICAL_PLATFORM = "manylinux_2_17_x86_64.manylinux2014_x86_64"
EXACT_WHEEL_PAYLOAD_SHA256 = {
    "311": "7daeaaf4e349a625cde746db0e2f152bf534d15f243d819c95f7534e0bc6e62c",
    "312": "71d661618d9b707f61ac372a0d51047692f4339b639a602bfa5c14ab5e4211d6",
    "313": "d4bd81bf36f65d2f43263a7ce2c7f6ac2263dfd371c8bd0941fe7b57c69e535b",
    "314": "a74b13b0da0238bae641c0f82943f6e33f56f519f79077cbdd8c56e1140b8c69",
}
EXACT_WHEEL_MEMBER = "eqiora-0.1.0.dist-info/WHEEL"
EXACT_RECORD_MEMBER = "eqiora-0.1.0.dist-info/RECORD"
def exact_wheel_name(compact_python: str, *, version: str = "0.1.0") -> str:
    return (
        f"eqiora-{version}-cp{compact_python}-cp{compact_python}-"
        f"{EXACT_PHYSICAL_PLATFORM}.whl"
    )


def exact_wheel_tags(compact_python: str) -> tuple[str, str]:
    prefix = f"cp{compact_python}-cp{compact_python}-"
    return (
        f"{prefix}manylinux_2_17_x86_64",
        f"{prefix}manylinux2014_x86_64",
    )


def maturin_wheel_payload(
    compact_python: str,
    *,
    tags: tuple[str, ...] | None = None,
) -> bytes:
    observed_tags = exact_wheel_tags(compact_python) if tags is None else tags
    tag_lines = "".join(f"Tag: {tag}\n" for tag in observed_tags)
    return (
        "Wheel-Version: 1.0\n"
        "Generator: maturin (1.15.0)\n"
        "Root-Is-Purelib: false\n"
        f"{tag_lines}"
    ).encode("utf-8")


def record_payload_for_wheel(
    wheel_member: str,
    wheel: bytes,
    record_member: str,
) -> bytes:
    digest = base64.urlsafe_b64encode(hashlib.sha256(wheel).digest()).rstrip(b"=")
    return (
        wheel_member.encode("utf-8")
        + b",sha256="
        + digest
        + f",{len(wheel)}\n".encode("ascii")
        + record_member.encode("utf-8")
        + b",,\n"
    )


def maturin_record_payload(compact_python: str) -> bytes:
    return record_payload_for_wheel(
        EXACT_WHEEL_MEMBER,
        maturin_wheel_payload(compact_python),
        EXACT_RECORD_MEMBER,
    )


def write_maturin_wheel(
    path: Path,
    compact_python: str,
    *,
    tags: tuple[str, ...] | None = None,
    members: tuple[tuple[str, bytes], ...] | None = None,
) -> None:
    match = re.fullmatch(r"eqiora-(.+)-cp[0-9]+-cp[0-9]+-.+\.whl", path.name)
    if match is None:
        raise AssertionError(f"synthetic wheel has an invalid filename: {path.name}")
    version = match.group(1)
    entries = members or (
        (
            f"eqiora-{version}.dist-info/WHEEL",
            maturin_wheel_payload(compact_python, tags=tags),
        ),
    )
    if not any(name.endswith(".dist-info/METADATA") for name, _payload in entries):
        entries = (
            *entries,
            (
                f"eqiora-{version}.dist-info/METADATA",
                (
                    "Metadata-Version: 2.4\n"
                    "Name: eqiora\n"
                    f"Version: {version}\n\n"
                ).encode("utf-8"),
            ),
        )
    if not any(name.endswith(".dist-info/RECORD") for name, _payload in entries):
        wheels = tuple(
            (name, payload)
            for name, payload in entries
            if name.endswith(".dist-info/WHEEL")
        )
        if wheels:
            wheel_member, wheel_payload = wheels[0]
            record_member = f"{wheel_member.removesuffix('/WHEEL')}/RECORD"
            record_payload = record_payload_for_wheel(
                wheel_member,
                wheel_payload,
                record_member,
            )
        else:
            record_member = EXACT_RECORD_MEMBER
            record_payload = maturin_record_payload(compact_python)
        entries = (*entries, (record_member, record_payload))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Duplicate name:", category=UserWarning
        )
        with zipfile.ZipFile(path, mode="w") as archive:
            for name, payload in entries:
                member = zipfile.ZipInfo(name)
                member.create_system = 3
                member.external_attr = 0o100644 << 16
                member.compress_type = zipfile.ZIP_STORED
                archive.writestr(member, payload)


def wheel_byte_identity(
    path: Path,
    *,
    version: str = "0.1.0",
) -> tuple[bytes, bytes, bytes]:
    archive_bytes = path.read_bytes()
    with zipfile.ZipFile(path, mode="r") as archive:
        wheel_bytes = archive.read(f"eqiora-{version}.dist-info/WHEEL")
        record_bytes = archive.read(f"eqiora-{version}.dist-info/RECORD")
    return archive_bytes, wheel_bytes, record_bytes


@contextlib.contextmanager
def reject_post_producer_wheel_writes(
    sealed_paths: set[Path],
) -> Iterator[None]:
    real_builtin_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open
    real_os_rename = os.rename
    real_os_replace = os.replace
    real_path_write_bytes = Path.write_bytes
    real_zipfile = zipfile.ZipFile

    def key(value: object) -> Path | None:
        candidate = value
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            candidate = getattr(candidate, "name", None)
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return None
        try:
            return Path(candidate).resolve()
        except (OSError, TypeError, ValueError):
            return None

    def reject(value: object, operation: str) -> None:
        observed = key(value)
        if observed is not None and observed in sealed_paths:
            raise AssertionError(
                f"post-producer wheel rewrite is forbidden: {operation}: {observed}"
            )

    def opens_for_write(mode: object) -> bool:
        return any(token in str(mode) for token in ("w", "a", "x", "+"))

    def guarded_builtin_open(
        file: object, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        if opens_for_write(mode):
            reject(file, f"builtins.open({mode})")
        return real_builtin_open(file, mode, *args, **kwargs)

    def guarded_io_open(
        file: object, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        if opens_for_write(mode):
            reject(file, f"io.open({mode})")
        return real_io_open(file, mode, *args, **kwargs)

    def guarded_os_open(
        file: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & write_flags:
            reject(file, f"os.open({flags})")
        return real_os_open(file, flags, *args, **kwargs)

    def guarded_write_bytes(path: Path, data: bytes) -> int:
        reject(path, "Path.write_bytes")
        return real_path_write_bytes(path, data)

    def guarded_rename(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        reject(source, "os.rename(source)")
        reject(destination, "os.rename(destination)")
        real_os_rename(source, destination, *args, **kwargs)

    def guarded_replace(
        source: object,
        destination: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        reject(source, "os.replace(source)")
        reject(destination, "os.replace(destination)")
        real_os_replace(source, destination, *args, **kwargs)

    def guarded_zipfile(
        file: object,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> zipfile.ZipFile:
        if opens_for_write(mode):
            reject(file, f"zipfile.ZipFile({mode})")
        return real_zipfile(file, mode, *args, **kwargs)

    with (
        mock.patch.object(builtins, "open", side_effect=guarded_builtin_open),
        mock.patch.object(io, "open", side_effect=guarded_io_open),
        mock.patch.object(os, "open", side_effect=guarded_os_open),
        mock.patch.object(os, "rename", side_effect=guarded_rename),
        mock.patch.object(os, "replace", side_effect=guarded_replace),
        mock.patch.object(Path, "write_bytes", new=guarded_write_bytes),
        mock.patch.object(zipfile, "ZipFile", side_effect=guarded_zipfile),
    ):
        yield


class PythonCandidateTests(unittest.TestCase):
    def config(self) -> DistributionConfig:
        return DistributionConfig(
            cargo_version="0.1.0",
            interpreters=("3.11", "3.12", "3.13", "3.14"),
            wheel_platform="manylinux_2_17_x86_64",
            extras_interpreter="3.13",
            numpy_floor_interpreter="3.12",
            numpy_floor="numpy==2.1.0",
            uv="uv==0.12.1",
            maturin="maturin==1.15.0",
            pytest="pytest==9.1.1",
            mypy="mypy==2.3.0",
            twine="twine==6.2.0",
            torch="torch==2.14.0",
            jax=("jax==0.11.0", "jaxlib==0.11.0"),
            matplotlib="matplotlib==3.11.1",
            rust="1.89",
        )

    def test_release_identity_has_one_python_version_and_exact_tag(self) -> None:
        self.assertEqual(python_distribution_version("0.1.0"), "0.1.0")
        self.assertEqual(self.config().expected_tag, "v0.1.0")
        require_expected_tag(
            SourceIdentity(commit="0" * 40, tags=("v0.1.0",)),
            self.config().expected_tag,
        )
        with self.assertRaisesRegex(
            CandidateError,
            "requires exact tag v0.1.0",
        ):
            require_expected_tag(
                SourceIdentity(commit="0" * 40, tags=("v0.1.1",)),
                self.config().expected_tag,
            )
        for rejected in (
            "0.1.0-alpha.1",
            "0.1.0-beta.1",
            "0.1.0-rc.1",
            "0.1.0+local",
            "00.1.0",
            "0.01.0",
            "0.1.00",
        ):
            with self.assertRaises(CandidateError, msg=rejected):
                python_distribution_version(rejected)

    def test_candidate_producer_has_no_handwritten_release_identity(self) -> None:
        source = (
            REPOSITORY_ROOT / "tools/release/python_candidate.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('config.python_version != "0.1.0"', source)
        self.assertNotIn('"v0.1.0"', source)

    def test_standard_release_tools_group_is_the_only_uv_version_source(self) -> None:
        document = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        config = python_candidate_module.load_config()

        self.assertNotIn("uv", document["tool"]["eqiora-distribution"])
        self.assertEqual(
            document["dependency-groups"]["release-tools"],
            [config.twine, config.uv],
        )


    @mock.patch("python_candidate.tool_version", return_value="uv 0.12.1")
    def test_release_tool_requires_the_exact_reviewed_uv(
        self,
        version: mock.Mock,
    ) -> None:
        require_exact_uv("/usr/bin/uv", "uv==0.12.1")
        version.assert_called_once_with(["/usr/bin/uv", "--version"])

        version.return_value = "uv 0.12.0"
        with self.assertRaisesRegex(CandidateError, "requires uv 0.12.1"):
            require_exact_uv("/usr/bin/uv", "uv==0.12.1")

        for malformed in ("uv>=0.12.1", "uv==0.12", "uv==../0.12.1"):
            with self.assertRaisesRegex(CandidateError, "requirement is malformed"):
                exact_uv_version(malformed)

    def test_exact_uv_is_installed_once_in_a_versioned_cache(self) -> None:
        calls: list[list[str]] = []

        def checked(argv: list[str], **_kwargs: object) -> str:
            calls.append(argv)
            if argv[1:3] == ["-m", "venv"]:
                (_virtual_environment := Path(argv[-1]) / "bin").mkdir(parents=True)
                (_virtual_environment / "python").touch()
                return ""
            if argv[1:4] == ["-m", "pip", "install"]:
                Path(argv[0]).with_name("uv").touch()
                return ""
            if argv[-1:] == ["--version"]:
                return "uv 0.12.1"
            self.fail(f"unexpected command: {argv}")

        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            cache = Path(temporary) / "tools"
            with mock.patch.object(
                python_candidate_module,
                "checked_run",
                side_effect=checked,
            ):
                first = ensure_exact_uv("uv==0.12.1", cache_root=cache)
                install_call_count = len(calls)
                second = ensure_exact_uv("uv==0.12.1", cache_root=cache)

        self.assertEqual(first, second)
        self.assertEqual(
            Path(first),
            cache.resolve() / "uv" / "0.12.1" / "bin" / "uv",
        )
        self.assertEqual(install_call_count, 4)
        self.assertEqual(len(calls), 5)
        self.assertEqual(
            calls[1][-3:],
            [
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "uv==0.12.1",
            ],
        )

    @mock.patch("python_candidate.checked_run")
    def test_public_smoke_uses_the_installed_interpreter_in_isolated_mode(
        self,
        checked: mock.Mock,
    ) -> None:
        run_public_smoke(
            python=Path("/candidate/bin/python"),
            extracted=Path("/sdist"),
            run_root=Path("/consumer"),
            expected_version="0.1.0",
            profile="base",
        )

        checked.assert_called_once_with(
            [
                "/candidate/bin/python",
                "-I",
                "/sdist/tools/release/python_public_smoke.py",
                "--expected-version",
                "0.1.0",
                "--profile",
                "base",
            ],
            cwd=Path("/consumer"),
        )

    def test_wheel_contract_accepts_complete_typed_optional_metadata(self) -> None:
        license_bytes = b"license\n"
        notice_bytes = b"notice\n"
        metadata = b"""\
Metadata-Version: 2.4
Name: eqiora
Version: 0.1.0
Requires-Python: <3.15,>=3.11
License-Expression: Apache-2.0
License-File: LICENSE
License-File: NOTICE
Provides-Extra: jax
Provides-Extra: gmsh
Provides-Extra: matplotlib
Provides-Extra: torch
Provides-Extra: viewer
Requires-Dist: numpy<3,>=2.1
Requires-Dist: gmsh==4.15.2 ; extra == 'gmsh'
Requires-Dist: torch>=2.14,<2.15; extra == "torch"
Requires-Dist: jax==0.11.0; python_version >= "3.12" and extra == "jax"
Requires-Dist: jaxlib==0.11.0; python_version >= "3.12" and extra == "jax"
Requires-Dist: matplotlib>=3.10,<3.12; extra == "matplotlib"
Requires-Dist: anywidget==0.11.0; extra == "viewer"

typed candidate
"""
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / exact_wheel_name("313")
            dist_info = "eqiora-0.1.0.dist-info/"
            with zipfile.ZipFile(wheel, mode="w") as archive:
                for name in (
                    "eqiora/__init__.py",
                    "eqiora/__init__.pyi",
                    "eqiora/diff.pyi",
                    "eqiora/fsi.pyi",
                    "eqiora/jax.pyi",
                    "eqiora/matplotlib.pyi",
                    "eqiora/viewer.pyi",
                    "eqiora/solid.pyi",
                    "eqiora/torch.pyi",
                    "eqiora/py.typed",
                    "eqiora/_viewer/THIRD_PARTY_NOTICES.txt",
                    "eqiora/_viewer/static/viewer.css",
                    "eqiora/_viewer/static/viewer.mjs",
                    "eqiora/examples/steady-flow-past-cylinder.eqi",
                    "eqiora/examples/transient-flow-past-cylinder.eqi",
                    "eqiora/examples/mixed-boundary-elasticity.eqi",
                    "eqiora/examples/fixed-reference-fsi.eqi",
                    "eqiora/_eqiora.cpython-313-x86_64-linux-gnu.so",
                    f"{dist_info}sboms/eqiora-python.cyclonedx.json",
                ):
                    archive.writestr(name, b"")
                archive.writestr(f"{dist_info}METADATA", metadata)
                archive.writestr(
                    f"{dist_info}WHEEL",
                    maturin_wheel_payload("313"),
                )
                archive.writestr(f"{dist_info}licenses/LICENSE", license_bytes)
                archive.writestr(f"{dist_info}licenses/NOTICE", notice_bytes)

            version, record = inspect_wheel(
                wheel,
                python_version="3.13",
                config=self.config(),
                license_bytes=license_bytes,
                notice_bytes=notice_bytes,
            )

        self.assertEqual(version, "0.1.0")
        self.assertEqual(record["filename"], exact_wheel_name("313"))
        self.assertEqual(record["python"], "3.13")
        self.assertEqual(record["platform"], "manylinux_2_17_x86_64")
        self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")

    def test_wheel_contract_rejects_framework_as_a_base_dependency(self) -> None:
        license_bytes = b"license\n"
        notice_bytes = b"notice\n"
        metadata = b"""\
Metadata-Version: 2.4
Name: eqiora
Version: 0.1.0
Requires-Python: <3.15,>=3.11
License-Expression: Apache-2.0
License-File: LICENSE
License-File: NOTICE
Provides-Extra: jax
Provides-Extra: gmsh
Provides-Extra: matplotlib
Provides-Extra: torch
Provides-Extra: viewer
Requires-Dist: numpy<3,>=2.1
Requires-Dist: gmsh==4.15.2; extra == "gmsh"
Requires-Dist: torch>=2.14,<2.15
Requires-Dist: jax==0.11.0; extra == "jax"
Requires-Dist: jaxlib==0.11.0; extra == "jax"
Requires-Dist: matplotlib>=3.10,<3.12; extra == "matplotlib"
Requires-Dist: anywidget==0.11.0; extra == "viewer"

invalid candidate
"""
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / exact_wheel_name("313")
            dist_info = "eqiora-0.1.0.dist-info/"
            with zipfile.ZipFile(wheel, mode="w") as archive:
                for name in (
                    "eqiora/__init__.py",
                    "eqiora/__init__.pyi",
                    "eqiora/diff.pyi",
                    "eqiora/fsi.pyi",
                    "eqiora/jax.pyi",
                    "eqiora/matplotlib.pyi",
                    "eqiora/viewer.pyi",
                    "eqiora/solid.pyi",
                    "eqiora/torch.pyi",
                    "eqiora/py.typed",
                    "eqiora/_viewer/THIRD_PARTY_NOTICES.txt",
                    "eqiora/_viewer/static/viewer.css",
                    "eqiora/_viewer/static/viewer.mjs",
                    "eqiora/examples/steady-flow-past-cylinder.eqi",
                    "eqiora/examples/transient-flow-past-cylinder.eqi",
                    "eqiora/examples/mixed-boundary-elasticity.eqi",
                    "eqiora/examples/fixed-reference-fsi.eqi",
                    "eqiora/_eqiora.cpython-313-x86_64-linux-gnu.so",
                    f"{dist_info}sboms/eqiora-python.cyclonedx.json",
                ):
                    archive.writestr(name, b"")
                archive.writestr(f"{dist_info}METADATA", metadata)
                archive.writestr(
                    f"{dist_info}WHEEL",
                    maturin_wheel_payload("313"),
                )
                archive.writestr(f"{dist_info}licenses/LICENSE", license_bytes)
                archive.writestr(f"{dist_info}licenses/NOTICE", notice_bytes)

            with self.assertRaisesRegex(
                CandidateError,
                "torch must remain an optional-extra dependency",
            ):
                inspect_wheel(
                    wheel,
                    python_version="3.13",
                    config=self.config(),
                    license_bytes=license_bytes,
                    notice_bytes=notice_bytes,
                )


    def test_wheel_build_retains_exact_maturin_compressed_names_and_bytes(
        self,
    ) -> None:
        config = self.config()
        interpreters = {
            version: f"/managed/python{version}" for version in config.interpreters
        }
        commands: list[list[str]] = []
        producer_hashes: dict[str, str] = {}
        build_return_hashes: dict[str, str] = {}
        admission_hashes: dict[str, str] = {}
        producer_identities: dict[str, tuple[bytes, bytes, bytes]] = {}
        build_return_identities: dict[str, tuple[bytes, bytes, bytes]] = {}
        admission_identities: dict[str, tuple[bytes, bytes, bytes]] = {}
        sealed_wheel_paths: set[Path] = set()
        returned_wheel_names: tuple[str, ...] = ()
        expected_inventory: tuple[dict[str, object], ...] = ()
        wheel_contract_observations: dict[
            str, tuple[int, tuple[str, ...], set[str], str]
        ] = {}
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            output.mkdir()
            scratch = root / "scratch"
            extracted = root / "source" / f"eqiora-{config.python_version}"
            extracted.mkdir(parents=True)

            def checked_run(arguments: list[str], **_kwargs: object) -> str:
                commands.append(arguments)
                if "sdist" in arguments:
                    (output / f"eqiora-{config.python_version}.tar.gz").write_bytes(
                        b"sdist"
                    )
                    return ""
                compatibility = arguments[arguments.index("--compatibility") + 1]
                interpreter = arguments[arguments.index("--interpreter") + 1]
                version = next(
                    name for name, path in interpreters.items() if path == interpreter
                )
                compact = version.replace(".", "")
                self.assertEqual(compatibility, "manylinux_2_17")
                wheel = output / exact_wheel_name(
                    compact,
                    version=config.python_version,
                )
                write_maturin_wheel(wheel, compact)
                producer_identities[version] = wheel_byte_identity(wheel)
                producer_hashes[version] = hashlib.sha256(
                    producer_identities[version][0]
                ).hexdigest()
                sealed_wheel_paths.add(wheel.resolve())
                return ""

            with (
                reject_post_producer_wheel_writes(sealed_wheel_paths),
                mock.patch.object(
                    python_candidate_module, "checked_run", side_effect=checked_run
                ),
                mock.patch.object(
                    python_candidate_module,
                    "safe_extract_sdist",
                    return_value=extracted,
                ),
                mock.patch.object(
                    python_candidate_module,
                    "cargo_workspace_version",
                    return_value=config.cargo_version,
                ),
                mock.patch.object(
                    Path,
                    "rename",
                    side_effect=AssertionError("wheel rename is forbidden"),
                ),
                mock.patch.object(
                    Path,
                    "replace",
                    side_effect=AssertionError("wheel replace is forbidden"),
                ),
                mock.patch.object(
                    Path,
                    "hardlink_to",
                    side_effect=AssertionError("wheel link is forbidden"),
                ),
                mock.patch.object(
                    Path,
                    "symlink_to",
                    side_effect=AssertionError("wheel link is forbidden"),
                ),
                mock.patch.object(
                    os,
                    "rename",
                    side_effect=AssertionError("wheel rename is forbidden"),
                ),
                mock.patch.object(
                    os,
                    "replace",
                    side_effect=AssertionError("wheel replace is forbidden"),
                ),
                mock.patch.object(
                    os,
                    "link",
                    side_effect=AssertionError("wheel link is forbidden"),
                ),
                mock.patch.object(
                    os,
                    "symlink",
                    side_effect=AssertionError("wheel link is forbidden"),
                ),
                mock.patch(
                    "shutil.copy",
                    side_effect=AssertionError("wheel copy is forbidden"),
                ),
                mock.patch(
                    "shutil.copy2",
                    side_effect=AssertionError("wheel copy is forbidden"),
                ),
                mock.patch(
                    "shutil.copyfile",
                    side_effect=AssertionError("wheel copy is forbidden"),
                ),
                mock.patch(
                    "shutil.copytree",
                    side_effect=AssertionError("wheel copy is forbidden"),
                ),
            ):
                _sdist, wheels, _extracted = python_candidate_module.build_artifacts(
                    output=output,
                    scratch=scratch,
                    config=config,
                    uv="/managed/uv",
                    interpreters=interpreters,
                )
                build_return_identities = {
                    version: wheel_byte_identity(wheel)
                    for version, wheel in wheels.items()
                }
                build_return_hashes = {
                    version: hashlib.sha256(identity[0]).hexdigest()
                    for version, identity in build_return_identities.items()
                }
                executor = importlib.import_module("python_candidate_family")
                admitted = executor.admit_candidate_family(output)
                admitted_inventory = admitted.inventory
                expected_inventory = executor.family_inventory(output)
                returned_wheel_names = tuple(path.name for path in wheels.values())
                admission_identities = {
                    version: wheel_byte_identity(wheel)
                    for version, wheel in wheels.items()
                }
                admission_hashes = {
                    version: hashlib.sha256(identity[0]).hexdigest()
                    for version, identity in admission_identities.items()
                }
                for version, wheel in wheels.items():
                    compact = version.replace(".", "")
                    _name, _parsed_version, _build, filename_tags = (
                        parse_wheel_filename(wheel.name)
                    )
                    payload = admission_identities[version][1]
                    internal_tags = tuple(
                        line.removeprefix("Tag: ")
                        for line in payload.decode("utf-8").splitlines()
                        if line.startswith("Tag: ")
                    )
                    wheel_contract_observations[version] = (
                        len(payload),
                        internal_tags,
                        {str(tag) for tag in filename_tags},
                        hashlib.sha256(payload).hexdigest(),
                    )

        wheel_commands = [arguments for arguments in commands if "build" in arguments]
        self.assertEqual(len(wheel_commands), 4)
        for version, arguments in zip(config.interpreters, wheel_commands, strict=True):
            with self.subTest(interpreter=version):
                build_index = arguments.index("build")
                target_dir = str(scratch / "cargo-target")
                self.assertEqual(
                    arguments[:build_index],
                    [
                        "/managed/uv",
                        "tool",
                        "run",
                        "--from",
                        "maturin[zig]==1.15.0",
                        "maturin",
                    ],
                )
                self.assertEqual(
                    arguments[build_index:],
                    [
                        "build",
                        "--release",
                        "--zig",
                        "--compatibility",
                        "manylinux_2_17",
                        "--auditwheel",
                        "check",
                        "--interpreter",
                        interpreters[version],
                        "--target-dir",
                        target_dir,
                        "--out",
                        str(output),
                    ],
                )
                self.assertNotIn("manylinux2014", arguments)
                self.assertEqual(Path(target_dir), scratch / "cargo-target")
        self.assertEqual(
            returned_wheel_names,
            tuple(
                exact_wheel_name(
                    version.replace(".", ""),
                    version=config.python_version,
                )
                for version in config.interpreters
            ),
        )
        self.assertEqual(producer_hashes, build_return_hashes)
        self.assertEqual(build_return_hashes, admission_hashes)
        self.assertEqual(producer_identities, build_return_identities)
        self.assertEqual(build_return_identities, admission_identities)
        self.assertEqual(
            admitted_inventory,
            expected_inventory,
        )
        for version in config.interpreters:
            compact = version.replace(".", "")
            (
                payload_size,
                internal_tags,
                filename_tags,
                payload_sha256,
            ) = wheel_contract_observations[version]
            self.assertEqual(payload_size, 147)
            self.assertEqual(internal_tags, exact_wheel_tags(compact))
            self.assertEqual(
                filename_tags,
                set(internal_tags),
            )
            self.assertEqual(
                payload_sha256,
                EXACT_WHEEL_PAYLOAD_SHA256[compact],
            )
            self.assertEqual(
                admission_identities[version][2],
                maturin_record_payload(compact),
            )

    def test_post_producer_observer_rejects_byte_identical_wheel_rewrites(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            wheel = Path(temporary) / exact_wheel_name("311")
            write_maturin_wheel(wheel, "311")
            expected = wheel_byte_identity(wheel)
            sealed = {wheel.resolve()}
            rename_source = Path(temporary) / "byte-identical-rename-source.whl"
            replace_source = Path(temporary) / "byte-identical-replace-source.whl"
            rename_source.write_bytes(expected[0])
            replace_source.write_bytes(expected[0])

            def write_with_builtin_open() -> None:
                with builtins.open(wheel, mode="r+b") as destination:
                    destination.write(expected[0])

            def write_with_io_open() -> None:
                with io.open(wheel, mode="r+b") as destination:
                    destination.write(expected[0])

            def write_with_os_open() -> None:
                descriptor = os.open(wheel, os.O_WRONLY)
                os.close(descriptor)

            def write_with_zipfile() -> None:
                with zipfile.ZipFile(wheel, mode="a"):
                    pass

            rewrites: dict[str, Callable[[], object]] = {
                "Path.write_bytes": lambda: wheel.write_bytes(expected[0]),
                "builtins.open": write_with_builtin_open,
                "io.open": write_with_io_open,
                "os.open": write_with_os_open,
                "os.rename": lambda: os.rename(rename_source, wheel),
                "os.replace": lambda: os.replace(replace_source, wheel),
                "zipfile.ZipFile": write_with_zipfile,
            }
            for operation, rewrite in rewrites.items():
                with (
                    self.subTest(operation=operation),
                    reject_post_producer_wheel_writes(sealed),
                ):
                    with self.assertRaisesRegex(
                        AssertionError,
                        "post-producer wheel rewrite is forbidden",
                    ):
                        rewrite()
                self.assertEqual(wheel_byte_identity(wheel), expected)

    def test_producer_rejects_nonexact_physical_name_before_twine_or_rewrite(
        self,
    ) -> None:
        config = self.config()
        interpreters = {
            version: f"/managed/python{version}" for version in config.interpreters
        }
        mutant_names = {
            "old-canonical-only-optional-alias": (
                "eqiora-0.1.0-cp311-cp311-manylinux_2_17_x86_64.whl"
            ),
            "alias-first": (
                "eqiora-0.1.0-cp311-cp311-"
                "manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
            ),
            "broadened-dotted-suffix": (
                "eqiora-0.1.0-cp311-cp311-"
                f"{EXACT_PHYSICAL_PLATFORM}.manylinux_2_28_x86_64.whl"
            ),
        }
        for mutant, filename in mutant_names.items():
            with (
                self.subTest(mutant=mutant),
                tempfile.TemporaryDirectory(dir=Path.home()) as temporary,
            ):
                root = Path(temporary)
                output = root / "artifacts"
                output.mkdir()
                scratch = root / "scratch"
                extracted = root / "source" / f"eqiora-{config.python_version}"
                extracted.mkdir(parents=True)
                commands: list[list[str]] = []

                def checked_run(arguments: list[str], **_kwargs: object) -> str:
                    commands.append(arguments)
                    if "sdist" in arguments:
                        (output / f"eqiora-{config.python_version}.tar.gz").write_bytes(
                            b"sdist"
                        )
                    elif "build" in arguments:
                        write_maturin_wheel(output / filename, "311")
                    return ""

                with (
                    mock.patch.object(
                        python_candidate_module,
                        "checked_run",
                        side_effect=checked_run,
                    ),
                    mock.patch.object(
                        python_candidate_module,
                        "safe_extract_sdist",
                        return_value=extracted,
                    ),
                    mock.patch.object(
                        python_candidate_module,
                        "cargo_workspace_version",
                        return_value=config.cargo_version,
                    ),
                    mock.patch.object(
                        Path,
                        "rename",
                        side_effect=AssertionError("rename cannot repair the mutant"),
                    ),
                    mock.patch.object(
                        Path,
                        "replace",
                        side_effect=AssertionError("replace cannot repair the mutant"),
                    ),
                    mock.patch.object(
                        os,
                        "rename",
                        side_effect=AssertionError("rename cannot repair the mutant"),
                    ),
                    mock.patch.object(
                        os,
                        "replace",
                        side_effect=AssertionError("replace cannot repair the mutant"),
                    ),
                    mock.patch.object(
                        os,
                        "link",
                        side_effect=AssertionError("link cannot repair the mutant"),
                    ),
                    mock.patch(
                        "shutil.copy",
                        side_effect=AssertionError("copy cannot repair the mutant"),
                    ),
                    mock.patch(
                        "shutil.copy2",
                        side_effect=AssertionError("copy cannot repair the mutant"),
                    ),
                    mock.patch(
                        "shutil.copyfile",
                        side_effect=AssertionError("copy cannot repair the mutant"),
                    ),
                ):
                    with self.assertRaises(CandidateError):
                        python_candidate_module.build_artifacts(
                            output=output,
                            scratch=scratch,
                            config=config,
                            uv="/managed/uv",
                            interpreters=interpreters,
                        )

                wheel_builds = [call for call in commands if "build" in call]
                twine_calls = [call for call in commands if "twine" in call]
                self.assertEqual(len(wheel_builds), 1)
                self.assertEqual(twine_calls, [])
                self.assertEqual(
                    {path.name for path in output.iterdir()},
                    {f"eqiora-{config.python_version}.tar.gz", filename},
                )

    def test_producer_output_collision_fails_without_overwrite_or_cleanup(self) -> None:
        config = self.config()
        collisions = (
            "eqiora-0.1.0-cp311-cp311-manylinux_2_17_x86_64.whl",
            exact_wheel_name("311"),
            ".eqiora-0.1.0-cp311.partial.whl",
        )
        for filename in collisions:
            with (
                self.subTest(filename=filename),
                tempfile.TemporaryDirectory(dir=Path.home()) as temporary,
            ):
                root = Path(temporary)
                output = root / "artifacts"
                output.mkdir()
                collision = output / filename
                collision.write_bytes(b"pre-existing bytes")
                checked = mock.Mock()
                with mock.patch.object(
                    python_candidate_module,
                    "checked_run",
                    checked,
                ):
                    with self.assertRaises(CandidateError):
                        python_candidate_module.build_artifacts(
                            output=output,
                            scratch=root / "scratch",
                            config=config,
                            uv="/managed/uv",
                            interpreters={
                                version: f"/managed/python{version}"
                                for version in config.interpreters
                            },
                        )

                checked.assert_not_called()
                self.assertEqual(collision.read_bytes(), b"pre-existing bytes")
                self.assertEqual(tuple(output.iterdir()), (collision,))


    def test_sdist_extraction_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "invalid.tar.gz"
            with tarfile.open(archive, mode="w:gz") as destination:
                member = tarfile.TarInfo("../escape")
                payload = b"not allowed"
                member.size = len(payload)
                destination.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(CandidateError, "escapes its root"):
                safe_extract_sdist(archive, root / "extract")

    def test_consumer_tree_preserves_repository_relative_fixture_paths(self) -> None:
        # test_vertical_slice reads the canonical model schema during collection.
        self.assertIn(
            Path("crates/eqiora-api/schemas/compile-v2.schema.json"),
            PYTHON_TEST_RESOURCES,
        )
        # test_coupled_scalar_example executes this checked-in public example.
        self.assertIn(
            Path("examples/python/coupled_scalar.py"), PYTHON_TEST_RESOURCES
        )
        # Controls tests use the maintained adapter, but load the package from the wheel.
        self.assertIn(
            Path("examples/standard-sampled-components/src/main.eqi"),
            PYTHON_TEST_RESOURCES,
        )
        self.assertIn(
            Path("crates/eqiora-api/packages/Eqiora.Controls.Sampled/src/sampled.eqi"),
            PYTHON_TEST_RESOURCES,
        )
        # The installed Reference test executes displayed Python and its sources.
        reference = Path("docs/site/src/content/docs/reference")
        for relative in (
            reference / "standard-packages/continuum.mdx",
            reference / "standard-packages/controls.mdx",
            reference / "language/_examples/clocked.eqi",
            Path("crates/eqiora-api/packages/Eqiora.Fluid.InertialStokes/src/inertial_stokes.eqi"),
        ):
            self.assertIn(relative, PYTHON_TEST_RESOURCES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extracted = root / "source"
            run_root = root / "consumer"
            run_root.mkdir()
            # Derived from the constant rather than restated: a copy of the
            # fixture list here would silently stop covering a fixture added
            # to `PYTHON_TEST_FIXTURES`, which is the drift this test exists
            # to catch.
            files = (
                "bindings/python/tests/test_vertical_slice.py",
                "bindings/python/typecheck/base.py",
                *(str(fixture / "payload.json") for fixture in PYTHON_TEST_FIXTURES),
                *(str(resource) for resource in PYTHON_TEST_RESOURCES),
            )
            self.assertGreaterEqual(len(files), 4)
            for relative in files:
                path = extracted / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")

            controls_manifest = Path("crates/eqiora-api/packages/Eqiora.Controls.Sampled/package.json")
            (extracted / controls_manifest).write_text("{}", encoding="utf-8")

            tests, typecheck = prepare_base_consumer_tree(extracted, run_root)

            self.assertEqual(tests, run_root / "bindings/python/tests")
            self.assertEqual(typecheck, run_root / "bindings/python/typecheck")
            test_path = tests / "test_vertical_slice.py"
            self.assertEqual(test_path.parents[3], run_root)
            for relative in files:
                self.assertTrue((run_root / relative).is_file())
            # The standalone source fixture cannot replace the installed bundle.
            self.assertFalse((run_root / controls_manifest).exists())


class CandidateProfileFanoutContractTests(unittest.TestCase):
    COMPLETE_NAMES = (
        "base-3.11",
        "base-3.12",
        "base-3.13",
        "base-3.14",
        "numpy-floor-3.12",
        "generated-public-api",
        "torch-3.13",
        "jax-3.13",
        "matplotlib-3.13",
        "typing-3.13",
    )

    @staticmethod
    def profiles_module() -> object:
        # Keep the rest of this file collectable before the new private module
        # exists; the focused tests still fail at their exact missing seam.
        return importlib.import_module("python_candidate_profiles")

    @staticmethod
    def can_overlap(left: object, right: object) -> bool:
        return (
            left.cpu_slots + right.cpu_slots <= 2
            and left.memory_mib + right.memory_mib <= 4096
            and left.gpu_slots + right.gpu_slots <= 0
            and set(left.locks).isdisjoint(right.locks)
        )

    def test_complete_and_development_plans_are_exact_and_resource_admitted(
        self,
    ) -> None:
        profiles = self.profiles_module()
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            complete = profiles.build_profile_plan(
                Path(temporary), self.config(), skip_extras=False
            )
            development = profiles.build_profile_plan(
                Path(temporary) / "development",
                self.config(),
                skip_extras=True,
            )

        self.assertEqual(profiles.COMPLETE_PROFILE_NAMES, self.COMPLETE_NAMES)
        self.assertEqual(
            profiles.DEVELOPMENT_PROFILE_NAMES,
            self.COMPLETE_NAMES[:6],
        )
        self.assertEqual(tuple(item.name for item in complete), self.COMPLETE_NAMES)
        self.assertEqual(
            tuple(item.name for item in development), self.COMPLETE_NAMES[:6]
        )

        by_name = {item.name: item.resources for item in complete}
        for request in by_name.values():
            self.assertLessEqual(request.cpu_slots, 2)
            self.assertLessEqual(request.memory_mib, 4096)
            self.assertEqual(request.gpu_slots, 0)

        heavy = ("torch-3.13", "jax-3.13", "typing-3.13")
        for index, name in enumerate(heavy):
            for other in heavy[index + 1 :]:
                self.assertFalse(
                    self.can_overlap(by_name[name], by_name[other]),
                    f"heavy profiles {name} and {other} were jointly admitted",
                )
            self.assertTrue(
                self.can_overlap(by_name[name], by_name["matplotlib-3.13"]),
                f"heavy profile {name} cannot overlap a fitting light profile",
            )
        self.assertTrue(self.can_overlap(by_name["base-3.11"], by_name["base-3.12"]))

    def test_base_profile_dispatch_uses_only_profile_inputs(self) -> None:
        profiles = self.profiles_module()
        config = self.config()
        base = mock.create_autospec(
            python_candidate_module.run_base_profile,
            return_value=["check:base-3.11"],
        )

        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary)
            workspace = profiles.build_profile_plan(
                root, config, skip_extras=False
            )[0]
            extracted = root / "source"
            wheel = root / "candidate-cp311.whl"
            with mock.patch.object(
                python_candidate_module,
                "run_base_profile",
                new=base,
            ):
                profile_receipt = python_candidate_module.execute_profile(
                    workspace,
                    uv="/reviewed/uv",
                    config=config,
                    wheels={"3.11": wheel},
                    extracted=extracted,
                    interpreters={"3.11": "/reviewed/python3.11"},
                )

        self.assertEqual(profile_receipt.checks, ("check:base-3.11",))
        base.assert_called_once_with(
            uv="/reviewed/uv",
            interpreter="/reviewed/python3.11",
            python_version="3.11",
            wheel=wheel,
            extracted=extracted,
            workspace=workspace,
            config=config,
        )

    def test_base_profile_receipts_gmsh_only_after_base_checks_and_mesh_tests(
        self,
    ) -> None:
        profiles = self.profiles_module()
        config = self.config()
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary)
            workspace = profiles.build_profile_plan(
                root, config, skip_extras=False
            )[0]
            wheel = root / "eqiora-cp311.whl"
            extracted = root / "source"
            python = workspace.environment / "bin/python"
            events = mock.Mock()
            install = events.install_environment
            install.return_value = python
            run = events.run
            run.return_value = ""
            public_smoke = events.run_public_smoke

            with (
                mock.patch.object(profiles, "install_environment", new=install),
                mock.patch.object(
                    profiles,
                    "prepare_base_consumer_tree",
                    return_value=(
                        workspace.consumer / "tests",
                        workspace.consumer / "typecheck",
                    ),
                ),
                mock.patch.object(
                    profiles, "prepare_exact_cylinder_demo_consumer"
                ),
                mock.patch.object(
                    profiles, "prepare_mixed_boundary_elasticity_demo_consumer"
                ),
                mock.patch.object(profiles, "assert_installed_origin"),
                mock.patch.object(profiles, "assert_matplotlib_is_optional"),
                mock.patch.object(profiles, "run_public_smoke", new=public_smoke),
            ):
                checks = profiles.run_base_profile(
                    uv="/reviewed/uv",
                    interpreter="/reviewed/python3.11",
                    python_version="3.11",
                    wheel=wheel,
                    extracted=extracted,
                    workspace=workspace,
                    config=config,
                    run=run,
                )

        tests = workspace.consumer / "tests"
        typecheck = workspace.consumer / "typecheck"
        gmsh_tests = tuple(
            tests / name
            for name in (
                "test_gmsh_meshing.py",
                "test_exact_cylinder_stokes_result.py",
            )
        )
        gmsh_path = str(python.parent)
        if inherited_path := os.environ.get("PATH"):
            gmsh_path = os.pathsep.join((gmsh_path, inherited_path))
        self.assertEqual(
            checks,
            [
                "cp311:installed-wheel",
                "cp311:base-and-numpy",
                "cp311:packaged-exact-cylinder-model-demo",
                "cp311:packaged-mixed-boundary-elasticity-demo",
                "cp311:async-and-cancellation",
                "cp311:strict-base-typing",
                "cp311:public-smoke-base",
                "cp311:matplotlib-free-base",
            ],
        )
        self.assertEqual(
            events.mock_calls,
            [
                mock.call.install_environment(
                    uv="/reviewed/uv",
                    interpreter="/reviewed/python3.11",
                    environment=workspace.environment,
                    requirements=[str(wheel), config.pytest, config.mypy],
                    run=run,
                ),
                mock.call.run(
                    [
                        str(python),
                        "-I",
                        "-m",
                        "pytest",
                        "-q",
                        str(tests),
                        "--ignore",
                        str(gmsh_tests[0]),
                        "--ignore",
                        str(gmsh_tests[1]),
                    ],
                    cwd=workspace.consumer,
                ),
                mock.call.run(
                    [
                        str(python),
                        "-I",
                        "-m",
                        "mypy",
                        "--strict",
                        str(typecheck / "base.py"),
                    ],
                    cwd=workspace.consumer,
                ),
                mock.call.run_public_smoke(
                    python=python,
                    extracted=extracted,
                    run_root=workspace.consumer,
                    expected_version=config.python_version,
                    profile="base",
                    run=run,
                ),
                mock.call.run(
                    [
                        "/reviewed/uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        f"{wheel}[gmsh]",
                    ],
                    cwd=workspace.environment.parent,
                ),
                mock.call.run(
                    [
                        str(python),
                        "-I",
                        "-m",
                        "pytest",
                        "-q",
                        *(str(test) for test in gmsh_tests),
                    ],
                    cwd=workspace.consumer,
                    extra_environment={
                        "EQIORA_GMSH": str(
                            python.parent / ("gmsh.exe" if os.name == "nt" else "gmsh")
                        ),
                        "PATH": gmsh_path,
                    },
                ),
            ],
        )

    def test_base_public_smoke_rejects_gmsh_imported_with_eqiora(self) -> None:
        smoke = importlib.import_module("python_public_smoke")
        expected_version = "0.1.0"

        def reached_base_execution(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("base smoke continued after importing Gmsh")

        eqiora = types.SimpleNamespace(
            __version__=expected_version,
            Field=reached_base_execution,
        )
        real_import = builtins.__import__

        def import_with_gmsh(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == "eqiora":
                sys.modules["gmsh"] = types.ModuleType("gmsh")
                return eqiora
            return real_import(name, globals, locals, fromlist, level)

        with (
            mock.patch.dict(sys.modules, {"eqiora": eqiora}),
            mock.patch.object(builtins, "__import__", new=import_with_gmsh),
            mock.patch.object(
                smoke.importlib.metadata,
                "version",
                return_value=expected_version,
            ),
        ):
            sys.modules.pop("gmsh", None)
            try:
                smoke.base_smoke(expected_version)
            except AssertionError:
                pass
            except RuntimeError as error:
                self.fail(str(error))
            else:  # pragma: no cover - the fake base execution always stops
                self.fail("base smoke accepted an Eqiora import that loaded Gmsh")

    def config(self) -> DistributionConfig:
        return PythonCandidateTests().config()

    @contextlib.contextmanager
    def mocked_candidate_build(
        self,
        root: Path,
        profile_callback: Callable[[str, types.SimpleNamespace], None],
    ) -> Iterator[types.SimpleNamespace]:
        config = self.config()
        output = root / "artifacts"
        scratch = root / "candidate-scratch"
        extracted = scratch / "source"
        extracted.mkdir(parents=True)
        (extracted / "LICENSE").write_text("license\n", encoding="utf-8")
        (extracted / "NOTICE").write_text("notice\n", encoding="utf-8")
        sdist = scratch / "eqiora-0.1.0.tar.gz"
        sdist.write_bytes(b"sdist")
        wheels = {
            version: scratch / f"eqiora-0.1.0-cp{version.replace('.', '')}.whl"
            for version in config.interpreters
        }
        for version, wheel in wheels.items():
            wheel.write_bytes(f"wheel-{version}".encode())

        observations = types.SimpleNamespace(
            output=output,
            scratch=scratch,
            extracted=extracted,
            sdist=sdist,
            wheels=wheels,
            inspections=0,
            interpreter_resolutions=0,
            active_interpreter_resolutions=0,
            maximum_interpreter_resolutions=0,
            temporary_calls=[],
            scratch_exited=False,
        )
        interpreter_lock = threading.Lock()

        def temporary_directory(*args: object, **kwargs: object) -> object:
            observations.temporary_calls.append((args, kwargs))
            parent = kwargs.get("dir")
            if parent is not None and not Path(parent).resolve().is_relative_to(
                Path.home().resolve()
            ):
                raise AssertionError("candidate scratch escaped home")

            @contextlib.contextmanager
            def owned_scratch() -> object:
                try:
                    yield str(scratch)
                finally:
                    observations.scratch_exited = True

            return owned_scratch()

        def inspect(*args: object, **kwargs: object) -> tuple[str, dict[str, object]]:
            observations.inspections += 1
            python_version = kwargs["python_version"]
            wheel = args[0] if args else kwargs["wheel"]
            return config.python_version, {
                "filename": wheel.name,
                "kind": "wheel",
                "python": python_version,
                "sha256": python_candidate_module.sha256(wheel),
            }

        def interpreter(_uv: str, version: str) -> str:
            with interpreter_lock:
                observations.active_interpreter_resolutions += 1
                observations.maximum_interpreter_resolutions = max(
                    observations.maximum_interpreter_resolutions,
                    observations.active_interpreter_resolutions,
                )
            time.sleep(0.005)
            with interpreter_lock:
                observations.active_interpreter_resolutions -= 1
                observations.interpreter_resolutions += 1
            return f"/managed/python-{version}"

        def base(**kwargs: object) -> list[str]:
            name = f"base-{kwargs['python_version']}"
            profile_callback(name, observations)
            return [f"check:{name}"]

        def numpy_floor(**kwargs: object) -> tuple[list[str], dict[str, str]]:
            name = "numpy-floor-3.12"
            profile_callback(name, observations)
            return [f"check:{name}"], {
                "python": "3.12",
                "observed": "2.1.0",
            }

        def optional(**kwargs: object) -> list[str]:
            name = f"{kwargs['name']}-3.13"
            profile_callback(name, observations)
            return [f"check:{name}"]

        def typing(**_kwargs: object) -> str:
            name = "typing-3.13"
            profile_callback(name, observations)
            return f"check:{name}"

        def checked(argv: list[str], **_kwargs: object) -> str:
            if any(part.endswith("generate_python_api.py") for part in argv):
                profile_callback("generated-public-api", observations)
            return ""

        build_artifacts = mock.Mock(return_value=(sdist, wheels, extracted))
        manifest_writer = mock.Mock(return_value=output / "candidate.json")
        patches = (
            mock.patch.object(
                python_candidate_module.platform, "system", return_value="Linux"
            ),
            mock.patch.object(
                python_candidate_module.platform, "machine", return_value="x86_64"
            ),
            mock.patch.object(
                python_candidate_module, "load_config", return_value=config
            ),
            mock.patch.object(
                python_candidate_module,
                "source_identity",
                return_value=SourceIdentity("0" * 40, ()),
            ),
            mock.patch.object(
                python_candidate_module,
                "require_executable",
                side_effect=lambda name: f"/tool/{name}",
            ),
            mock.patch.object(
                python_candidate_module,
                "ensure_exact_uv",
                return_value="/tool/uv",
            ),
            mock.patch.object(
                python_candidate_module, "checked_run", side_effect=checked
            ),
            mock.patch.object(
                python_candidate_module, "build_artifacts", build_artifacts
            ),
            mock.patch.object(
                python_candidate_module, "inspect_wheel", side_effect=inspect
            ),
            mock.patch.object(
                python_candidate_module, "uv_interpreter", side_effect=interpreter
            ),
            mock.patch.object(
                python_candidate_module, "run_base_profile", side_effect=base
            ),
            mock.patch.object(
                python_candidate_module,
                "run_numpy_floor_profile",
                side_effect=numpy_floor,
            ),
            mock.patch.object(
                python_candidate_module, "run_optional_profile", side_effect=optional
            ),
            mock.patch.object(
                python_candidate_module, "run_full_typing_profile", side_effect=typing
            ),
            mock.patch.object(
                python_candidate_module, "write_manifest", manifest_writer
            ),
            mock.patch.object(
                python_candidate_module.tempfile,
                "TemporaryDirectory",
                side_effect=temporary_directory,
            ),
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            observations.build_artifacts = build_artifacts
            observations.manifest_writer = manifest_writer
            yield observations

    def test_direct_candidate_scratch_is_resolved_below_home(self) -> None:
        calls: Counter[str] = Counter()
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            with self.mocked_candidate_build(
                Path(temporary), lambda name, _observed: calls.update((name,))
            ) as observed:
                python_candidate_module.build_candidate(
                    observed.output,
                    require_tag=False,
                    skip_extras=True,
                )

        self.assertEqual(len(observed.temporary_calls), 1)
        self.assertEqual(calls, Counter(self.COMPLETE_NAMES[:6]))
        _args, keyword_arguments = observed.temporary_calls[0]
        parent = keyword_arguments.get("dir")
        self.assertIsNotNone(parent)
        self.assertTrue(Path(parent).resolve().is_relative_to(Path.home().resolve()))

    def test_build_has_one_barrier_fanout_and_frozen_manifest_merge(self) -> None:
        calls: Counter[str] = Counter()
        active: set[str] = set()
        overlap: set[frozenset[str]] = set()
        lock = threading.Lock()

        def profile(name: str, observations: types.SimpleNamespace) -> None:
            self.assertEqual(observations.inspections, 4)
            self.assertEqual(observations.interpreter_resolutions, 4)
            with lock:
                calls[name] += 1
                overlap.update(frozenset((name, other)) for other in active)
                active.add(name)
            time.sleep(0.04 if name == "base-3.11" else 0.02)
            with lock:
                active.remove(name)

        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            root = Path(temporary)
            with self.mocked_candidate_build(root, profile) as observed:
                manifest = python_candidate_module.build_candidate(
                    observed.output,
                    require_tag=False,
                    skip_extras=False,
                )

        self.assertEqual(manifest, observed.output / "candidate.json")
        self.assertEqual(calls, Counter(self.COMPLETE_NAMES))
        observed.build_artifacts.assert_called_once()
        observed.manifest_writer.assert_called_once()
        self.assertEqual(observed.maximum_interpreter_resolutions, 1)
        self.assertEqual(len(observed.temporary_calls), 1)

        heavy = {"torch-3.13", "jax-3.13", "typing-3.13"}
        self.assertFalse(any(pair <= heavy for pair in overlap))
        self.assertTrue(
            any(len(pair & heavy) == 1 and len(pair - heavy) == 1 for pair in overlap)
        )
        self.assertIn(frozenset(("base-3.11", "base-3.12")), overlap)

        manifest_arguments = observed.manifest_writer.call_args.kwargs
        self.assertEqual(
            manifest_arguments["checks"],
            [
                "twine-strict",
                "sdist-to-wheel-rebuild",
                *(f"check:{name}" for name in self.COMPLETE_NAMES),
            ],
        )
        self.assertEqual(
            manifest_arguments["dependency_profiles"],
            {"numpy_floor": {"python": "3.12", "observed": "2.1.0"}},
        )

    def test_final_identity_rejects_profile_mutation_before_manifest(self) -> None:
        mutations = {
            "sdist": lambda observed: observed.sdist.write_bytes(b"mutated sdist"),
            "wheel": lambda observed: observed.wheels["3.11"].write_bytes(
                b"mutated wheel"
            ),
            "extracted source": lambda observed: (
                observed.extracted / "unexpected"
            ).write_text("mutation\n", encoding="utf-8"),
        }
        for target, mutate in mutations.items():
            with self.subTest(target=target):
                mutated = False

                def profile(_name: str, observed: types.SimpleNamespace) -> None:
                    nonlocal mutated
                    if not mutated:
                        mutate(observed)
                        mutated = True

                with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
                    with self.mocked_candidate_build(
                        Path(temporary), profile
                    ) as observed:
                        with self.assertRaises(CandidateError):
                            python_candidate_module.build_candidate(
                                observed.output,
                                require_tag=False,
                                skip_extras=True,
                            )
                        observed.manifest_writer.assert_not_called()

    def test_profile_failures_join_before_cleanup_and_block_manifest(self) -> None:
        rendezvous = threading.Barrier(2)
        started: list[str] = []
        lock = threading.Lock()

        def profile(name: str, observed: types.SimpleNamespace) -> None:
            if name not in {"base-3.11", "base-3.12"}:
                return
            with lock:
                started.append(name)
            rendezvous.wait(timeout=1.0)
            if name == "base-3.11":
                time.sleep(0.03)
            self.assertFalse(observed.scratch_exited)
            raise RuntimeError(f"diagnostic from {name}")

        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            with self.mocked_candidate_build(Path(temporary), profile) as observed:
                with self.assertRaises(CandidateError) as raised:
                    python_candidate_module.build_candidate(
                        observed.output,
                        require_tag=False,
                        skip_extras=True,
                    )
                observed.manifest_writer.assert_not_called()
            self.assertTrue(observed.scratch_exited)

        self.assertCountEqual(started, ["base-3.11", "base-3.12"])
        diagnostic = str(raised.exception)
        first = diagnostic.index("base-3.11")
        second = diagnostic.index("base-3.12")
        self.assertLess(first, second)

    def test_profile_writable_roots_and_environment_are_disjoint(self) -> None:
        profiles = self.profiles_module()
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            scratch = Path(temporary)
            plan = profiles.build_profile_plan(
                scratch, self.config(), skip_extras=False
            )

            writable: list[Path] = []
            for item in plan:
                self.assertTrue(item.root.is_relative_to(scratch))
                paths = [
                    item.environment,
                    item.consumer,
                    item.temporary,
                    item.log,
                ]
                if item.matplotlib_config is not None:
                    paths.append(item.matplotlib_config)
                for path in paths:
                    self.assertTrue(path.is_relative_to(item.root))
                writable.extend(paths)

                environment = dict(item.environment_variables)
                self.assertEqual(Path(environment["TMPDIR"]), item.temporary)
                if item.name == "matplotlib-3.13":
                    self.assertEqual(
                        Path(environment["MPLCONFIGDIR"]), item.matplotlib_config
                    )
                else:
                    self.assertIsNone(item.matplotlib_config)
                    self.assertNotIn("MPLCONFIGDIR", environment)

            self.assertEqual(len(set(writable)), len(writable))

            owners = {
                "EQIORA_TEST_TORCH_VERSION": "torch-3.13",
                "EQIORA_REQUIRE_JAX_ABI_PROBE": "jax-3.13",
                "EQIORA_TEST_JAX_VERSION": "jax-3.13",
                "EQIORA_TEST_PYTHON_VERSION": "jax-3.13",
                "JAX_ENABLE_X64": "jax-3.13",
                "XLA_FLAGS": "jax-3.13",
                "EQIORA_TEST_MATPLOTLIB_VERSION": "matplotlib-3.13",
                "MPLBACKEND": "matplotlib-3.13",
                "MPLCONFIGDIR": "matplotlib-3.13",
            }
            for variable, owner in owners.items():
                observed = [
                    item.name
                    for item in plan
                    if variable in dict(item.environment_variables)
                ]
                self.assertEqual(observed, [owner], variable)

            config = self.config()
            workspace = next(item for item in plan if item.name == "matplotlib-3.13")
            extracted = scratch / "source"
            source_test = extracted / "bindings/python/tests/test_matplotlib.py"
            source_test.parent.mkdir(parents=True)
            source_test.write_text(
                "def test_placeholder():\n    pass\n", encoding="utf-8"
            )
            python = workspace.environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            demos = tuple(
                workspace.consumer / name for name in ("exact.py", "mixed.py")
            )
            install = mock.Mock(return_value=python)

            def run(argv: list[str], **_kwargs: object) -> str:
                for value in argv:
                    output = Path(value)
                    if output.is_absolute() and output.suffix == ".png":
                        output.write_bytes(b"\x89PNG\r\n\x1a\n")
                return ""

            checked = mock.Mock(side_effect=run)
            with (
                mock.patch.object(profiles, "install_environment", new=install),
                mock.patch.object(
                    profiles,
                    "prepare_exact_cylinder_demo_consumer",
                    return_value=demos[0],
                ),
                mock.patch.object(
                    profiles,
                    "prepare_mixed_boundary_elasticity_demo_consumer",
                    return_value=demos[1],
                ),
                mock.patch.dict(
                    os.environ,
                    {"EQIORA_GMSH": "/ambient/gmsh", "PATH": "/ambient/bin"},
                ),
            ):
                checks = profiles.run_optional_profile(
                    name="matplotlib",
                    uv="/reviewed/uv",
                    interpreter="/reviewed/python3.13",
                    wheel=scratch / "candidate.whl",
                    extracted=extracted,
                    workspace=workspace,
                    config=config,
                    run=checked,
                )

            self.assertEqual(
                checks,
                [
                    "cp313:matplotlib",
                    "cp313:packaged-exact-cylinder-pressure-demo",
                    "cp313:packaged-mixed-boundary-displacement-demo",
                ],
            )
            install.assert_called_once_with(
                uv="/reviewed/uv",
                interpreter="/reviewed/python3.13",
                environment=workspace.environment,
                requirements=[
                    f"{scratch / 'candidate.whl'}[gmsh,matplotlib]",
                    config.pytest,
                    config.matplotlib,
                ],
                run=checked,
            )
            self.assertEqual(
                [tuple(call.args[0][2:4]) for call in checked.call_args_list],
                [
                    ("-m", "pytest"),
                    (str(demos[0]), "--pressure-png"),
                    (str(demos[1]), "--displacement-png"),
                ],
            )
            expected_environment = {
                "EQIORA_GMSH": str(
                    python.parent / ("gmsh.exe" if os.name == "nt" else "gmsh")
                ),
                "PATH": os.pathsep.join((str(python.parent), "/ambient/bin")),
            }
            for index, call in enumerate(checked.call_args_list):
                with self.subTest(matplotlib_run=index):
                    self.assertEqual(
                        call.kwargs.get("extra_environment"), expected_environment
                    )

    def test_reverse_completion_merges_immutable_receipts_in_frozen_order(
        self,
    ) -> None:
        profiles = self.profiles_module()
        first = profiles.ProfileReceipt(
            name="base-3.11",
            checks=("base-z", "base-a"),
            dependency_profiles=(),
            diagnostics=("base diagnostic",),
            log="BASE LOG\n",
        )
        second = profiles.ProfileReceipt(
            name="numpy-floor-3.12",
            checks=("numpy-floor",),
            dependency_profiles=(
                (
                    "numpy_floor",
                    (
                        ("observed", "2.1.0"),
                        ("python", "3.12"),
                    ),
                ),
            ),
            diagnostics=("numpy diagnostic",),
            log="NUMPY LOG\n",
        )
        with self.assertRaises(FrozenInstanceError):
            first.name = "mutated"

        forward = profiles.merge_profile_receipts(
            ("base-3.11", "numpy-floor-3.12"), (first, second)
        )
        reversed_completion = profiles.merge_profile_receipts(
            ("base-3.11", "numpy-floor-3.12"), (second, first)
        )

        self.assertEqual(forward, reversed_completion)
        self.assertEqual(forward.receipts, (first, second))
        self.assertEqual(forward.checks, ("base-z", "base-a", "numpy-floor"))
        self.assertEqual(
            forward.dependency_profiles,
            second.dependency_profiles,
        )
        self.assertEqual(
            forward.diagnostics,
            (
                ("base-3.11", "base diagnostic"),
                ("numpy-floor-3.12", "numpy diagnostic"),
            ),
        )
        self.assertEqual(
            forward.logs,
            (
                ("base-3.11", "BASE LOG\n"),
                ("numpy-floor-3.12", "NUMPY LOG\n"),
            ),
        )

        manifests: list[bytes] = []
        with (
            tempfile.TemporaryDirectory(dir=Path.home()) as temporary,
            mock.patch.object(
                python_candidate_module,
                "tool_version",
                return_value="reviewed tool",
            ),
        ):
            root = Path(temporary)
            for index, report in enumerate((forward, reversed_completion)):
                output = root / str(index)
                output.mkdir()
                sdist = output / "eqiora-0.1.0.tar.gz"
                sdist.write_bytes(b"one immutable source distribution")
                manifest = python_candidate_module.write_manifest(
                    output=output,
                    source=SourceIdentity("0" * 40, ()),
                    sdist=sdist,
                    version="0.1.0",
                    wheel_records=[],
                    checks=list(report.checks),
                    config=self.config(),
                    uv="/reviewed/uv",
                    complete_profiles=True,
                    dependency_profiles={
                        name: dict(values)
                        for name, values in report.dependency_profiles
                    },
                )
                manifests.append(manifest.read_bytes())
        self.assertEqual(manifests[0], manifests[1])
        document = json.loads(manifests[0])
        self.assertEqual(
            document["format"], "eqiora.python-distribution-candidate/v4"
        )
        self.assertNotIn("frontend", document["build"])

        for invalid in ((first,), (first, first), (first, second, second)):
            with self.assertRaisesRegex(ValueError, "receipt"):
                profiles.merge_profile_receipts(
                    ("base-3.11", "numpy-floor-3.12"), invalid
                )


class CandidateFamilyAdmissionTests(unittest.TestCase):
    @staticmethod
    def write_family(root: Path) -> None:
        (root / "eqiora-0.1.0.tar.gz").write_bytes(b"sdist")
        for compact_python in EXACT_WHEEL_INTERPRETERS:
            write_maturin_wheel(
                root / exact_wheel_name(compact_python),
                compact_python,
            )

    def test_exact_family_is_admitted_in_interpreter_order(self) -> None:
        family_module = importlib.import_module("python_candidate_family")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_family(root)
            family = family_module.admit_candidate_family(root)

        self.assertEqual(family.version, "0.1.0")
        self.assertEqual(
            tuple(path.name for path in family.wheels),
            tuple(exact_wheel_name(python) for python in EXACT_WHEEL_INTERPRETERS),
        )
        self.assertEqual(len(family.inventory), 5)

    def test_extra_and_linked_members_fail_closed(self) -> None:
        family_module = importlib.import_module("python_candidate_family")
        for mutation in ("extra", "hard-link"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                self.write_family(root)
                if mutation == "extra":
                    (root / "receipt.json").write_bytes(b"metadata")
                else:
                    source = root / exact_wheel_name("311")
                    linked = root / "linked.whl"
                    linked.hardlink_to(source)
                with self.assertRaises(CandidateError):
                    family_module.admit_candidate_family(root)
