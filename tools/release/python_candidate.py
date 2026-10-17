#!/usr/bin/env python3
"""Build and verify one commit-bound Eqiora Python distribution candidate."""

from __future__ import annotations

import argparse
import email.parser
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import python_candidate_profiles as candidate_profiles
from candidate_manifest import (
    MANIFEST_FORMAT,
    PROFILE_CHECKS,
    REQUIRED_PROFILES,
    load_candidate_family,
    verify_artifacts,
)
from python_candidate_common import (
    CandidateError,
    DistributionConfig,
    candidate_payload_identity,
    checked_run,
    command_context,
    home_scratch_parent,
    python_distribution_version as python_distribution_version,
    sha256 as sha256,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT
PYPROJECT = ROOT / "pyproject.toml"
PYTHON_TEST_FIXTURES = candidate_profiles.PYTHON_TEST_FIXTURES
PYTHON_TEST_RESOURCES = candidate_profiles.PYTHON_TEST_RESOURCES
GIT_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SourceIdentity:
    """Exact source state from which a candidate is built."""

    commit: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CandidateProfileSummary:
    """Accepted profile observations used only to finalize one manifest."""

    config: DistributionConfig
    uv: str
    wheel_record: dict[str, Any]
    checks: tuple[str, ...]
    dependency_profiles: dict[str, dict[str, str]]


def exact_group_requirement(
    document: dict[str, Any],
    group: str,
    package: str,
) -> str:
    """Return one exact standard dependency-group requirement."""

    requirements = document.get("dependency-groups", {}).get(group, [])
    prefix = f"{package}=="
    matches = [
        requirement
        for requirement in requirements
        if isinstance(requirement, str) and requirement.startswith(prefix)
    ]
    if len(matches) != 1 or matches[0] == prefix:
        raise CandidateError(
            f"dependency-groups.{group} must contain one exact {package} requirement"
        )
    return matches[0]


def load_config() -> DistributionConfig:
    """Load the single reviewed matrix and tool inventory."""

    document = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    project = document["project"]
    if "version" in project or project.get("dynamic") != ["version"]:
        raise CandidateError("Python version must derive only from Cargo")
    build_requires = document["build-system"]["requires"]
    maturin = [item for item in build_requires if item.startswith("maturin==")]
    if len(maturin) != 1:
        raise CandidateError(
            "build-system.requires must contain one exact maturin== version"
        )

    raw = document["tool"]["eqiora-distribution"]
    uv = exact_group_requirement(document, "release-tools", "uv")
    twine = exact_group_requirement(document, "release-tools", "twine")
    interpreters = tuple(raw["ordinary-gil-cpython"])
    jax = tuple(raw["tested-jax"])
    config = DistributionConfig(
        cargo_version=cargo["workspace"]["package"]["version"],
        interpreters=interpreters,
        wheel_platform=raw["wheel-platform"],
        extras_interpreter=raw["extras-python"],
        numpy_floor_interpreter=raw["numpy-floor-python"],
        numpy_floor=raw["tested-numpy-floor"],
        uv=uv,
        maturin=maturin[0],
        pytest=raw["pytest"],
        mypy=raw["mypy"],
        twine=twine,
        torch=raw["tested-torch"],
        jax=jax,
        matplotlib=raw["tested-matplotlib"],
        rust=cargo["workspace"]["package"]["rust-version"],
    )
    if config.interpreters != ("3.11", "3.12", "3.13", "3.14"):
        raise CandidateError("the first candidate matrix must be CPython 3.11-3.14")
    if config.extras_interpreter not in config.interpreters:
        raise CandidateError("extras-python must belong to the wheel matrix")
    if config.numpy_floor_interpreter not in config.interpreters:
        raise CandidateError("numpy-floor-python must belong to the wheel matrix")
    if config.numpy_floor_interpreter != "3.12" or config.numpy_floor != "numpy==2.1.0":
        raise CandidateError(
            "the first candidate must verify the exact NumPy 2.1.0 floor on CPython 3.12"
        )
    if config.wheel_platform != "manylinux_2_17_x86_64":
        raise CandidateError(
            "the first candidate must retain the manylinux_2_17_x86_64 floor"
        )
    exact_uv_version(config.uv)
    return config


def require_executable(name: str) -> str:
    """Resolve one conventional release tool or fail clearly."""

    executable = shutil.which(name)
    if executable is None:
        raise CandidateError(f"required executable is unavailable: {name}")
    return executable


def exact_uv_version(requirement: str) -> str:
    """Return the path-safe version from one exact ``uv`` requirement."""
    name, separator, expected = requirement.partition("==")
    components = expected.split(".")
    if (
        name != "uv"
        or not separator
        or len(components) != 3
        or any(
            not component.isascii() or not component.isdecimal()
            for component in components
        )
    ):
        raise CandidateError("the uv build-tool requirement is malformed")
    return expected


def require_exact_uv(executable: str, requirement: str) -> None:
    """Require the declared release tool rather than an ambient compatible one."""

    expected = exact_uv_version(requirement)
    observed = tool_version([executable, "--version"]).split()
    if len(observed) < 2 or observed[0] != "uv" or observed[1] != expected:
        raise CandidateError(
            f"candidate requires uv {expected}, observed {' '.join(observed)!r}"
        )


def _virtual_environment_executable(environment: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return environment / directory / f"{name}{suffix}"


def ensure_exact_uv(
    requirement: str,
    *,
    cache_root: Path | None = None,
) -> str:
    """Resolve the reviewed ``uv`` release into an immutable home cache entry."""

    version = exact_uv_version(requirement)
    home = Path.home().resolve()
    root = (
        cache_root.resolve()
        if cache_root is not None
        else home / ".cache" / "eqiora" / "tools"
    )
    if cache_root is None and not root.is_relative_to(home):
        raise CandidateError("the uv tool cache must remain below the home directory")
    uv_root = root / "uv"
    uv_root.mkdir(parents=True, exist_ok=True)
    resolved_root = uv_root.resolve()
    if cache_root is None and not resolved_root.is_relative_to(home):
        raise CandidateError("the uv tool cache must remain below the home directory")

    tool_root = resolved_root / version
    executable = _virtual_environment_executable(tool_root, "uv")
    if executable.is_file():
        require_exact_uv(str(executable), requirement)
        return str(executable)
    if tool_root.exists():
        raise CandidateError(f"the cached uv tool is incomplete: {tool_root}")

    with tempfile.TemporaryDirectory(
        prefix=f".{version}-", dir=resolved_root
    ) as temporary:
        staged = Path(temporary) / "tool"
        checked_run([sys.executable, "-m", "venv", str(staged)])
        staged_python = _virtual_environment_executable(staged, "python")
        checked_run(
            [
                str(staged_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                requirement,
            ]
        )
        staged_uv = _virtual_environment_executable(staged, "uv")
        require_exact_uv(str(staged_uv), requirement)
        try:
            staged.rename(tool_root)
        except OSError as error:
            if executable.is_file():
                require_exact_uv(str(executable), requirement)
                return str(executable)
            raise CandidateError(
                f"failed to publish cached uv tool: {tool_root}"
            ) from error

    require_exact_uv(str(executable), requirement)
    return str(executable)


def source_identity() -> SourceIdentity:
    """Require a clean commit and return its full identity."""

    status = checked_run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        capture=True,
    )
    if status:
        raise CandidateError("a Python candidate requires a clean source tree")
    commit = checked_run(["git", "rev-parse", "HEAD"], capture=True)
    if len(commit) != 40:
        raise CandidateError("git did not return a full source commit")
    tags = tuple(
        line
        for line in checked_run(
            ["git", "tag", "--points-at", commit],
            capture=True,
        ).splitlines()
        if line
    )
    return SourceIdentity(commit=commit, tags=tags)


def require_expected_tag(source: SourceIdentity, expected_tag: str) -> None:
    """Require the release tag derived from the normalized artifact version."""

    if expected_tag not in source.tags:
        raise CandidateError(
            f"publication requires exact tag {expected_tag} on the source commit"
        )


def require_annotated_expected_tag(
    source: SourceIdentity,
    expected_tag: str,
    *,
    git_query: Callable[..., str] = checked_run,
) -> None:
    """Require the exact annotated tag to peel to the candidate commit."""

    require_expected_tag(source, expected_tag)
    reference = f"refs/tags/{expected_tag}"
    if git_query(["git", "cat-file", "-t", reference], capture=True) != "tag":
        raise CandidateError(
            f"publication requires annotated tag {expected_tag}, not a lightweight tag"
        )
    peeled = git_query(
        ["git", "rev-parse", f"{reference}^{{commit}}"],
        capture=True,
    )
    if peeled != source.commit:
        raise CandidateError(
            f"annotated tag {expected_tag} does not peel to the candidate commit"
        )


def cargo_workspace_version(root: Path) -> str:
    """Read the sole authored release version from one source root."""

    document = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    return document["workspace"]["package"]["version"]


def safe_extract_sdist(archive: Path, destination: Path) -> Path:
    """Extract a regular-file/directory sdist without path traversal."""

    root = destination.resolve()
    with tarfile.open(archive, mode="r:gz") as source:
        members = source.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise CandidateError(f"sdist path escapes its root: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise CandidateError(
                    f"sdist contains a non-regular member: {member.name}"
                )
        source.extractall(destination, members=members)

    children = [path for path in destination.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise CandidateError("sdist must contain exactly one top-level directory")
    extracted = children[0]
    required = (
        extracted / "Cargo.toml",
        extracted / "Cargo.lock",
        extracted / "pyproject.toml",
        extracted / "crates/eqiora-python/Cargo.toml",
    )
    missing = [
        str(path.relative_to(extracted)) for path in required if not path.is_file()
    ]
    if missing:
        raise CandidateError(f"sdist is incomplete: {', '.join(missing)}")
    return extracted


def uv_interpreter(uv: str, version: str) -> str:
    """Resolve an exact managed CPython interpreter."""

    checked_run([uv, "python", "install", version])
    interpreter = checked_run(
        [uv, "python", "find", "--managed-python", version],
        capture=True,
    ).splitlines()[-1]
    if not Path(interpreter).is_file():
        raise CandidateError(f"uv did not resolve CPython {version}")
    observed = checked_run(
        [
            interpreter,
            "-I",
            "-c",
            "import platform,sys; print(platform.python_implementation()); "
            "print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        capture=True,
    ).splitlines()
    if observed != ["CPython", version]:
        raise CandidateError(
            f"requested CPython {version}, resolved {' '.join(observed)}"
        )
    return interpreter


def maturin_package(config: DistributionConfig, *, zig: bool) -> str:
    """Return the exact maturin tool requirement, optionally with Zig."""

    name, version = config.maturin.split("==", maxsplit=1)
    suffix = "[zig]" if zig else ""
    return f"{name}{suffix}=={version}"


def _producer_file_identity(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise CandidateError(
            f"candidate producer emitted a non-regular file: {path.name}"
        )
    return path.stat().st_size, sha256(path)


def build_artifacts(
    *,
    output: Path,
    scratch: Path,
    config: DistributionConfig,
    uv: str,
    interpreter: str,
) -> tuple[Path, Path, Path]:
    """Build one sdist, then the shared wheel solely from its extracted content."""

    if not output.is_dir() or output.is_symlink() or any(output.iterdir()):
        raise CandidateError("candidate producer output directory must be empty")
    rust_environment = {"RUSTUP_TOOLCHAIN": config.rust}
    sdist_tool = maturin_package(config, zig=False)
    checked_run(
        [
            uv,
            "tool",
            "run",
            "--from",
            sdist_tool,
            "maturin",
            "sdist",
            "--out",
            str(output),
        ],
        cwd=PACKAGE,
        extra_environment=rust_environment,
    )
    expected_sdist = f"eqiora-{config.python_version}.tar.gz"
    members = {path.name: path for path in output.iterdir()}
    if set(members) != {expected_sdist}:
        raise CandidateError(
            f"source distribution identity drifted: expected {expected_sdist}, "
            f"received {', '.join(sorted(members)) or '<nothing>'}"
        )
    sdist = members[expected_sdist]
    sdist_identity = _producer_file_identity(sdist)

    extracted = safe_extract_sdist(sdist, scratch / "source")
    if cargo_workspace_version(extracted) != config.cargo_version:
        raise CandidateError(
            "source distribution Cargo version differs from the candidate source"
        )
    target = scratch / "cargo-target"
    wheel_tool = maturin_package(config, zig=True)
    checked_run(
        [
            uv,
            "tool",
            "run",
            "--from",
            wheel_tool,
            "maturin",
            "build",
            "--release",
            "--zig",
            "--compatibility",
            "manylinux_2_17",
            "--auditwheel",
            "check",
            "--compression-method",
            "deflated",
            "--compression-level",
            "9",
            "--interpreter",
            interpreter,
            "--target-dir",
            str(target),
            "--out",
            str(output),
        ],
        cwd=extracted,
        extra_environment=rust_environment,
    )
    family = admit_candidate_family(output)
    if family.sdist != sdist or _producer_file_identity(sdist) != sdist_identity:
        raise CandidateError("wheel build mutated retained source distribution")
    return sdist, family.wheel, extracted


def parse_metadata(payload: bytes) -> email.message.Message:
    """Parse one package metadata document under the standard email grammar."""

    return email.parser.BytesParser().parsebytes(payload)


def inspect_wheel(
    wheel: Path,
    *,
    config: DistributionConfig,
    license_bytes: bytes,
    notice_bytes: bytes,
) -> tuple[str, dict[str, Any]]:
    """Verify tags, package files, metadata, and notices in one wheel."""

    _family_admission().require_exact_maturin_wheel(wheel, version=config.python_version)

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise CandidateError(f"wheel has an ambiguous METADATA: {wheel.name}")
        metadata = parse_metadata(archive.read(metadata_names[0]))
        dist_info = metadata_names[0].removesuffix("METADATA")

        required = (
            "eqiora/__init__.py",
            "eqiora/__init__.pyi",
            "eqiora/colab.py",
            "eqiora/colab.pyi",
            "eqiora/diff.pyi",
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
            f"{dist_info}licenses/LICENSE",
            f"{dist_info}licenses/NOTICE",
        )
        missing = [name for name in required if name not in names]
        if missing:
            raise CandidateError(
                f"wheel omits required package files: {', '.join(missing)}"
            )
        if archive.read(f"{dist_info}licenses/LICENSE") != license_bytes:
            raise CandidateError("wheel LICENSE does not match the source license")
        if archive.read(f"{dist_info}licenses/NOTICE") != notice_bytes:
            raise CandidateError("wheel NOTICE does not match the source notice")
        extensions = [name for name in names if name.endswith(".so")]
        if extensions != ["eqiora/_eqiora.abi3.so"]:
            raise CandidateError("wheel must contain exactly the stable-ABI native extension")
        if "eqiora/_eqiora.pyi" in names:
            raise CandidateError("the private native module must not be published")
        if not any(".dist-info/sboms/" in name for name in names):
            raise CandidateError("wheel omits its generated native dependency SBOM")
    if metadata["Name"] != "eqiora":
        raise CandidateError("wheel distribution name is not eqiora")
    version = metadata["Version"]
    if version != config.python_version:
        raise CandidateError(
            f"wheel metadata version {version!r} differs from {config.python_version!r}"
        )
    requires_python = {item.strip() for item in metadata["Requires-Python"].split(",")}
    if requires_python != {">=3.11", "<3.15"}:
        raise CandidateError("wheel has an unexpected Requires-Python")
    if metadata["License-Expression"] != "Apache-2.0":
        raise CandidateError("wheel has an unexpected license expression")
    if sorted(metadata.get_all("License-File", [])) != ["LICENSE", "NOTICE"]:
        raise CandidateError("wheel has incomplete PEP 639 license metadata")

    dependencies = metadata.get_all("Requires-Dist", [])
    normalized = [
        dependency.lower().replace(" ", "").replace("'", '"')
        for dependency in dependencies
    ]
    numpy_requirements = [item for item in normalized if item.startswith("numpy")]
    if (
        len(numpy_requirements) != 1
        or ">=2.1" not in numpy_requirements[0]
        or "<3" not in numpy_requirements[0]
        or ";" in numpy_requirements[0]
    ):
        raise CandidateError("wheel must declare the reviewed NumPy range")
    gmsh_requirements = [item for item in normalized if item.startswith("gmsh")]
    if gmsh_requirements != ['gmsh==4.15.2;extra=="gmsh"']:
        raise CandidateError("wheel must declare exactly the Gmsh 4.15.2 extra")
    for framework in ("torch", "jax", "jaxlib", "matplotlib"):
        declarations = [item for item in normalized if item.startswith(framework)]
        if not declarations or any("extra==" not in item for item in declarations):
            raise CandidateError(
                f"{framework} must remain an optional-extra dependency"
            )
    anywidget_requirements = [
        item for item in normalized if item.startswith("anywidget")
    ]
    if anywidget_requirements != ["anywidget==0.11.0"]:
        raise CandidateError(
            "wheel must declare exactly anywidget 0.11.0 as a base dependency"
        )
    expected_extras = ["gmsh", "jax", "matplotlib", "torch"]
    if sorted(metadata.get_all("Provides-Extra", [])) != expected_extras:
        raise CandidateError(
            "wheel must expose exactly the reviewed optional extras"
        )

    return version, {
        "filename": wheel.name,
        "kind": "wheel",
        "python": "3.11",
        "abi": "abi3",
        "platform": config.wheel_platform,
        "size": wheel.stat().st_size,
        "sha256": sha256(wheel),
    }


prepare_base_consumer_tree = candidate_profiles.prepare_base_consumer_tree
prepare_exact_cylinder_demo_consumer = (
    candidate_profiles.prepare_exact_cylinder_demo_consumer
)
prepare_mixed_boundary_elasticity_demo_consumer = (
    candidate_profiles.prepare_mixed_boundary_elasticity_demo_consumer
)
def run_public_smoke(
    *,
    python: Path,
    extracted: Path,
    run_root: Path,
    expected_version: str,
    profile: str,
) -> None:
    """Replay one published quick start against an installed wheel."""

    candidate_profiles.run_public_smoke(
        python=python,
        extracted=extracted,
        run_root=run_root,
        expected_version=expected_version,
        profile=profile,
        run=checked_run,
    )


def run_base_profile(
    *,
    uv: str,
    interpreter: str,
    python_version: str,
    wheel: Path,
    extracted: Path,
    workspace: candidate_profiles.ProfileWorkspace,
    config: DistributionConfig,
) -> list[str]:
    return candidate_profiles.run_base_profile(
        uv=uv,
        interpreter=interpreter,
        python_version=python_version,
        wheel=wheel,
        extracted=extracted,
        workspace=workspace,
        config=config,
        run=checked_run,
    )


def run_optional_profile(
    *,
    name: str,
    uv: str,
    interpreter: str,
    wheel: Path,
    extracted: Path,
    workspace: candidate_profiles.ProfileWorkspace,
    config: DistributionConfig,
) -> list[str]:
    return candidate_profiles.run_optional_profile(
        name=name,
        uv=uv,
        interpreter=interpreter,
        wheel=wheel,
        extracted=extracted,
        workspace=workspace,
        config=config,
        run=checked_run,
    )


def run_numpy_floor_profile(
    *,
    uv: str,
    interpreter: str,
    wheel: Path,
    extracted: Path,
    workspace: candidate_profiles.ProfileWorkspace,
    config: DistributionConfig,
) -> tuple[list[str], dict[str, str]]:
    return candidate_profiles.run_numpy_floor_profile(
        uv=uv,
        interpreter=interpreter,
        wheel=wheel,
        extracted=extracted,
        workspace=workspace,
        config=config,
        run=checked_run,
    )


def run_full_typing_profile(
    *,
    uv: str,
    interpreter: str,
    wheel: Path,
    extracted: Path,
    workspace: candidate_profiles.ProfileWorkspace,
    config: DistributionConfig,
) -> str:
    return candidate_profiles.run_full_typing_profile(
        uv=uv,
        interpreter=interpreter,
        wheel=wheel,
        extracted=extracted,
        workspace=workspace,
        config=config,
        run=checked_run,
    )


def execute_profile(
    workspace: candidate_profiles.ProfileWorkspace,
    *,
    uv: str,
    config: DistributionConfig,
    wheel: Path,
    extracted: Path,
    interpreters: dict[str, str],
) -> candidate_profiles.ProfileReceipt:
    """Execute one profile solely within its pre-declared writable root."""

    workspace.root.mkdir(parents=True)
    workspace.temporary.mkdir()
    if workspace.matplotlib_config is not None:
        workspace.matplotlib_config.mkdir()

    name = workspace.name
    checks: list[str]
    dependency_profiles: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    with command_context(
        environment=dict(workspace.environment_variables),
        log=workspace.log,
    ):
        if name.startswith("base-"):
            python_version = name.removeprefix("base-")
            checks = run_base_profile(
                uv=uv,
                interpreter=interpreters[python_version],
                python_version=python_version,
                wheel=wheel,
                extracted=extracted,
                workspace=workspace,
                config=config,
            )
        elif name == "numpy-floor-3.12":
            checks, numpy_floor = run_numpy_floor_profile(
                uv=uv,
                interpreter=interpreters[config.numpy_floor_interpreter],
                wheel=wheel,
                extracted=extracted,
                workspace=workspace,
                config=config,
            )
            dependency_profiles = (("numpy_floor", tuple(sorted(numpy_floor.items()))),)
        elif name == "generated-public-api":
            checked_run(
                [
                    interpreters[config.extras_interpreter],
                    "-I",
                    str(extracted / "tools/docs/generate_python_api.py"),
                    "--check",
                ],
                cwd=extracted,
            )
            checks = ["check:generated-public-api"]
        elif name in {"torch-3.13", "jax-3.13", "matplotlib-3.13"}:
            profile = name.removesuffix("-3.13")
            checks = run_optional_profile(
                name=profile,
                uv=uv,
                interpreter=interpreters[config.extras_interpreter],
                wheel=wheel,
                extracted=extracted,
                workspace=workspace,
                config=config,
            )
        elif name == "typing-3.13":
            checks = [
                run_full_typing_profile(
                    uv=uv,
                    interpreter=interpreters[config.extras_interpreter],
                    wheel=wheel,
                    extracted=extracted,
                    workspace=workspace,
                    config=config,
                )
            ]
        else:  # pragma: no cover - plan construction owns the closed set
            raise CandidateError(f"unknown candidate profile: {name}")

    log = workspace.log.read_text(encoding="utf-8") if workspace.log.is_file() else ""
    return candidate_profiles.ProfileReceipt(
        name=name,
        checks=tuple(checks),
        dependency_profiles=dependency_profiles,
        diagnostics=(),
        log=log,
    )


def replay_profile_logs(
    plan: tuple[candidate_profiles.ProfileWorkspace, ...],
) -> None:
    """Replay joined profile logs in frozen logical order."""

    for workspace in plan:
        if not workspace.log.is_file():
            continue
        print(f"=== Python candidate profile: {workspace.name} ===", flush=True)
        payload = workspace.log.read_text(encoding="utf-8")
        if payload:
            print(payload, end="" if payload.endswith("\n") else "\n", flush=True)


def tool_version(argv: list[str]) -> str:
    """Read one single-line tool version."""

    return checked_run(argv, capture=True).splitlines()[0]


def write_manifest(
    *,
    output: Path,
    source: SourceIdentity,
    sdist: Path,
    version: str,
    wheel_record: dict[str, Any],
    checks: list[str],
    config: DistributionConfig,
    uv: str,
    dependency_profiles: dict[str, dict[str, str]],
) -> Path:
    """Write deterministic provenance for the accepted artifact set."""

    manifest_checks = [
        "generated-public-api" if check == "check:generated-public-api" else check
        for check in checks
    ]
    artifacts = [
        {
            "filename": sdist.name,
            "kind": "sdist",
            "size": sdist.stat().st_size,
            "sha256": sha256(sdist),
        },
        wheel_record,
    ]
    manifest = {
        "format": MANIFEST_FORMAT,
        "project": "eqiora",
        "version": version,
        "acceptance": "complete",
        "source": {
            "commit": source.commit,
            "expected_tag": config.expected_tag,
            "tags": list(source.tags),
            "tree": "clean",
        },
        "build": {
            "sdist_rebuilt": True,
            "wheel_family": {
                "implementation": "CPython",
                "ordinary_gil": True,
                "versions": list(config.interpreters),
                "platform": config.wheel_platform,
                "abi3": True,
            },
            "tools": {
                "maturin": config.maturin,
                "uv": tool_version([uv, "--version"]),
                "rustc": tool_version(["rustc", f"+{config.rust}", "--version"]),
                "cargo": tool_version(["cargo", f"+{config.rust}", "--version"]),
                "pytest": config.pytest,
                "mypy": config.mypy,
                "twine": config.twine,
            },
            "dependency_profiles": dependency_profiles,
        },
        "artifacts": artifacts,
        "checks": sorted(manifest_checks),
        "nonclaims": [
            "reproducible-build-certification",
            "artifact-signature",
            "macos-or-windows",
            "free-threaded-cpython",
            "production-pypi-publication",
        ],
    }
    path = output / f"eqiora-{version}-python-candidate.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _family_admission() -> Any:
    """Import artifact-family admission without coupling it to profile execution."""

    import python_candidate_family

    return python_candidate_family


def family_inventory(directory: Path) -> tuple[dict[str, object], ...]:
    return _family_admission().family_inventory(directory)


def admit_candidate_family(directory: Path) -> Any:
    return _family_admission().admit_candidate_family(directory)


def _admit_producer_return(
    output: Path,
    *,
    sdist: Path,
    wheel: Path,
    config: DistributionConfig,
) -> tuple[dict[str, object], ...]:
    family = admit_candidate_family(output)
    if (
        family.sdist != sdist
        or family.wheel != wheel
        or family.version != config.python_version
    ):
        raise CandidateError("producer return differs from the admitted exact family")
    return family.inventory


def _require_candidate_host(output: Path) -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise CandidateError("the first Python candidate builder requires Linux x86-64")
    if output == ROOT or output.is_relative_to(ROOT):
        raise CandidateError("candidate output must remain outside the source tree")


def _require_expected_source(expected_commit: str) -> SourceIdentity:
    if GIT_SHA.fullmatch(expected_commit) is None:
        raise CandidateError("expected commit must be one full lowercase revision")
    source = source_identity()
    if source.commit != expected_commit:
        raise CandidateError("clean source commit differs from the expected revision")
    return source


def prepare_candidate(
    *,
    expected_commit: str,
    out: Path,
    require_tag: bool,
) -> Path:
    """Build one immutable sdist/stable-ABI-wheel family without acceptance metadata."""

    output = out.resolve()
    _require_candidate_host(output)
    source = _require_expected_source(expected_commit)
    config = load_config()
    if require_tag:
        require_annotated_expected_tag(source, config.expected_tag)
    if output.exists() and any(output.iterdir()):
        raise CandidateError("candidate family output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    uv = ensure_exact_uv(config.uv)
    require_executable("git")
    require_executable("rustc")
    require_executable("cargo")
    rustup = require_executable("rustup")
    scratch_parent = home_scratch_parent("python-candidate-prepare")
    with tempfile.TemporaryDirectory(
        prefix="eqiora-python-prepare-", dir=scratch_parent
    ) as temporary:
        scratch = Path(temporary)
        candidate_temporary = scratch / "tmp"
        candidate_temporary.mkdir()
        with command_context(environment={"TMPDIR": str(candidate_temporary)}):
            checked_run(
                [rustup, "toolchain", "install", config.rust, "--profile", "minimal"]
            )
            interpreter = uv_interpreter(uv, config.interpreters[0])
            sdist, wheel, extracted = build_artifacts(
                output=output,
                scratch=scratch,
                config=config,
                uv=uv,
                interpreter=interpreter,
            )
            producer_inventory = _admit_producer_return(
                output,
                sdist=sdist,
                wheel=wheel,
                config=config,
            )
            checked_run(
                [
                    uv,
                    "tool",
                    "run",
                    "--from",
                    config.twine,
                    "twine",
                    "check",
                    "--strict",
                    str(sdist),
                    str(wheel),
                ]
            )
            license_bytes = (extracted / "LICENSE").read_bytes()
            notice_bytes = (extracted / "NOTICE").read_bytes()
            version, _record = inspect_wheel(
                wheel,
                config=config,
                license_bytes=license_bytes,
                notice_bytes=notice_bytes,
            )
            if version != config.python_version:
                raise CandidateError("prepared wheel version drifted")
    admitted = admit_candidate_family(output)
    if admitted.version != config.python_version:
        raise CandidateError("prepared family identity differs from Cargo")
    if admitted.inventory != producer_inventory:
        raise CandidateError("prepared family differs from the producer return")
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    return output


def run_candidate_profiles(
    *,
    family: Any,
) -> CandidateProfileSummary:
    """Run every release profile against one retained family."""

    config = load_config()
    uv = ensure_exact_uv(config.uv)
    require_executable("rustc")
    require_executable("cargo")
    scratch_parent = home_scratch_parent("python-candidate-finalize-profiles")
    with tempfile.TemporaryDirectory(
        prefix="eqiora-python-finalize-", dir=scratch_parent
    ) as temporary:
        scratch = Path(temporary)
        extracted = safe_extract_sdist(family.sdist, scratch / "source")
        wheel = family.wheel
        interpreters = {
            version: uv_interpreter(uv, version) for version in config.interpreters
        }
        checked_run(
            [
                uv,
                "tool",
                "run",
                "--from",
                config.twine,
                "twine",
                "check",
                "--strict",
                str(family.sdist),
                str(wheel),
            ]
        )
        license_bytes = (extracted / "LICENSE").read_bytes()
        notice_bytes = (extracted / "NOTICE").read_bytes()
        version, record = inspect_wheel(
            wheel,
            config=config,
            license_bytes=license_bytes,
            notice_bytes=notice_bytes,
        )
        if version != family.version or family.version != config.python_version:
            raise CandidateError("retained wheel version disagrees with the family")
        initial_identity = candidate_payload_identity(family.sdist, wheel, extracted)
        plan = candidate_profiles.build_profile_plan(scratch, config)
        outcomes = candidate_profiles.run_profile_tasks(
            plan,
            lambda workspace: execute_profile(
                workspace,
                uv=uv,
                config=config,
                wheel=wheel,
                extracted=extracted,
                interpreters=interpreters,
            ),
        )
        replay_profile_logs(plan)
        by_name = {outcome.name: outcome for outcome in outcomes}
        failures = tuple(
            (workspace.name, by_name[workspace.name].error)
            for workspace in plan
            if by_name[workspace.name].error is not None
        )
        if failures:
            diagnostics = "\n".join(f"- {name}: {error}" for name, error in failures)
            raise CandidateError(f"candidate validation profiles failed:\n{diagnostics}")
        if candidate_payload_identity(family.sdist, wheel, extracted) != initial_identity:
            raise CandidateError("candidate profiles mutated retained artifact inputs")
        receipts = tuple(outcome.value for outcome in outcomes if outcome.value is not None)
        try:
            merged = candidate_profiles.merge_profile_receipts(
                candidate_profiles.COMPLETE_PROFILE_NAMES,
                receipts,
            )
        except ValueError as error:
            raise CandidateError(f"candidate profile receipts are invalid: {error}") from error
        dependency_profiles = {
            name: dict(values) for name, values in merged.dependency_profiles
        }
        return CandidateProfileSummary(
            config=config,
            uv=uv,
            wheel_record=record,
            checks=("twine-strict", "sdist-to-wheel-rebuild", *merged.checks),
            dependency_profiles=dependency_profiles,
        )


def _family_version_from_sdist(sdist: Path) -> str:
    prefix = "eqiora-"
    suffix = ".tar.gz"
    if not sdist.name.startswith(prefix) or not sdist.name.endswith(suffix):
        raise CandidateError("candidate sdist filename is malformed")
    version = sdist.name[len(prefix) : -len(suffix)]
    if not version:
        raise CandidateError("candidate sdist version is empty")
    return version


def _admit_candidate_profile_summary(
    profiles: object,
    *,
    artifact_root: Path,
    family: Any,
    entry_inventory: tuple[dict[str, object], ...],
) -> CandidateProfileSummary:
    """Admit one complete profile-owner result before publication side effects."""

    if not isinstance(profiles, CandidateProfileSummary):
        raise CandidateError(
            "candidate profile owner did not return CandidateProfileSummary"
        )

    try:
        current_inventory = family_inventory(artifact_root)
    except (CandidateError, OSError) as error:
        raise CandidateError("candidate family changed during finalization") from error
    if current_inventory != entry_inventory or family.inventory != entry_inventory:
        raise CandidateError("candidate family changed during finalization")

    try:
        sdist = family.sdist
        wheel = family.wheel
        version = family.version
        family_paths = tuple(artifact_root.iterdir())
    except (AttributeError, OSError, TypeError) as error:
        raise CandidateError(
            "candidate family is not the exact one-sdist/stable-ABI-wheel set"
        ) from error
    if (
        not isinstance(sdist, Path)
        or not isinstance(version, str)
        or not version
        or len(family_paths) != 2
        or not isinstance(wheel, Path)
        or set(family_paths) != {sdist, wheel}
        or sdist.name != f"eqiora-{version}.tar.gz"
        or wheel.suffix != ".whl"
    ):
        raise CandidateError(
            "candidate family is not the exact one-sdist/stable-ABI-wheel set"
        )

    try:
        config = profiles.config
        uv = profiles.uv
        reviewed = load_config()
    except (AttributeError, CandidateError, OSError, TypeError, ValueError) as error:
        raise CandidateError(
            "candidate profile summary has invalid configuration"
        ) from error
    if (
        not isinstance(config, DistributionConfig)
        or config != reviewed
        or config.python_version != version
        or config.interpreters != ("3.11", "3.12", "3.13", "3.14")
        or config.wheel_platform != "manylinux_2_17_x86_64"
        or not isinstance(uv, str)
        or not uv
    ):
        raise CandidateError("candidate profile summary has invalid configuration")

    record_members = {
        "filename",
        "kind",
        "python",
        "abi",
        "platform",
        "size",
        "sha256",
    }
    try:
        record = profiles.wheel_record
        valid_record = (
            isinstance(record, dict)
            and set(record) == record_members
            and record.get("filename") == wheel.name
            and record.get("kind") == "wheel"
            and record.get("python") == "3.11"
            and record.get("abi") == "abi3"
            and record.get("platform") == config.wheel_platform
            and isinstance(record.get("size"), int)
            and not isinstance(record.get("size"), bool)
            and record.get("size") == wheel.stat().st_size
            and record.get("size", 0) > 0
            and record.get("sha256") == sha256(wheel)
        )
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise CandidateError(
            "candidate profile summary does not bind the exact stable-ABI wheel"
        ) from error
    if not valid_record:
        raise CandidateError(
            "candidate profile summary does not bind the exact stable-ABI wheel"
        )

    try:
        checks = profiles.checks
        if (
            not isinstance(checks, tuple)
            or any(not isinstance(check, str) or not check for check in checks)
            or len(set(checks)) != len(checks)
        ):
            raise ValueError
        normalized_checks = tuple(
            "generated-public-api" if check == "check:generated-public-api" else check
            for check in checks
        )
        if len(set(normalized_checks)) != len(normalized_checks):
            raise ValueError
        required_checks = set().union(
            *(PROFILE_CHECKS[profile] for profile in REQUIRED_PROFILES)
        )
        producer_checks = required_checks | {
            f"cp{python.replace('.', '')}:packaged-exact-cylinder-model-demo"
            for python in config.interpreters
        }
        if set(normalized_checks) != producer_checks:
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise CandidateError(
            "candidate profile summary does not contain the exact complete check set"
        ) from error

    numpy_version = config.numpy_floor.split("==", maxsplit=1)[1]
    expected_dependency_profiles = {
        "numpy_floor": {
            "python": config.numpy_floor_interpreter,
            "requirement": config.numpy_floor,
            "observed": numpy_version,
            "profile": (
                f"cp{config.numpy_floor_interpreter.replace('.', '')}:"
                f"numpy-{numpy_version}-floor"
            ),
        }
    }
    if profiles.dependency_profiles != expected_dependency_profiles:
        raise CandidateError(
            "candidate profile summary has invalid dependency profiles"
        )
    return profiles


def finalize_candidate(
    *,
    expected_commit: str,
    artifacts: Path,
    manifest_out: Path,
) -> Path:
    """Run the release profiles and retain one closed v6 manifest."""

    artifact_root = artifacts
    metadata_root = manifest_out
    _require_candidate_host(artifact_root.resolve())
    _require_candidate_host(metadata_root.resolve())
    roots = tuple(path.resolve() for path in (artifact_root, metadata_root))
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise CandidateError("family and metadata paths must be disjoint")
    source = _require_expected_source(expected_commit)
    family = admit_candidate_family(artifact_root)
    entry_inventory = family_inventory(artifact_root)
    if metadata_root.exists() and any(metadata_root.iterdir()):
        raise CandidateError("candidate metadata output directory must be empty")
    metadata_root.mkdir(parents=True, exist_ok=True)
    try:
        profiles = run_candidate_profiles(family=family)
        profiles = _admit_candidate_profile_summary(
            profiles,
            artifact_root=artifact_root,
            family=family,
            entry_inventory=entry_inventory,
        )
        sdist = family.sdist
        manifest = write_manifest(
            output=metadata_root,
            source=source,
            sdist=sdist,
            version=_family_version_from_sdist(sdist),
            wheel_record=profiles.wheel_record,
            checks=list(profiles.checks),
            config=profiles.config,
            uv=profiles.uv,
            dependency_profiles=profiles.dependency_profiles,
        )
        if family_inventory(artifact_root) != entry_inventory:
            raise CandidateError("candidate family changed during finalization")
        candidate = load_candidate_family(
            manifest,
            artifact_root,
            requested_profiles=REQUIRED_PROFILES,
        )
        verify_artifacts(candidate, artifact_root)
        if {path.name for path in metadata_root.iterdir()} != {manifest.name}:
            raise CandidateError("candidate metadata output is not the exact manifest")
        return manifest
    except Exception:
        for path in metadata_root.iterdir():
            if path.is_file() and not path.is_symlink():
                path.unlink()
        raise


def parse_args() -> argparse.Namespace:
    """Parse the closed prepare/finalize release handoff."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--expected-commit", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--require-tag", action="store_true")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--expected-commit", required=True)
    finalize.add_argument("--artifacts", type=Path, required=True)
    finalize.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    arguments = parse_args()
    try:
        if arguments.command == "prepare":
            result = prepare_candidate(
                expected_commit=arguments.expected_commit,
                out=arguments.out,
                require_tag=arguments.require_tag,
            )
        else:
            result = finalize_candidate(
                expected_commit=arguments.expected_commit,
                artifacts=arguments.artifacts,
                manifest_out=arguments.manifest_out,
            )
    except (CandidateError, OSError, subprocess.CalledProcessError) as error:
        print(f"Python candidate failed: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
