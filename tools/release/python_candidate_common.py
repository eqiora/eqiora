"""Shared private runtime for Python candidate construction and validation."""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class CandidateError(RuntimeError):
    """The requested artifact set is not an acceptable release candidate."""


def sha256(path: Path) -> str:
    """Return a lowercase SHA-256 file identity."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash path identity and contents for one extracted regular source tree."""

    digest = hashlib.sha256()
    paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            raise CandidateError(
                f"candidate source tree contains a symbolic link: {relative!r}"
            )
        if path.is_dir():
            digest.update(b"directory\0" + relative + b"\0")
            continue
        if not path.is_file():
            raise CandidateError(
                f"candidate source tree contains a non-regular path: {relative!r}"
            )
        digest.update(b"file\0" + relative + b"\0")
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def candidate_payload_identity(
    sdist: Path,
    wheels: dict[str, Path],
    extracted: Path,
) -> tuple[tuple[str, str], ...]:
    """Freeze every shared read-only input consumed by validation profiles."""

    return (
        ("sdist", sha256(sdist)),
        *((f"wheel-{version}", sha256(wheels[version])) for version in sorted(wheels)),
        ("extracted-source", source_tree_sha256(extracted)),
    )


def python_distribution_version(cargo_version: str) -> str:
    """Admit one canonical stable SemVer identity for Cargo and Python."""

    components = cargo_version.split(".")
    if len(components) != 3 or any(
        not component
        or not component.isascii()
        or not component.isdecimal()
        or (component != "0" and component.startswith("0"))
        for component in components
    ):
        raise CandidateError(f"invalid Cargo release version: {cargo_version}")
    return cargo_version


@dataclass(frozen=True)
class DistributionConfig:
    """Reviewed Python distribution inputs from ``pyproject.toml``."""

    cargo_version: str
    interpreters: tuple[str, ...]
    wheel_platform: str
    extras_interpreter: str
    numpy_floor_interpreter: str
    numpy_floor: str
    uv: str
    maturin: str
    pytest: str
    mypy: str
    twine: str
    torch: str
    jax: tuple[str, ...]
    matplotlib: str
    rust: str

    @property
    def python_version(self) -> str:
        """Normalized distribution version derived from Cargo."""

        return python_distribution_version(self.cargo_version)

    @property
    def expected_tag(self) -> str:
        """Only release tag admitted for this authored version."""

        return f"v{self.python_version}"


_COMMAND_ENVIRONMENT: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "candidate_command_environment", default={}
)
_COMMAND_LOG: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "candidate_command_log", default=None
)


@contextlib.contextmanager
def command_context(
    *, environment: dict[str, str], log: Path | None = None
) -> Iterator[None]:
    """Apply task-local subprocess environment and output capture."""

    environment_token = _COMMAND_ENVIRONMENT.set(environment)
    log_token = _COMMAND_LOG.set(log)
    try:
        yield
    finally:
        _COMMAND_LOG.reset(log_token)
        _COMMAND_ENVIRONMENT.reset(environment_token)


def checked_run(
    argv: list[str],
    *,
    cwd: Path = ROOT,
    capture: bool = False,
    extra_environment: dict[str, str] | None = None,
) -> str:
    """Run one shell-free command under a source-isolated environment."""

    environment = os.environ.copy()
    environment.pop("DISPLAY", None)
    environment.pop("MATPLOTLIBRC", None)
    environment.pop("MPLCONFIGDIR", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    environment.update(_COMMAND_ENVIRONMENT.get())
    if extra_environment is not None:
        environment.update(extra_environment)

    log = _COMMAND_LOG.get()
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    if capture:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = completed.stdout
        if log is not None and output:
            with log.open("a", encoding="utf-8") as destination:
                destination.write(output)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode, argv, output=output
            )
        return output.strip()

    if log is None:
        subprocess.run(argv, cwd=cwd, env=environment, check=True)
    else:
        with log.open("ab") as destination:
            subprocess.run(
                argv,
                cwd=cwd,
                env=environment,
                check=True,
                stdout=destination,
                stderr=subprocess.STDOUT,
            )
    return ""


def home_scratch_parent(namespace: str) -> Path:
    """Return a writable scratch parent below the resolved home filesystem."""

    home = Path.home().resolve()
    lane_root = os.environ.get("EQIORA_VERIFY_LANE_ROOT")
    temporary_root = os.environ.get("TMPDIR")
    if lane_root:
        base = Path(lane_root).expanduser().resolve()
        if not base.is_relative_to(home):
            raise CandidateError(
                "candidate scratch must remain below the home directory"
            )
        parent = base / namespace
    elif temporary_root and Path(temporary_root).expanduser().resolve().is_relative_to(
        home
    ):
        parent = Path(temporary_root).expanduser().resolve() / namespace
    else:
        parent = home / ".cache" / "eqiora" / namespace
    parent.mkdir(parents=True, exist_ok=True)
    return parent
