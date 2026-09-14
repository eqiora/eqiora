#!/usr/bin/env python3
"""Build, install, and test the Python package in one ephemeral interpreter."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT
TESTS = ROOT / "bindings/python/tests"
GMSH_EVIDENCE = tuple(
    TESTS / name
    for name in (
        "test_gmsh_meshing.py",
        "test_exact_cylinder_stokes_result.py",
    )
)
_PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
_DISTRIBUTION = _PROJECT["tool"]["eqiora-distribution"]
BUILD_TOOLS = (
    _PROJECT["build-system"]["requires"][0],
    _DISTRIBUTION["pytest"],
)


def venv_python(environment: Path) -> Path:
    """Return the platform-native interpreter path for one virtual environment."""
    if os.name == "nt":
        return environment / "Scripts/python.exe"
    return environment / "bin/python"


def venv_environment(
    environment: Path,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Activate a virtual environment for child tools without a shell."""
    child_environment = dict(os.environ if base is None else base)
    scripts = venv_python(environment).parent
    inherited_path = child_environment.get("PATH")
    child_environment["PATH"] = (
        str(scripts)
        if not inherited_path
        else os.pathsep.join((str(scripts), inherited_path))
    )
    child_environment["VIRTUAL_ENV"] = str(environment)
    return child_environment


def run(
    argv: list[str],
    *,
    cwd: Path = ROOT,
    virtual_environment: Path | None = None,
) -> None:
    child_environment = os.environ.copy()
    child_environment.pop("PYTHONPATH", None)
    child_environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    if virtual_environment is not None:
        child_environment = venv_environment(
            virtual_environment,
            base=child_environment,
        )
    subprocess.run(argv, cwd=cwd, env=child_environment, check=True)


def uv_gate_command(uv: str, python: str) -> list[str]:
    """Build the current source tree even when its package version is unchanged."""
    return [
        uv,
        "run",
        "--directory",
        str(PACKAGE),
        "--isolated",
        "--no-editable",
        "--reinstall-package",
        "eqiora",
        "--with",
        BUILD_TOOLS[1],
        "--python",
        python,
        "python",
        "-m",
        "pytest",
        "-q",
        str(TESTS),
        *(
            argument
            for evidence in GMSH_EVIDENCE
            for argument in ("--ignore", str(evidence))
        ),
    ]


def _uv_gmsh_gate_command(uv: str, python: str) -> list[str]:
    return [
        uv,
        "run",
        "--directory",
        str(PACKAGE),
        "--isolated",
        "--no-editable",
        "--reinstall-package",
        "eqiora",
        "--extra",
        "gmsh",
        "--with",
        BUILD_TOOLS[1],
        "--python",
        python,
        "python",
        "-m",
        "pytest",
        "-q",
        *(str(evidence) for evidence in GMSH_EVIDENCE),
    ]


def main() -> int:
    try:
        if uv := shutil.which("uv"):
            run(uv_gate_command(uv, sys.executable))
            run(_uv_gmsh_gate_command(uv, sys.executable))
            return 0
        with tempfile.TemporaryDirectory(prefix="eqiora-python-gate-") as directory:
            environment = Path(directory)
            run([sys.executable, "-m", "venv", str(environment)])
            python = str(venv_python(environment))
            run(
                [python, "-m", "pip", "install", *BUILD_TOOLS],
                virtual_environment=environment,
            )
            run(
                [python, "-m", "pip", "install", "--no-build-isolation", "."],
                cwd=PACKAGE,
                virtual_environment=environment,
            )
            run(
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    str(TESTS),
                    *(
                        argument
                        for evidence in GMSH_EVIDENCE
                        for argument in ("--ignore", str(evidence))
                    ),
                ],
                cwd=PACKAGE,
                virtual_environment=environment,
            )
            run(
                [python, "-m", "pip", "install", "--no-build-isolation", ".[gmsh]"],
                cwd=PACKAGE,
                virtual_environment=environment,
            )
            run(
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    *(str(evidence) for evidence in GMSH_EVIDENCE),
                ],
                cwd=PACKAGE,
                virtual_environment=environment,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Python package gate failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
