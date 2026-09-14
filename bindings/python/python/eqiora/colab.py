"""Prepare a coherent Google Colab runtime for Eqiora's notebook viewer."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from ctypes.util import find_library
from importlib import import_module
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from typing import NamedTuple

__all__ = ["prepare"]

_SUPPORTED_PYTHON = tuple((3, minor) for minor in range(11, 15))
_SUPPORTED_DISTRIBUTIONS = {
    "anywidget": "0.11.0",
    "gmsh": "4.15.2",
}
_DISTRIBUTION_MODULES = {
    "eqiora": ("eqiora", "eqiora/__init__.py"),
    "numpy": ("numpy", "numpy/__init__.py"),
    "gmsh": ("gmsh", "gmsh.py"),
    "anywidget": ("anywidget", "anywidget/__init__.py"),
    "ipywidgets": ("ipywidgets", "ipywidgets/__init__.py"),
    "traitlets": ("traitlets", "traitlets/__init__.py"),
}


class _Identity(NamedTuple):
    version: str | None
    file: str


def prepare() -> None:
    """Validate and prepare the current runtime for Eqiora's Colab viewer."""

    python = sys.version_info[:2]
    implementation = sys.implementation.name
    if implementation != "cpython" or python not in _SUPPORTED_PYTHON:
        supported = ", ".join(f"{major}.{minor}" for major, minor in _SUPPORTED_PYTHON)
        raise RuntimeError(
            f"Eqiora supports CPython {supported}; this runtime is "
            f"{implementation} {python[0]}.{python[1]}"
        )

    colab = _is_colab()
    installed = _installed_identities()
    _check_supported_distributions(installed)
    loaded = _loaded_identities()
    conflicts = _identity_conflicts(loaded, installed)
    if conflicts:
        _print_diagnostic(colab=colab, installed=installed, loaded=loaded)
        if not colab:
            names = ", ".join(conflicts)
            raise RuntimeError(
                f"installed packages changed under loaded modules ({names}); restart Python"
            )
        _restart_runtime()
        raise RuntimeError("Colab restart did not terminate the runtime")

    if colab:
        _provide_glu()
        _enable_custom_widget_manager()
    _print_diagnostic(colab=colab, installed=installed, loaded=loaded)


def _is_colab() -> bool:
    try:
        return find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _installed_identities() -> dict[str, _Identity]:
    identities: dict[str, _Identity] = {}
    packages = {
        owner: _distribution(owner)
        for owner, _relative in _DISTRIBUTION_MODULES.values()
    }
    for name, (owner, relative) in _DISTRIBUTION_MODULES.items():
        package = packages[owner]
        identities[name] = _Identity(
            version=package.version,
            file=str(Path(package.locate_file(relative)).resolve()),
        )
    return identities


def _distribution(name: str) -> Distribution:
    try:
        return distribution(name)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"Eqiora's Colab runtime requires the {name!r} distribution"
        ) from error


def _loaded_identities() -> dict[str, _Identity]:
    identities: dict[str, _Identity] = {}
    for name, (_owner, _relative) in _DISTRIBUTION_MODULES.items():
        module = sys.modules.get(name)
        if module is None:
            continue
        version = getattr(module, "__version__", None)
        file = getattr(module, "__file__", None)
        identities[name] = _Identity(
            version=str(version) if version is not None else None,
            file=str(Path(file).resolve()) if file is not None else "",
        )
    return identities


def _check_supported_distributions(installed: dict[str, _Identity]) -> None:
    for name, supported in _SUPPORTED_DISTRIBUTIONS.items():
        observed = installed[name].version
        if observed != supported:
            raise RuntimeError(
                f"Eqiora requires {name}=={supported}; this runtime has {observed}"
            )


def _identity_conflicts(
    loaded: dict[str, _Identity], installed: dict[str, _Identity]
) -> tuple[str, ...]:
    return tuple(
        name
        for name, identity in loaded.items()
        if identity.file != installed[name].file
        or (
            identity.version is not None and identity.version != installed[name].version
        )
    )


def _provide_glu() -> None:
    if find_library("GLU") is not None:
        return
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "libglu1-mesa"],
        check=True,
    )


def _enable_custom_widget_manager() -> None:
    output = import_module("google.colab.output")
    output.enable_custom_widget_manager()


def _restart_runtime() -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def _print_diagnostic(
    *,
    colab: bool,
    installed: dict[str, _Identity],
    loaded: dict[str, _Identity],
) -> None:
    print(
        {
            "runtime": {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "colab": colab,
            },
            "packages": {
                name: {
                    "installed": identity._asdict(),
                    "loaded": (loaded[name]._asdict() if name in loaded else None),
                }
                for name, identity in installed.items()
            },
        },
        flush=True,
    )
