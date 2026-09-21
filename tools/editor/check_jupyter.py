#!/usr/bin/env python3
"""Rebuild the notebook frontend in scratch and reject stale packaged assets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = Path("editor/jupyter")
GRAMMAR = Path("editor/eqiora/syntaxes/eqiora.tmLanguage.json")


def verify_assets(expected: Path, generated: Path) -> None:
    """Compare the actual distributable output, including missing and extra files."""

    def files(directory: Path) -> dict[str, bytes]:
        return {
            path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*")
            if path.is_file()
        }

    before, after = files(expected), files(generated)
    changed = sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )
    if not before or not after or changed:
        raise ValueError(
            "stale Jupyter assets; run npm run build in editor/jupyter and commit wheel-data: "
            + ", ".join(changed)
        )


def main() -> int:
    try:
        package = json.loads((ROOT / FRONTEND / "package.json").read_text())
        output = Path(package["jupyterlab"]["outputDir"])
        builder_version = package["devDependencies"]["@jupyterlab/builder"]
        scratch_root = Path.home() / ".cache/eqiora/jupyter-build"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="check-", dir=scratch_root
        ) as temporary:
            scratch = Path(temporary)
            # Respect the repository's ignored build/cache paths. Current tracked
            # edits and new nonignored source files are both inputs to the build.
            paths = (
                subprocess.check_output(
                    [
                        "git",
                        "ls-files",
                        "-z",
                        "--cached",
                        "--others",
                        "--exclude-standard",
                        "--",
                        str(FRONTEND),
                        str(GRAMMAR),
                    ],
                    cwd=ROOT,
                )
                .decode()
                .split("\0")
            )
            for name in filter(None, paths):
                relative = Path(name)
                if (
                    relative.is_relative_to(FRONTEND / output)
                    or not (ROOT / relative).exists()
                ):
                    continue
                target = scratch / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            environment = scratch / "venv"
            python = environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            if uv := shutil.which("uv"):
                subprocess.run(
                    [uv, "venv", "--python", sys.executable, str(environment)],
                    check=True,
                )
                subprocess.run(
                    [
                        uv,
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        f"jupyterlab=={builder_version}",
                    ],
                    check=True,
                )
            else:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(environment)], check=True
                )
                subprocess.run(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        f"jupyterlab=={builder_version}",
                    ],
                    check=True,
                )
            child_env = dict(
                os.environ,
                PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
            )
            npx = shutil.which("npx")
            if npx is None:
                raise ValueError("npx is required for the maintainer frontend check")
            for arguments in (
                ("ci", "--ignore-scripts"),
                ("run", "typecheck"),
                ("run", "build"),
            ):
                subprocess.run(
                    [npx, "--yes", package["packageManager"], *arguments],
                    cwd=scratch / FRONTEND,
                    env=child_env,
                    check=True,
                )
            verify_assets(ROOT / FRONTEND / output, scratch / FRONTEND / output)
        print("Jupyter packaged assets match the locked frontend build")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Jupyter asset check failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
