#!/usr/bin/env python3
"""Project the current command-line interface into MDX."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path


SOURCE_SHA_PATTERN = re.compile(r"[0-9a-f]{40}", flags=re.ASCII)
GIT_IDENTITY_ENVIRONMENT = {
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
}
OUTPUTS = {
    "cli": Path("docs/site/src/content/docs/reference/cli/index.mdx"),
}


class ProjectionError(RuntimeError):
    """A live interface or committed projection violates the documentation contract."""


def _regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    try:
        details = path.stat()
    except FileNotFoundError as error:
        raise ProjectionError(f"missing {label}: {path}") from error
    if not stat.S_ISREG(details.st_mode):
        raise ProjectionError(f"{label} must be a regular file: {path}")
    if executable and details.st_mode & 0o111 == 0:
        raise ProjectionError(f"{label} is not executable: {path}")
    return path.resolve(strict=True)


def _text(path: Path, label: str) -> str:
    source = _regular_file(path, label)
    try:
        payload = source.read_bytes()
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProjectionError(f"{label} is not UTF-8: {path}") from error
    if b"\r" in payload or not text.endswith("\n"):
        raise ProjectionError(f"{label} must be LF-only and end in one LF: {path}")
    return text


def _command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "COLUMNS": "100",
            "LANG": "C",
            "LC_ALL": "C",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "TZ": "UTC",
        }
    )
    return environment


def _git_identity_observation(
    repository: Path, arguments: list[str], label: str
) -> str:
    environment = os.environ.copy()
    for name in GIT_IDENTITY_ENVIRONMENT:
        environment.pop(name, None)
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            cwd=repository,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProjectionError(f"could not observe {label}: {error}") from error
    if completed.returncode != 0:
        raise ProjectionError(f"could not observe {label}")
    if completed.stderr:
        raise ProjectionError(f"Git wrote to stderr while observing {label}")
    try:
        output = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProjectionError(f"Git emitted non-UTF-8 {label}") from error
    if (
        b"\r" in completed.stdout
        or not output.endswith("\n")
        or output.count("\n") != 1
    ):
        raise ProjectionError(f"Git emitted malformed {label}")
    return output.removesuffix("\n")


def _admit_source_identity(repository: Path, source_sha: str) -> None:
    if SOURCE_SHA_PATTERN.fullmatch(source_sha) is None:
        raise ProjectionError(
            "source SHA must be exactly 40 lowercase hexadecimal characters"
        )
    environment_sha = os.environ.get("EQIORA_SITE_SOURCE_SHA")
    if environment_sha is not None and environment_sha != source_sha:
        raise ProjectionError("source SHA disagrees with EQIORA_SITE_SOURCE_SHA")

    if not os.path.lexists(repository / ".git"):
        return

    top_level_text = _git_identity_observation(
        repository, ["rev-parse", "--show-toplevel"], "canonical Git top level"
    )
    top_level_path = Path(top_level_text)
    try:
        top_level = top_level_path.resolve(strict=True)
    except OSError as error:
        raise ProjectionError(
            "Git top level is not a canonical existing path"
        ) from error
    if (
        not top_level_path.is_absolute()
        or top_level_text != str(top_level)
        or top_level != repository
    ):
        raise ProjectionError("Git top level disagrees with the repository root")

    head = _git_identity_observation(
        repository, ["rev-parse", "--verify", "HEAD"], "Git HEAD"
    )
    if SOURCE_SHA_PATTERN.fullmatch(head) is None:
        raise ProjectionError("Git HEAD is not a canonical 40-character commit")
    if head != source_sha:
        raise ProjectionError("Git HEAD disagrees with the source SHA")


def _run(
    binary: Path,
    arguments: list[str],
    *,
    cwd: Path,
    label: str,
) -> str:
    try:
        completed = subprocess.run(
            [str(binary), *arguments],
            cwd=cwd,
            env=_command_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProjectionError(f"could not capture {label}: {error}") from error
    if completed.returncode != 0:
        raise ProjectionError(f"{label} exited with {completed.returncode}")
    if completed.stderr:
        raise ProjectionError(f"{label} wrote to stderr")
    if len(completed.stdout) > 2 * 1024 * 1024:
        raise ProjectionError(f"{label} exceeded the documentation capture limit")
    try:
        output = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProjectionError(f"{label} did not emit UTF-8") from error
    if b"\r" in completed.stdout or not output.endswith("\n"):
        raise ProjectionError(f"{label} must emit LF-terminated output")
    return output


def _workspace_version(repository: Path) -> str:
    cargo_path = repository / "Cargo.toml"
    cargo = tomllib.loads(_text(cargo_path, "workspace manifest"))
    try:
        version = cargo["workspace"]["package"]["version"]
    except (KeyError, TypeError) as error:
        raise ProjectionError("Cargo.toml has no workspace.package.version") from error
    if not isinstance(version, str) or not version:
        raise ProjectionError("workspace.package.version is not a non-empty string")
    return version


def _capture_cli(repository: Path, binary: Path, version: str) -> dict[str, str]:
    cwd = repository.parent
    root_help = _run(binary, ["--help"], cwd=cwd, label="eqiora --help")
    check_help = _run(binary, ["check", "--help"], cwd=cwd, label="eqiora check --help")
    observed_version = _run(binary, ["--version"], cwd=cwd, label="eqiora --version")
    expected_version = f"eqiora {version}\n"
    if observed_version != expected_version:
        raise ProjectionError(
            "eqiora --version disagrees with workspace.package.version: "
            f"expected {expected_version!r}, observed {observed_version!r}"
        )
    return {
        "root_help": root_help,
        "check_help": check_help,
        "version": observed_version,
    }


def _code_block(value: str) -> str:
    rendered = value.removesuffix("\n")
    if "```" in rendered:
        raise ProjectionError(
            "captured command output cannot be represented in the fixed MDX fence"
        )
    return rendered


def _cli_page(cli: dict[str, str]) -> str:
    return f"""---
title: Command-line interface
description: Install the command-line tool, check a model, and read compiler diagnostics.
---

{{/* Generated by tools/docs/generate_interface_reference.py; do not edit. */}}

Use `eqiora check` to compile a local `.eqi` file and find syntax, name,
and unit errors before running a simulation.

## Install the command

The command is a Rust executable. From the Eqiora source checkout used for your
[Python installation](/get-started/), install it with Cargo:

```bash
cargo install --locked --path crates/eqiora --features cli --bin eqiora
```

Keep Cargo's binary directory on `PATH` (normally `~/.cargo/bin`). Installing
the Python package alone does not install this command.

## Check a model

Save this as `decay.eqi`:

```eqiora
model decay(parameter rate: 1 / s = 1) {{
    state x: 1;
    initial {{ x = 1; }}
    relation flow {{
        derivative(x) + rate * x = 0;
    }}
}}
```

```bash
eqiora check decay.eqi
```

A successful check exits with status 0. It compiles the model; it does not
integrate the equation or write a time series. Continue with the
[Python run example](/reference/python/#compile-and-run) to compute `x(t)`.

Change the parameter unit from `1 / s` to `s` and run the check again. The
compiler reports the incompatible dimensions and exits with a nonzero status.
Fix the unit before choosing a numerical method.

For more syntax, see the [language reference](/reference/language/).

## Version

```console
$ eqiora --version
{_code_block(cli["version"])}
```

## Command help

```console
$ eqiora --help
{_code_block(cli["root_help"])}
```

## `check` help

```console
$ eqiora check --help
{_code_block(cli["check_help"])}
```
"""


def _render(repository: Path, eqiora_binary: Path) -> dict[str, str]:
    version = _workspace_version(repository)
    cli = _capture_cli(repository, eqiora_binary, version)
    return {"cli": _cli_page(cli)}


def _write_or_check(repository: Path, rendered: dict[str, str], *, check: bool) -> None:
    failures: list[str] = []
    for name, relative in OUTPUTS.items():
        target = repository / relative
        payload = rendered[name].encode("utf-8")
        if b"\r" in payload or not payload.endswith(b"\n"):
            raise ProjectionError(
                f"generated {name} projection is not canonical UTF-8/LF"
            )
        if check:
            try:
                observed = target.read_bytes()
            except FileNotFoundError:
                failures.append(f"missing generated projection: {relative}")
                continue
            if observed != payload:
                failures.append(f"generated projection is stale: {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    if failures:
        raise ProjectionError("\n".join(failures))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root containing Cargo.toml and the output paths",
    )
    parser.add_argument("--eqiora-binary", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare generated bytes without changing the checkout",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    repository = arguments.repository.resolve(strict=True)
    _admit_source_identity(repository, arguments.source_sha)
    eqiora_binary = _regular_file(
        arguments.eqiora_binary, "eqiora binary", executable=True
    )
    rendered = _render(repository, eqiora_binary)
    _write_or_check(repository, rendered, check=arguments.check)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProjectionError as error:
        print(f"interface reference projection failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
