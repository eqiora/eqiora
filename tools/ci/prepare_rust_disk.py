#!/usr/bin/env python3
"""Check Stable's disk floor and conditionally remove two unused hosted SDKs."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

MIN_FREE_BYTES = 32 * 1024**3


def sdk_allocated_bytes(root: Path) -> int | None:
    """Measure allocated bytes without following links or double-counting hardlinks."""
    try:
        root.lstat()
    except FileNotFoundError:
        return None
    pending = [root]
    seen = set()
    total = 0
    while pending:
        path = pending.pop()
        info = path.lstat()
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        total += info.st_blocks * 512
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)
    return total


def prepare() -> None:
    required = {
        "GITHUB_ACTIONS": "true",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "RUNNER_OS": "Linux",
    }
    if any(os.environ.get(name) != value for name, value in required.items()):
        raise RuntimeError(
            "requires GITHUB_ACTIONS=true, RUNNER_ENVIRONMENT=github-hosted and "
            "RUNNER_OS=Linux; refusing to measure or change this host"
        )

    before = shutil.disk_usage(".").free
    print(f"rust-disk-preflight: free_bytes_before={before} required_bytes={MIN_FREE_BYTES}", flush=True)
    android = sdk_allocated_bytes(Path("/usr/local/lib/android"))
    dotnet = sdk_allocated_bytes(Path("/usr/share/dotnet"))
    for name, size in (("/usr/local/lib/android", android), ("/usr/share/dotnet", dotnet)):
        print(f"rust-disk-preflight: sdk={name} present={size is not None} allocated_bytes={size}", flush=True)

    if before < MIN_FREE_BYTES:
        if android is not None:
            subprocess.run(["sudo", "-n", "rm", "-rf", "--", "/usr/local/lib/android"], check=True)
        if dotnet is not None:
            subprocess.run(["sudo", "-n", "rm", "-rf", "--", "/usr/share/dotnet"], check=True)

    after = shutil.disk_usage(".").free
    print(f"rust-disk-preflight: free_bytes_after={after} required_bytes={MIN_FREE_BYTES}", flush=True)
    if after < MIN_FREE_BYTES:
        raise RuntimeError(
            f"insufficient disk space before Rust compilation: available_bytes={after} "
            f"required_bytes={MIN_FREE_BYTES}; only the two unused SDK roots may be removed"
        )


def main() -> int:
    try:
        prepare()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"rust-disk-preflight: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
