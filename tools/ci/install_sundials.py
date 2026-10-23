#!/usr/bin/env python3
"""Build the reviewed serial double-precision IDA release in an empty prefix."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = "7.8.0"
URL = f"https://github.com/llnl/sundials/releases/download/v{VERSION}/sundials-{VERSION}.tar.gz"
# Digest published on the upstream GitHub release asset, independently checked.
SHA256 = "69ec92653e998e4841b59d363b3abf21299251991390f52917402737164ca574"


def install(prefix: Path) -> None:
    prefix = prefix.resolve()
    if prefix.exists():
        raise ValueError(f"installation prefix must not already exist: {prefix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Build scratch follows the requested home-backed installation, never /tmp.
    with tempfile.TemporaryDirectory(prefix="sundials-build-", dir=prefix.parent) as scratch:
        root = Path(scratch)
        archive = root / "source.tar.gz"
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with archive.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != SHA256:
            raise ValueError("SUNDIALS release archive checksum mismatch; refusing to extract or build")
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
        subprocess.run([
            "cmake", "-S", str(root / f"sundials-{VERSION}"), "-B", str(root / "build"),
            "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DCMAKE_INSTALL_LIBDIR=lib", "-DSUNDIALS_PRECISION=double",
            "-DSUNDIALS_INDEX_SIZE=64", "-DBUILD_SHARED_LIBS=ON", "-DBUILD_STATIC_LIBS=OFF",
            "-DBUILD_IDA=ON", "-DBUILD_IDAS=OFF", "-DBUILD_ARKODE=OFF",
            "-DBUILD_CVODE=OFF", "-DBUILD_CVODES=OFF", "-DBUILD_KINSOL=OFF",
            "-DEXAMPLES_ENABLE_C=OFF", "-DEXAMPLES_ENABLE_CXX=OFF",
        ], check=True)
        subprocess.run(["cmake", "--build", str(root / "build"), "--parallel", "4"], check=True)
        subprocess.run(["cmake", "--install", str(root / "build")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    install(args.prefix)


if __name__ == "__main__":
    main()
