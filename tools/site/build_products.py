#!/usr/bin/env python3
"""Build the compiled products consumed by the exact documentation site once."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


RECEIPT_SCHEMA = "eqiora.site.build-products/v1"
REQUIRED_CONSUMERS = {
    "eqiora-cli": "interface-reference",
    "rust-reference": "site-assembly",
    "xtask": "facade-admission",
}
REQUIRED_DOMAINS = {
    "rust-executables": "cargo-dev",
    "rust-reference": "cargo-doc/rustdoc-all-features",
}


@dataclass(frozen=True)
class Product:
    name: str
    consumer: str
    output: str


@dataclass(frozen=True)
class BuildStep:
    name: str
    domain: str
    command: tuple[str, ...]
    products: tuple[Product, ...]
    environment: tuple[tuple[str, str], ...] = ()


def plan(
    scratch: Path, *, cargo: str, rust_toolchain: str
) -> tuple[BuildStep, ...]:
    target = scratch / "cargo-target"
    rustdoc_target = scratch / "rustdoc-target"
    rust_environment = (("RUSTUP_TOOLCHAIN", rust_toolchain),)
    return (
        BuildStep(
            name="rust-executables",
            domain="cargo-dev",
            command=(
                cargo,
                "build",
                "--locked",
                "--profile",
                "dev",
                "--bin",
                "eqiora",
                "--bin",
                "xtask",
                "--features",
                "eqiora/cli",
                "--target-dir",
                str(target),
                "-p",
                "eqiora",
                "-p",
                "xtask",
            ),
            products=(
                Product(
                    "eqiora-cli", "interface-reference", str(target / "debug/eqiora")
                ),
                Product("xtask", "facade-admission", str(target / "debug/xtask")),
            ),
            environment=(
                *rust_environment,
                ("CARGO_PROFILE_DEV_DEBUG", "0"),
                ("CARGO_PROFILE_DEV_INCREMENTAL", "false"),
                ("CARGO_PROFILE_DEV_OPT_LEVEL", "0"),
            ),
        ),
        BuildStep(
            name="rust-reference",
            domain="cargo-doc/rustdoc-all-features",
            command=(
                cargo,
                "doc",
                "--locked",
                "-p",
                "eqiora",
                "--lib",
                "--no-deps",
                "--all-features",
                "--target-dir",
                str(rustdoc_target),
            ),
            products=(
                Product(
                    "rust-reference",
                    "site-assembly",
                    str(rustdoc_target / "doc/eqiora/index.html"),
                ),
            ),
            environment=rust_environment,
        ),
    )


def validate(build_plan: Sequence[BuildStep]) -> None:
    step_names: set[str] = set()
    product_names: set[str] = set()
    cargo_executables: set[str] = set()
    rust_toolchains: set[str] = set()
    for step in build_plan:
        if not step.name or step.name in step_names:
            raise ValueError(f"duplicate or empty build step: {step.name!r}")
        step_names.add(step.name)
        if not step.domain or not step.command or not step.products:
            raise ValueError(f"incomplete build step: {step.name}")
        if step.domain != REQUIRED_DOMAINS.get(step.name):
            raise ValueError(
                f"build step has the wrong compilation domain: {step.name}"
            )
        if step.name == "rust-executables":
            profile = step.command.index("--profile") if "--profile" in step.command else -1
            if (
                "--release" in step.command
                or profile < 0
                or step.command[profile + 1 : profile + 2] != ("dev",)
            ):
                raise ValueError("executable step is outside its development profile")
        if step.name == "rust-reference" and "--release" in step.command:
            raise ValueError("Rust reference step changed its documentation profile")
        if step.name == "rust-reference" and "--all-features" not in step.command:
            raise ValueError("Rust reference step lost its all-features closure")
        environment_names = [name for name, _ in step.environment]
        if len(environment_names) != len(set(environment_names)):
            raise ValueError(f"duplicate environment binding: {step.name}")
        toolchains = [
            value for name, value in step.environment if name == "RUSTUP_TOOLCHAIN"
        ]
        if len(toolchains) != 1 or not toolchains[0]:
            raise ValueError(f"build step lacks one Rust toolchain: {step.name}")
        rust_toolchains.add(toolchains[0])
        if step.name in {"rust-executables", "rust-reference"}:
            if not step.command[0]:
                raise ValueError(f"build step lacks a Cargo executable: {step.name}")
            cargo_executables.add(step.command[0])
        for product in step.products:
            if not product.name or product.name in product_names:
                raise ValueError(f"duplicate or empty product: {product.name!r}")
            product_names.add(product.name)
            expected_consumer = REQUIRED_CONSUMERS.get(product.name)
            if product.consumer != expected_consumer or not product.output:
                raise ValueError(
                    f"product has a wrong consumer or output: {product.name}"
                )
    if step_names != REQUIRED_DOMAINS.keys():
        missing = sorted(REQUIRED_DOMAINS.keys() - step_names)
        extra = sorted(step_names - REQUIRED_DOMAINS.keys())
        raise ValueError(
            f"build step closure differs: missing={missing}, extra={extra}"
        )
    if product_names != REQUIRED_CONSUMERS.keys():
        missing = sorted(REQUIRED_CONSUMERS.keys() - product_names)
        extra = sorted(product_names - REQUIRED_CONSUMERS.keys())
        raise ValueError(
            f"build plan product closure differs: missing={missing}, extra={extra}"
        )
    if len(cargo_executables) != 1 or len(rust_toolchains) != 1:
        raise ValueError("compilation domains differ in Cargo or Rust toolchain identity")


def _admit_outputs(step: BuildStep) -> list[str]:
    outputs = [Path(product.output) for product in step.products]
    missing = [str(path) for path in outputs if not path.is_file() or path.is_symlink()]
    if missing:
        raise RuntimeError(
            f"{step.name} did not produce regular files: {', '.join(missing)}"
        )
    return [str(path) for path in outputs]


def execute(build_plan: Sequence[BuildStep], receipt: Path) -> None:
    validate(build_plan)
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(f"build receipt already exists: {receipt}")
    records: list[dict[str, object]] = []
    domain_counts: dict[str, int] = {}
    for step in build_plan:
        started = time.monotonic_ns()
        environment = os.environ.copy()
        environment.update(step.environment)
        subprocess.run(step.command, check=True, env=environment)
        outputs = _admit_outputs(step)
        elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
        domain_counts[step.domain] = domain_counts.get(step.domain, 0) + 1
        records.append(
            {
                "name": step.name,
                "domain": step.domain,
                "invocations": 1,
                "elapsed_ms": elapsed_ms,
                "products": [
                    {**asdict(product), "output": output, "build_count": 1}
                    for product, output in zip(step.products, outputs, strict=True)
                ],
            }
        )
        print(f"site build: {step.name} {elapsed_ms} ms", flush=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema": RECEIPT_SCHEMA,
                "steps": records,
                "invocations_by_domain": domain_counts,
                "total_invocations": len(records),
                "total_products": sum(len(step.products) for step in build_plan),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--rust-toolchain", required=True)
    arguments = parser.parse_args()
    scratch = arguments.scratch_root
    receipt = arguments.receipt
    cargo = arguments.cargo
    rust_toolchain = arguments.rust_toolchain
    if (
        not scratch.is_absolute()
        or scratch.resolve() != scratch
        or scratch.is_symlink()
        or not scratch.is_dir()
        or receipt.parent != scratch
        or receipt.is_symlink()
    ):
        parser.error("receipt must be a direct child of a real scratch root")
    if (
        not cargo.is_absolute()
        or not cargo.is_file()
        or not os.access(cargo, os.X_OK)
        or re.fullmatch(r"[A-Za-z0-9._-]+", rust_toolchain) is None
    ):
        parser.error("cargo must be an absolute executable and toolchain an installed alias")
    execute(
        plan(scratch, cargo=str(cargo), rust_toolchain=rust_toolchain), receipt
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
