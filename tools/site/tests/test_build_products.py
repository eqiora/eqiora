from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tools.site import build_products

CARGO = "/cargo/bin/cargo"
RUST_TOOLCHAIN = "1.97.1-x86_64-unknown-linux-gnu"


def plan(scratch: Path) -> tuple[build_products.BuildStep, ...]:
    return build_products.plan(
        scratch, cargo=CARGO, rust_toolchain=RUST_TOOLCHAIN
    )


class BuildProductsTests(unittest.TestCase):
    def test_executable_build_enables_the_cli_feature(self) -> None:
        steps = plan(Path("/build-scratch"))
        command = steps[0].command
        self.assertEqual(command[command.index("--features") + 1], "eqiora/cli")
        self.assertNotIn("eqiora-verify", command)
        self.assertNotIn("--bins", command)
        self.assertEqual(
            [command[i + 1] for i, flag in enumerate(command) if flag == "--bin"],
            ["eqiora", "xtask"],
        )

    def test_plan_rejects_duplicate_product(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            steps = list(plan(Path(temporary)))
            steps[1] = replace(steps[1], products=(steps[0].products[0],))
            with self.assertRaisesRegex(ValueError, "duplicate or empty product"):
                build_products.validate(steps)

    def test_plan_rejects_product_without_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            steps = list(plan(Path(temporary)))
            product = replace(steps[0].products[0], consumer="")
            steps[0] = replace(steps[0], products=(product, *steps[0].products[1:]))
            with self.assertRaisesRegex(ValueError, "wrong consumer"):
                build_products.validate(steps)

    def test_plan_rejects_cargo_profile_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            steps = list(plan(Path(temporary)))
            command = tuple("release" if arg == "dev" else arg for arg in steps[0].command)
            steps[0] = replace(steps[0], command=command)
            with self.assertRaisesRegex(ValueError, "development profile"):
                build_products.validate(steps)

            steps = list(plan(Path(temporary)))
            steps[1] = replace(
                steps[1],
                command=(
                    *steps[1].command[:3],
                    "--release",
                    *steps[1].command[3:],
                ),
            )
            with self.assertRaisesRegex(ValueError, "documentation profile"):
                build_products.validate(steps)

    def test_plan_rejects_feature_and_domain_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            steps = list(plan(Path(temporary)))
            steps[1] = replace(steps[1], domain="cargo-dev")
            with self.assertRaisesRegex(ValueError, "wrong compilation domain"):
                build_products.validate(steps)

            steps = list(plan(Path(temporary)))
            command = tuple(
                argument
                for argument in steps[1].command
                if argument != "--all-features"
            )
            steps[1] = replace(steps[1], command=command)
            with self.assertRaisesRegex(ValueError, "all-features closure"):
                build_products.validate(steps)

    def test_plan_rejects_toolchain_boundary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            steps = list(plan(Path(temporary)))
            steps[1] = replace(
                steps[1],
                environment=(("RUSTUP_TOOLCHAIN", "stable"),),
            )
            with self.assertRaisesRegex(ValueError, "toolchain identity"):
                build_products.validate(steps)

            steps = list(plan(Path(temporary)))
            steps[1] = replace(
                steps[1], command=("/other/cargo", *steps[1].command[1:])
            )
            with self.assertRaisesRegex(ValueError, "Cargo or Rust"):
                build_products.validate(steps)

    def test_execution_builds_each_product_once_in_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            receipt = scratch / "build-products.json"
            invocations: list[tuple[tuple[str, ...], str | None, str | None]] = []

            def run(command, *, check, env):  # type: ignore[no-untyped-def]
                self.assertIs(check, True)
                command = tuple(command)
                invocations.append(
                    (
                        command,
                        env.get("CARGO_TARGET_DIR"),
                        env.get("RUSTUP_TOOLCHAIN"),
                    )
                )
                target = scratch / "cargo-target"
                if command[:2] == (CARGO, "build"):
                    development = target / "debug"
                    development.mkdir(parents=True)
                    for name in ("eqiora", "xtask"):
                        (development / name).write_text(name, encoding="utf-8")
                    self.assertEqual(env["CARGO_PROFILE_DEV_DEBUG"], "0")
                    self.assertEqual(env["CARGO_PROFILE_DEV_INCREMENTAL"], "false")
                    self.assertEqual(env["CARGO_PROFILE_DEV_OPT_LEVEL"], "0")
                elif command[:2] == (CARGO, "doc"):
                    rustdoc = scratch / "rustdoc-target/doc/eqiora"
                    rustdoc.mkdir(parents=True)
                    (rustdoc / "index.html").write_text("rustdoc", encoding="utf-8")
                else:
                    self.fail(f"unexpected build command: {command!r}")

            with patch.object(build_products.subprocess, "run", side_effect=run):
                build_products.execute(plan(scratch), receipt)

            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], build_products.RECEIPT_SCHEMA)
            self.assertEqual(payload["total_invocations"], 2)
            self.assertEqual(payload["total_products"], 3)
            self.assertEqual(
                payload["invocations_by_domain"],
                {
                    "cargo-dev": 1,
                    "cargo-doc/rustdoc-all-features": 1,
                },
            )
            products = [
                product for step in payload["steps"] for product in step["products"]
            ]
            self.assertEqual(len({product["name"] for product in products}), 3)
            self.assertTrue(all(product["build_count"] == 1 for product in products))
            self.assertNotIn("cache_hit", payload)
            for record in [payload, *payload["steps"], *products]:
                self.assertFalse(any("cache" in key for key in record))
            self.assertTrue(
                all(toolchain == RUST_TOOLCHAIN for _, _, toolchain in invocations)
            )
            command = invocations[0][0]
            self.assertNotIn("--release", command)
            self.assertEqual(command[command.index("--profile") + 1], "dev")
            self.assertEqual(
                command[command.index("--target-dir") + 1],
                str(scratch / "cargo-target"),
            )
            self.assertNotIn("--release", invocations[1][0])
            self.assertEqual(
                invocations[1][0][invocations[1][0].index("--target-dir") + 1],
                str(scratch / "rustdoc-target"),
            )

    def test_missing_product_prevents_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            receipt = scratch / "build-products.json"
            with patch.object(build_products.subprocess, "run"):
                with self.assertRaisesRegex(RuntimeError, "did not produce"):
                    build_products.execute(plan(scratch), receipt)
            self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
