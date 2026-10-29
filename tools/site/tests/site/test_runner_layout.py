from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path

from fixture import (
    REPOSITORY,
    _write_rustc_preflight_double,
    checker,
    copy_editor_check_inputs,
    git_object_authority,
    historical_git,
    pinned_node_path,
)

SCRATCH_ROOT = Path.home() / ".cache/eqiora/site-oracle-tests"
BASIS_SHA = "19968da984c16e718baeb9faa5aae04260896c29"
BASIS_TREE = "1d19473c487b8035608cc88cbd99757f2b95865a"
AGENTS_SHA256 = "ffc9b0381a01c16b3d72389ef777842215c48b65d6eda6881f5e75bfa5d531c0"
BROWSER_SHA256 = "377a9fc07b495c3b8d0ab8ec1fdfd153641759f9756fea34d8dfce4de82f5025"
BROWSER_BYTES = 301_186_376
SOURCE_SUCCESS = "site source: exact optional CLAUDE.md topology admitted"
BROWSER_SUCCESS = "site browser: exact locked full Chromium supply admitted"
POST_SB_SENTINEL = 85
TOOLCHAIN_BYTES = 66
TOOLCHAIN_BLOB = "73cb934de4706a914c15e8db2a3c037ce75699d9"
TOOLCHAIN_SHA256 = "a6a0bbd29ffaa8182dc22d1d9149709f1091e47df40ed96eb8a78a711c66a4ce"
MISMATCH_TOOLCHAIN = b'[toolchain]\nchannel = "1.85.0"\n'
CHECKER_MODULES = (
    "build_products.py",
    "check_site.py",
    "check_site_artifact.py",
    "check_site_html.py",
    "check_site_references.py",
    "check_site_rustdoc.py",
    "check_site_sitemap.py",
    "check_site_starlight.py",
    "check_site_starlight_content.py",
    "check_site_supply.py",
)


class ToolchainSelectorTests(unittest.TestCase):
    def test_mise_supplies_stable_and_the_exact_docs_release(self) -> None:
        toolchain = tomllib.loads(
            (REPOSITORY / "rust-toolchain.toml").read_text(encoding="utf-8")
        )
        mise = tomllib.loads(
            (REPOSITORY / "mise.toml").read_text(encoding="utf-8")
        )
        lock = tomllib.loads(
            (REPOSITORY / "mise.lock").read_text(encoding="utf-8")
        )
        self.assertIs(mise["settings"]["locked"], True)
        self.assertEqual(toolchain["toolchain"]["channel"], "stable")
        self.assertEqual(
            [entry["version"] for entry in mise["tools"]["rust"]],
            ["stable", "1.97.1-x86_64-unknown-linux-gnu"],
        )
        self.assertEqual(mise["tools"]["rust"][1]["os"], ["linux/x64"])
        self.assertEqual(
            lock["tools"]["rust"],
            [
                {
                    "version": "1.97.1-x86_64-unknown-linux-gnu",
                    "backend": "core:rust",
                },
                {"version": "stable", "backend": "core:rust"},
            ],
        )


class OfflineRunnerLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

        browser_root_value = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        browser_sha256 = os.environ.get("EQIORA_SITE_BROWSER_SHA256")
        browser_bytes = os.environ.get("EQIORA_SITE_BROWSER_BYTES")
        if not browser_root_value or not browser_sha256 or not browser_bytes:
            raise AssertionError("the official browser identity inputs are required")
        cls.browser_root = Path(browser_root_value)
        cls.browser = cls.browser_root / "chromium-1248/chrome-linux64/chrome"
        if (
            not cls.browser_root.is_absolute()
            or cls.browser_root.resolve() != cls.browser_root
            or cls.browser_root.name != "eqiora-pw-1.64.1-r1248"
            or cls.browser.is_symlink()
            or not cls.browser.is_file()
            or not os.access(cls.browser, os.X_OK)
            or cls.browser.stat().st_size != BROWSER_BYTES
            or hashlib.sha256(cls.browser.read_bytes()).hexdigest() != BROWSER_SHA256
            or browser_sha256 != BROWSER_SHA256
            or browser_bytes != str(BROWSER_BYTES)
        ):
            raise AssertionError("the official full Chromium supply changed")
        version = subprocess.run(
            [str(cls.browser), "--version"],
            check=False,
            env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        if (
            version.returncode != 0
            or version.stderr
            or version.stdout != checker.FULL_CHROMIUM_VERSION_STDOUT
        ):
            raise AssertionError("the official full Chromium version bytes changed")

        site = REPOSITORY / "docs/site"
        for relative in (
            "node_modules/@playwright/test/package.json",
            "node_modules/playwright/package.json",
            "node_modules/playwright-core/package.json",
            "node_modules/playwright-core/browsers.json",
        ):
            if not (site / relative).is_file():
                raise AssertionError(
                    "the exact locked Playwright packages must be installed first"
                )

    @staticmethod
    def _copy_locked_browser_supply(source: Path) -> None:
        site = source / "docs/site"
        site.mkdir(parents=True)
        shutil.copy2(REPOSITORY / "docs/site/package.json", site / "package.json")
        shutil.copy2(
            REPOSITORY / "docs/site/package-lock.json", site / "package-lock.json"
        )
        modules = site / "node_modules"
        modules.mkdir()

        def link_or_copy(origin: str, destination: str) -> str:
            try:
                os.link(origin, destination)
            except OSError:
                shutil.copy2(origin, destination)
            return destination

        for package in ("@playwright/test", "playwright", "playwright-core"):
            destination = modules / package
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                REPOSITORY / "docs/site/node_modules" / package,
                destination,
                copy_function=link_or_copy,
            )

    def _environment(self, root: Path, scratch: Path) -> dict[str, str]:
        scratch = scratch.resolve()
        authority = git_object_authority()
        environment = os.environ.copy()
        environment.pop("MISE_TRUSTED_CONFIG_PATHS", None)
        environment.pop("RUSTUP_TOOLCHAIN", None)
        environment.update(
            {
                "LC_ALL": "C",
                "TZ": "UTC",
                "npm_config_offline": "true",
                "CARGO_NET_OFFLINE": "true",
                "HOME": str(root / "home"),
                "PATH": f"{root / 'fixture-bin'}{os.pathsep}{pinned_node_path(root)}",
                "EQIORA_API_SCRATCH": str(scratch),
                "EQIORA_SITE_SOURCE_ROOT": str(scratch / "source"),
                "EQIORA_SITE_ASTRO_OUT_DIR": str(scratch / "astro"),
                "EQIORA_SITE_CARGO_TARGET": str(scratch / "cargo-target"),
                "EQIORA_SITE_RUSTDOC_TARGET": str(scratch / "rustdoc-target"),
                "EQIORA_SITE_RUSTDOC_STAGE": str(scratch / "rustdoc-stage"),
                "EQIORA_SITE_ARTIFACT": str(scratch / "build/site"),
                "EQIORA_SITE_SOURCE_SHA": authority.head,
                "EQIORA_SITE_GIT_OBJECT_REPOSITORY": str(authority.root),
                "PLAYWRIGHT_BROWSERS_PATH": str(self.browser_root),
                "EQIORA_SITE_BROWSER_SHA256": BROWSER_SHA256,
                "EQIORA_SITE_BROWSER_BYTES": str(BROWSER_BYTES),
                "TRACE_FILE": str(root / "trace.log"),
            }
        )
        return environment

    def _copy_exact_toolchain(self, source: Path) -> None:
        origin = REPOSITORY / "rust-toolchain.toml"
        self.assertTrue(origin.is_file())
        self.assertFalse(origin.is_symlink())
        payload = origin.read_bytes()
        self.assertEqual(len(payload), TOOLCHAIN_BYTES)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), TOOLCHAIN_SHA256)
        self.assertEqual(
            hashlib.sha1(b"blob 66\0" + payload, usedforsecurity=False).hexdigest(),
            TOOLCHAIN_BLOB,
        )
        destination = source / "rust-toolchain.toml"
        shutil.copyfile(origin, destination)
        destination.chmod(0o644)
        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o644)

    def _layout(
        self, root: Path, *, admitted_link: bool = True
    ) -> tuple[Path, dict[str, str]]:
        scratch = root / "scratch"
        source = scratch / "source"
        (scratch / "build").mkdir(parents=True)
        self._copy_locked_browser_supply(source)
        (source / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        copy_editor_check_inputs(source)
        self._copy_exact_toolchain(source)
        shutil.copy2(REPOSITORY / "AGENTS.md", source / "AGENTS.md")
        if admitted_link:
            (source / "CLAUDE.md").symlink_to("AGENTS.md")
        runner = source / "tools/site/run_offline_site_checks.sh"
        runner.parent.mkdir(parents=True)
        for module in CHECKER_MODULES:
            shutil.copy2(
                REPOSITORY / "tools/site" / module,
                source / "tools/site" / module,
            )
        shutil.copy2(REPOSITORY / "tools/site/run_offline_site_checks.sh", runner)
        runner.chmod(0o755)
        fixture_bin = root / "fixture-bin"
        fixture_bin.mkdir()
        _write_rustc_preflight_double(fixture_bin)
        rustup_proxy_bin = root / "home/.cargo/bin"
        rustup_proxy_bin.mkdir(parents=True)
        shutil.copy2(fixture_bin / "rustc", rustup_proxy_bin / "rustc")
        shutil.copy2(fixture_bin / "rustup", rustup_proxy_bin / "rustup")
        sentinel = fixture_bin / "dpkg-query"
        sentinel.write_text(
            "#!/bin/sh\nprintf 'post-sb-preflight\\n' >> \"$TRACE_FILE\"\nexit 85\n",
            encoding="utf-8",
        )
        sentinel.chmod(0o755)
        return runner, self._environment(root, scratch)

    def _archive_layout(self, root: Path) -> tuple[Path, dict[str, str]]:
        scratch = root / "scratch"
        source = scratch / "source"
        (scratch / "build").mkdir(parents=True)
        source.mkdir()
        archive = root / "source.tar"
        with archive.open("wb") as target:
            target.write(historical_git("archive", "--format=tar", BASIS_SHA))
        subprocess.run(["tar", "-xf", str(archive), "-C", str(source)], check=True)
        self.assertTrue(stat.S_ISLNK((source / "CLAUDE.md").lstat().st_mode))
        self.assertFalse((source / "AGENTS.md").is_symlink())
        self.assertTrue((source / "AGENTS.md").is_file())
        (source / "docs/site/node_modules").mkdir(exist_ok=True)
        runner = source / "tools/site/run_offline_site_checks.sh"
        return runner, self._environment(root, scratch)

    def _assert_frozen_basis_archive(self, root: Path) -> None:
        archive = root / "basis.tar"
        with archive.open("wb") as target:
            target.write(historical_git("archive", "--format=tar", BASIS_SHA))
        self.assertEqual(archive.stat().st_size, 34_088_960)
        self.assertEqual(
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            "69b560bac187303143a1a53bf82f73b8129c84a31fb1df4bfd109cee33f29252",
        )

    @staticmethod
    def _run(
        runner: Path, environment: dict[str, str], *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(runner), *arguments],
            check=False,
            cwd=environment["EQIORA_SITE_SOURCE_ROOT"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )

    def _assert_sb_positive(
        self,
        result: subprocess.CompletedProcess[str],
        environment: dict[str, str],
    ) -> None:
        trace = Path(environment["TRACE_FILE"])
        self.assertEqual(
            result.returncode,
            POST_SB_SENTINEL,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertEqual(
            trace.read_text(encoding="utf-8").splitlines(), ["post-sb-preflight"]
        )
        self.assertEqual(result.stdout.count(SOURCE_SUCCESS), 1)
        self.assertEqual(result.stdout.count(BROWSER_SUCCESS), 1)
        self.assertLess(
            result.stdout.index(SOURCE_SUCCESS), result.stdout.index(BROWSER_SUCCESS)
        )

    def _assert_toolchain_rejection(
        self,
        result: subprocess.CompletedProcess[str],
        environment: dict[str, str],
    ) -> None:
        self.assertNotIn(result.returncode, (0, POST_SB_SENTINEL))
        self.assertFalse(Path(environment["TRACE_FILE"]).exists())
        self.assertNotIn(SOURCE_SUCCESS, result.stdout)
        self.assertNotIn(BROWSER_SUCCESS, result.stdout)
        source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
        selected = subprocess.run(
            ["rustc", "-Vv"],
            check=False,
            cwd=source,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        stable = subprocess.run(
            ["rustc", "+stable", "-Vv"],
            check=False,
            cwd=source,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        self.assertEqual((selected.returncode, stable.returncode), (0, 0))
        self.assertEqual((selected.stderr, stable.stderr), (b"", b""))
        self.assertNotEqual(selected.stdout, stable.stdout)

    def test_00_s01_exact_archive_and_optional_link_pass_before_mutants(self) -> None:
        tree_identity = (
            historical_git("rev-parse", f"{BASIS_SHA}^{{tree}}").decode().strip()
        )
        self.assertEqual(tree_identity, BASIS_TREE)
        tree = historical_git("ls-tree", "-r", "-z", BASIS_SHA).split(b"\0")
        links = [entry for entry in tree if entry.startswith(b"120000 ")]
        self.assertEqual(
            links,
            [b"120000 blob 47dc3e3d863cfb5727b87d785d09abf9743c0a72\tCLAUDE.md"],
        )
        self.assertIn(
            b"100644 blob 61c1bbede492aef4a9c85fa364d031e012621809\tAGENTS.md",
            tree,
        )
        payload = historical_git("show", f"{BASIS_SHA}:CLAUDE.md")
        target = historical_git("show", f"{BASIS_SHA}:AGENTS.md")
        self.assertEqual(payload, b"AGENTS.md")
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "a54ff182c7e8acf56acfd6e4b9c3ff41e2c41a31c9b211b2deb9df75d9a478f9",
        )
        self.assertEqual(len(target), 12_408)
        self.assertEqual(
            hashlib.sha256(target).hexdigest(),
            "ffc9b0381a01c16b3d72389ef777842215c48b65d6eda6881f5e75bfa5d531c0",
        )

        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            self._assert_frozen_basis_archive(Path(temporary))

        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            _, environment = self._archive_layout(Path(temporary))
            source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
            before = (
                os.readlink(source / "CLAUDE.md"),
                hashlib.sha256((source / "AGENTS.md").read_bytes()).hexdigest(),
            )
            self.assertEqual(checker.check_source_topology(source, before[1]), [])
            self.assertEqual(
                (
                    os.readlink(source / "CLAUDE.md"),
                    hashlib.sha256((source / "AGENTS.md").read_bytes()).hexdigest(),
                ),
                before,
            )

        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            _, environment = self._layout(Path(temporary), admitted_link=False)
            source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
            self.assertEqual(checker.check_source_topology(source), [])

        def extra_link(runner: Path, environment: dict[str, str]) -> None:
            Path(environment["EQIORA_SITE_SOURCE_ROOT"], "extra-link").symlink_to(
                "AGENTS.md"
            )

        def link_target(value: str):
            def mutate(runner: Path, environment: dict[str, str]) -> None:
                link = Path(environment["EQIORA_SITE_SOURCE_ROOT"], "CLAUDE.md")
                link.unlink()
                link.symlink_to(value)

            return mutate

        def missing_target(runner: Path, environment: dict[str, str]) -> None:
            Path(environment["EQIORA_SITE_SOURCE_ROOT"], "AGENTS.md").unlink()

        def directory_target(runner: Path, environment: dict[str, str]) -> None:
            target = Path(environment["EQIORA_SITE_SOURCE_ROOT"], "AGENTS.md")
            target.unlink()
            target.mkdir()

        def fifo_target(runner: Path, environment: dict[str, str]) -> None:
            target = Path(environment["EQIORA_SITE_SOURCE_ROOT"], "AGENTS.md")
            target.unlink()
            os.mkfifo(target)

        def symlink_target(runner: Path, environment: dict[str, str]) -> None:
            source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
            (source / "AGENTS.md").rename(source / "governance")
            (source / "AGENTS.md").symlink_to("governance")

        def target_byte_drift(runner: Path, environment: dict[str, str]) -> None:
            source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
            (source / "AGENTS.md").write_bytes(b"drifted governance\n")

        for label, mutate, expected_error in (
            ("additional source symlink", extra_link, "unadmitted symlinks"),
            ("absolute CLAUDE target", link_target("/AGENTS.md"), "exact link payload"),
            ("dot CLAUDE target", link_target("./AGENTS.md"), "exact link payload"),
            ("parent CLAUDE target", link_target("../AGENTS.md"), "exact link payload"),
            (
                "nested CLAUDE target",
                link_target("docs/AGENTS.md"),
                "exact link payload",
            ),
            (
                "alternate CLAUDE spelling",
                link_target("agents.md"),
                "exact link payload",
            ),
            (
                "CLAUDE target with final LF",
                link_target("AGENTS.md\n"),
                "exact link payload",
            ),
            ("missing AGENTS target", missing_target, "target is unavailable"),
            ("directory AGENTS target", directory_target, "regular non-symlink"),
            ("FIFO AGENTS target", fifo_target, "regular non-symlink"),
            ("symlink AGENTS target", symlink_target, "unadmitted symlinks"),
            (
                "same-commit AGENTS byte drift",
                target_byte_drift,
                "differs from the same-commit Git blob",
            ),
        ):
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary,
            ):
                runner, environment = self._layout(Path(temporary))
                source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
                expected = hashlib.sha256(
                    (source / "AGENTS.md").read_bytes()
                ).hexdigest()
                mutate(runner, environment)
                errors = checker.check_source_topology(source, expected)
                self.assertTrue(
                    any(expected_error in error for error in errors),
                    f"S-01 mutant missed its causal gate: {label}: {errors}",
                )

        runner_text = (REPOSITORY / "tools/site/run_offline_site_checks.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("+1.97.1", runner_text)
        errors = checker.check_runner_source_topology_text(runner_text)
        self.assertEqual(errors, [], "\n".join(errors))

    def test_01_normal_runner_reaches_post_sb_boundary_before_mutants(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            runner, environment = self._layout(Path(temporary))
            rustup_trace = Path(temporary) / "rustup-trace.log"
            rustc_trace = Path(temporary) / "rustc-trace.log"
            environment.update(
                {
                    "FIXTURE_RUSTUP_TRACE": str(rustup_trace),
                    "FIXTURE_RUSTC_TRACE": str(rustc_trace),
                }
            )
            self._assert_sb_positive(self._run(runner, environment), environment)
            self.assertFalse(rustup_trace.exists())
            self.assertEqual(rustc_trace.read_text(encoding="utf-8"), "stable\n")

        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            runner, environment = self._layout(Path(temporary))
            rustup_trace = Path(temporary) / "rustup-trace.log"
            rustc_trace = Path(temporary) / "rustc-trace.log"
            environment.update(
                {
                    "FIXTURE_STABLE_RUST_RELEASE": "1.98.0",
                    "FIXTURE_RUSTUP_TRACE": str(rustup_trace),
                    "FIXTURE_RUSTC_TRACE": str(rustc_trace),
                }
            )
            self._assert_sb_positive(self._run(runner, environment), environment)
            self.assertEqual(
                rustup_trace.read_text(encoding="utf-8"), "toolchain list\n"
            )
            self.assertEqual(
                rustc_trace.read_text(encoding="utf-8"),
                "stable-x86_64-unknown-linux-gnu\n"
                "1.97.1-x86_64-unknown-linux-gnu\n"
                "1.97.1-x86_64-unknown-linux-gnu\n",
            )

        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            runner, environment = self._layout(Path(temporary))
            environment.update(
                {
                    "FIXTURE_STABLE_RUST_RELEASE": "1.98.0",
                    "FIXTURE_PINNED_RUST_RELEASE": "1.98.0",
                }
            )
            result = self._run(runner, environment)
            self.assertNotIn(result.returncode, (0, POST_SB_SENTINEL))
            self.assertIn(
                "no installed Rust toolchain reports release 1.97.1",
                result.stderr,
            )
            self.assertFalse(Path(environment["TRACE_FILE"]).exists())

        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            runner, environment = self._layout(Path(temporary))
            toolchain = Path(
                environment["EQIORA_SITE_SOURCE_ROOT"], "rust-toolchain.toml"
            )
            toolchain.unlink()
            self._assert_toolchain_rejection(
                self._run(runner, environment), environment
            )

        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            runner, environment = self._layout(Path(temporary))
            toolchain = Path(
                environment["EQIORA_SITE_SOURCE_ROOT"], "rust-toolchain.toml"
            )
            toolchain.write_bytes(MISMATCH_TOOLCHAIN)
            toolchain.chmod(0o644)
            self._assert_toolchain_rejection(
                self._run(runner, environment), environment
            )

    def test_02_existing_layout_mutants_fail_closed(self) -> None:
        def extra_entry(runner: Path, environment: dict[str, str]) -> None:
            Path(environment["EQIORA_API_SCRATCH"], "unexpected").touch()

        def missing_build(runner: Path, environment: dict[str, str]) -> None:
            Path(environment["EQIORA_API_SCRATCH"], "build").rmdir()

        def stale_cargo_target(runner: Path, environment: dict[str, str]) -> None:
            Path(environment["EQIORA_SITE_CARGO_TARGET"]).mkdir()

        def stale_rustdoc_target(runner: Path, environment: dict[str, str]) -> None:
            Path(environment["EQIORA_SITE_RUSTDOC_TARGET"]).mkdir()

        def linked_source(runner: Path, environment: dict[str, str]) -> None:
            source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
            backing = source.parent.parent / "source-backing"
            source.rename(backing)
            source.symlink_to(backing, target_is_directory=True)

        for label, mutate in (
            ("extra top-level entry", extra_entry),
            ("missing build directory", missing_build),
            ("stale restored Cargo target", stale_cargo_target),
            ("stale restored Rustdoc target", stale_rustdoc_target),
            ("linked source", linked_source),
        ):
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary,
            ):
                runner, environment = self._layout(Path(temporary))
                mutate(runner, environment)
                result = self._run(runner, environment)
                self.assertNotIn(result.returncode, (0, POST_SB_SENTINEL))
                trace = Path(environment["TRACE_FILE"])
                self.assertFalse(trace.exists())
                self.assertNotIn(SOURCE_SUCCESS, result.stdout)
                self.assertNotIn(BROWSER_SUCCESS, result.stdout)

    def test_03_historical_preflight_option_is_rejected_before_observation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            runner, environment = self._layout(Path(temporary))
            result = self._run(runner, environment, "--preflight-only")
            self.assertEqual(result.returncode, 2)
            self.assertFalse(Path(environment["TRACE_FILE"]).exists())
            self.assertNotIn(SOURCE_SUCCESS, result.stdout)
            self.assertNotIn(BROWSER_SUCCESS, result.stdout)


if __name__ == "__main__":
    unittest.main()
