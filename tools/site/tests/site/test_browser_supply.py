from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fixture import REPOSITORY, SOURCE_SHA, checker, historical_git, pinned_node_path


PLAYWRIGHT_VERSION = "1.64.1"
CHROMIUM_REVISION = "1248"
CHROMIUM_VERSION = "156.0.8078.4"
CACHE_NAME = f"eqiora-pw-{PLAYWRIGHT_VERSION}-r{CHROMIUM_REVISION}"
EXECUTABLE_SUFFIX = Path(f"chromium-{CHROMIUM_REVISION}/chrome-linux64/chrome")
FULL_VERSION_STDOUT = b"Google Chrome for Testing 156.0.8078.4 \n"
HEADLESS_VERSION_STDOUT = b"Google Chrome for Testing 156.0.8078.4\n"
SCRATCH_ROOT = Path.home() / ".cache/eqiora/site-oracle-tests"
BASIS_SHA = "19968da984c16e718baeb9faa5aae04260896c29"
BASIS_PACKAGE_LOCK_SHA256 = (
    "4c64051270f4e00cfea70f8bd90d60e8703722c868c96b9e558503b2a049b2e4"
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


class BrowserSupplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compile_version_executable(path: Path, stdout: bytes) -> None:
        source = path.parent / "version.c"
        payload = "".join(f"\\x{byte:02x}" for byte in stdout)
        _write(
            source,
            "#include <fcntl.h>\n"
            "#include <stdlib.h>\n"
            "#include <unistd.h>\n"
            "int main(void) {\n"
            '  const char *marker = getenv("EQIORA_TEST_BROWSER_MARKER");\n'
            '  if (marker) { int fd = open(marker, O_WRONLY | O_CREAT | O_APPEND, 0600); if (fd >= 0) { write(fd, "x", 1); close(fd); } }\n'
            f'  static const char output[] = "{payload}";\n'
            "  return write(1, output, sizeof(output) - 1) < 0;\n"
            "}\n",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cc", "-O2", "-o", str(path), str(source)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        source.unlink()

    def _layout(self, root: Path) -> tuple[Path, dict[str, str], Path]:
        scratch = root / "scratch"
        source = scratch / "source"
        (scratch / "build").mkdir(parents=True)
        (source / "docs/site/node_modules").mkdir(parents=True)
        (source / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        package = {
            "engines": {"node": "24.18.1", "npm": "11.16.0"},
            "devDependencies": {"@playwright/test": PLAYWRIGHT_VERSION},
        }
        lock = {
            "packages": {
                "": {
                    "devDependencies": package["devDependencies"],
                    "engines": package["engines"],
                },
                "node_modules/@playwright/test": {"version": PLAYWRIGHT_VERSION},
                "node_modules/playwright": {"version": PLAYWRIGHT_VERSION},
                "node_modules/playwright-core": {"version": PLAYWRIGHT_VERSION},
            }
        }
        _write(source / "docs/site/package.json", json.dumps(package))
        _write(source / "docs/site/package-lock.json", json.dumps(lock))

        modules = source / "docs/site/node_modules"
        browsers = {
            "browsers": [
                {
                    "name": "chromium",
                    "revision": CHROMIUM_REVISION,
                    "installByDefault": True,
                    "browserVersion": CHROMIUM_VERSION,
                },
                {
                    "name": "chromium-headless-shell",
                    "revision": CHROMIUM_REVISION,
                    "installByDefault": True,
                    "browserVersion": CHROMIUM_VERSION,
                },
            ]
        }
        _write(modules / "playwright-core/package.json", '{"version":"1.64.1"}')
        _write(modules / "playwright-core/browsers.json", json.dumps(browsers))
        loader = (
            "const path = require('path');\n"
            "exports.chromium = { executablePath: () =>\n"
            "  process.env.EQIORA_TEST_BROWSER_PATH ||\n"
            "  path.join(process.env.PLAYWRIGHT_BROWSERS_PATH,\n"
            f"    '{EXECUTABLE_SUFFIX.as_posix()}') }};\n"
        )
        for module in ("playwright-core", "playwright", "@playwright/test"):
            _write(modules / module / "index.js", loader)
            _write(
                modules / module / "package.json",
                json.dumps({"name": module, "version": PLAYWRIGHT_VERSION}),
            )

        cache = root / "browser-supply" / CACHE_NAME
        executable = cache / EXECUTABLE_SUFFIX
        self._compile_version_executable(executable, FULL_VERSION_STDOUT)
        runner = source / "tools/site/run_offline_site_checks.sh"
        runner.parent.mkdir(parents=True)
        shutil.copy2(REPOSITORY / "tools/site/run_offline_site_checks.sh", runner)
        runner.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "LC_ALL": "C",
                "TZ": "UTC",
                "PATH": pinned_node_path(root),
                "EQIORA_API_SCRATCH": str(scratch.resolve()),
                "EQIORA_SITE_SOURCE_ROOT": str(source.resolve()),
                "EQIORA_SITE_ASTRO_OUT_DIR": str((scratch / "astro").resolve()),
                "EQIORA_SITE_CARGO_TARGET": str(
                    (scratch / "cargo-target").resolve()
                ),
                "EQIORA_SITE_RUSTDOC_TARGET": str(
                    (scratch / "rustdoc-target").resolve()
                ),
                "EQIORA_SITE_RUSTDOC_STAGE": str((scratch / "rustdoc-stage").resolve()),
                "EQIORA_SITE_ARTIFACT": str((scratch / "build/site").resolve()),
                "EQIORA_SITE_SOURCE_SHA": SOURCE_SHA,
                "PLAYWRIGHT_BROWSERS_PATH": str(cache.resolve()),
            }
        )
        return runner, environment, executable

    @staticmethod
    def _check_with_execution_spy(
        site_root: Path,
        browser_cache: Path,
        executable: Path,
        expected_sha256: str,
        expected_bytes: int,
        environment: dict[str, str],
    ) -> tuple[list[str], list[tuple[str, str, int, tuple[str, ...]]]]:
        real_run = subprocess.run
        attempted: list[tuple[str, str, int, tuple[str, ...]]] = []

        def observe(command, *args, **kwargs):
            argv = tuple(os.fspath(item) for item in command)
            resolver = len(argv) == 3 and argv[:2] == ("node", "-e")
            resolver = resolver and "chromium.executablePath()" in argv[2]
            if argv and not resolver:
                program = Path(argv[0]).resolve()
                attempted.append(
                    (
                        str(program),
                        checker.sha256(program),
                        program.stat().st_size,
                        argv,
                    )
                )
            return real_run(command, *args, **kwargs)

        with mock.patch.object(checker.subprocess, "run", side_effect=observe):
            errors = checker.check_browser_supply(
                site_root,
                browser_cache,
                expected_sha256,
                expected_bytes,
                environment,
            )
        return errors, attempted

    def test_00_b01_full_browser_positive_precedes_identity_mutants(self) -> None:
        lock_bytes = historical_git("show", f"{BASIS_SHA}:docs/site/package-lock.json")
        self.assertEqual(len(lock_bytes), 254_297)
        self.assertEqual(
            hashlib.sha256(lock_bytes).hexdigest(),
            BASIS_PACKAGE_LOCK_SHA256,
        )
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary:
            _, environment, executable = self._layout(Path(temporary))
            marker = Path(temporary) / "browser-executed"
            environment["EQIORA_TEST_BROWSER_MARKER"] = str(marker)
            self.assertTrue(executable.is_file())
            self.assertFalse(executable.is_symlink())
            self.assertTrue(os.access(executable, os.X_OK))
            self.assertEqual(executable.read_bytes()[:4], b"\x7fELF")
            site_root = Path(environment["EQIORA_SITE_SOURCE_ROOT"]) / "docs/site"
            browser_cache = Path(environment["PLAYWRIGHT_BROWSERS_PATH"])
            errors, attempted = self._check_with_execution_spy(
                site_root,
                browser_cache,
                executable,
                checker.sha256(executable),
                executable.stat().st_size,
                environment,
            )
            self.assertEqual(errors, [], "\n".join(errors))
            self.assertEqual(
                attempted,
                [
                    (
                        str(executable.resolve()),
                        checker.sha256(executable),
                        executable.stat().st_size,
                        (str(executable.resolve()), "--version"),
                    )
                ],
            )
            self.assertEqual(marker.read_bytes(), b"x")

        def missing(path: Path, environment: dict[str, str]) -> None:
            path.unlink()

        def wrong_revision(path: Path, environment: dict[str, str]) -> None:
            replacement = path.with_name("chrome-version-probe")
            shutil.copy2(path, replacement)
            environment["EQIORA_TEST_BROWSER_PATH"] = str(replacement)

        def wrong_browser_version(path: Path, environment: dict[str, str]) -> None:
            source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
            browsers_path = (
                source / "docs/site/node_modules/playwright-core/browsers.json"
            )
            browsers = json.loads(browsers_path.read_text(encoding="utf-8"))
            browsers["browsers"][0]["browserVersion"] = "151.0.7922.35"
            browsers_path.write_text(json.dumps(browsers), encoding="utf-8")

        def wrong_playwright_version(path: Path, environment: dict[str, str]) -> None:
            source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
            lock_path = source / "docs/site/package-lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["packages"]["node_modules/playwright-core"]["version"] = "1.62.2"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")

        def wrong_installed_version(path: Path, environment: dict[str, str]) -> None:
            source = Path(environment["EQIORA_SITE_SOURCE_ROOT"])
            package_path = source / "docs/site/node_modules/playwright/package.json"
            package = json.loads(package_path.read_text(encoding="utf-8"))
            package["version"] = "1.62.2"
            package_path.write_text(json.dumps(package), encoding="utf-8")

        def wrong_cache(path: Path, environment: dict[str, str]) -> None:
            cache = path.parents[2]
            replacement = cache.parent / "other-cache"
            cache.rename(replacement)
            environment["PLAYWRIGHT_BROWSERS_PATH"] = str(replacement)

        def escaped_path(path: Path, environment: dict[str, str]) -> None:
            escaped = path.parents[4] / "system-chrome"
            path.rename(escaped)
            environment["EQIORA_TEST_BROWSER_PATH"] = str(escaped)

        def symlink(path: Path, environment: dict[str, str]) -> None:
            original = path.with_name("chrome-real")
            path.rename(original)
            path.symlink_to(original.name)

        def directory(path: Path, environment: dict[str, str]) -> None:
            path.unlink()
            path.mkdir()

        def fifo(path: Path, environment: dict[str, str]) -> None:
            path.unlink()
            os.mkfifo(path)

        def non_executable(path: Path, environment: dict[str, str]) -> None:
            path.chmod(0o644)

        def identity_disagreement(path: Path, environment: dict[str, str]) -> None:
            pass

        def wrong_byte_count(path: Path, environment: dict[str, str]) -> None:
            pass

        def wrong_label(path: Path, environment: dict[str, str]) -> None:
            self._compile_version_executable(path, b"Chromium 156.0.8078.4\n")

        def headless_copy(path: Path, environment: dict[str, str]) -> None:
            self._compile_version_executable(path, HEADLESS_VERSION_STDOUT)

        def shell_shim(path: Path, environment: dict[str, str]) -> None:
            path.write_bytes(
                b'#!/bin/sh\n: > "$EQIORA_TEST_BROWSER_MARKER"\n'
                b"printf 'Google Chrome for Testing 156.0.8078.4 \\n'\n"
            )
            path.chmod(0o755)

        for label, mutate, expected_error in (
            ("missing full executable", missing, "executable is unavailable"),
            ("wrong revision", wrong_revision, "did not resolve the exact full"),
            (
                "wrong browser version metadata",
                wrong_browser_version,
                "metadata must select Chromium",
            ),
            (
                "wrong Playwright lock version",
                wrong_playwright_version,
                "locked playwright-core 1.64.1",
            ),
            (
                "wrong installed Playwright version",
                wrong_installed_version,
                "installed playwright 1.64.1",
            ),
            ("wrong cache suffix", wrong_cache, "cache must use exact"),
            (
                "path outside named cache",
                escaped_path,
                "did not resolve the exact full",
            ),
            ("symlink executable", symlink, "regular non-symlink"),
            ("directory executable", directory, "regular non-symlink"),
            ("FIFO executable", fifo, "regular non-symlink"),
            ("non-executable file", non_executable, "is not executable"),
            (
                "online/offline identity disagreement",
                identity_disagreement,
                "SHA-256 changed after online verification",
            ),
            (
                "wrong expected byte count",
                wrong_byte_count,
                "byte length changed after online verification",
            ),
            ("wrong exact version", wrong_label, "exact 41-byte locked version"),
            (
                "headless-shell substitution",
                headless_copy,
                "exact 41-byte locked version",
            ),
            ("shell shim", shell_shim, "acquired binary, not a shim"),
        ):
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as temporary,
            ):
                _, environment, executable = self._layout(Path(temporary))
                expected_sha256 = checker.sha256(executable)
                expected_bytes = executable.stat().st_size
                marker = Path(temporary) / "browser-executed"
                environment["EQIORA_TEST_BROWSER_MARKER"] = str(marker)
                mutate(executable, environment)
                if label == "online/offline identity disagreement":
                    expected_sha256 = "1" * 64
                elif label == "wrong expected byte count":
                    expected_bytes += 1
                elif label in {
                    "wrong exact version",
                    "headless-shell substitution",
                    "shell shim",
                }:
                    expected_sha256 = checker.sha256(executable)
                    expected_bytes = executable.stat().st_size
                site_root = Path(environment["EQIORA_SITE_SOURCE_ROOT"]) / "docs/site"
                browser_cache = Path(environment["PLAYWRIGHT_BROWSERS_PATH"])
                errors, attempted = self._check_with_execution_spy(
                    site_root,
                    browser_cache,
                    executable,
                    expected_sha256,
                    expected_bytes,
                    environment,
                )
                self.assertTrue(
                    any(expected_error in error for error in errors),
                    f"B-01 mutant missed its causal gate: {label}: {errors}",
                )
                if label in {
                    "wrong revision",
                    "online/offline identity disagreement",
                    "wrong expected byte count",
                    "non-executable file",
                    "shell shim",
                }:
                    self.assertEqual((attempted, marker.exists()), ([], False), label)

        runner_text = (REPOSITORY / "tools/site/run_offline_site_checks.sh").read_text(
            encoding="utf-8"
        )
        errors = checker.check_runner_browser_supply_text(runner_text)
        self.assertEqual(errors, [], "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
