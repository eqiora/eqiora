"""Mocked host-capacity checks; no host SDKs are measured or removed."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_rust_disk as disk

HOSTED = {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_OS": "Linux"}


class PrepareRustDiskTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, HOSTED, clear=True))
        self.usage = self.stack.enter_context(mock.patch.object(disk.shutil, "disk_usage"))
        self.size = self.stack.enter_context(mock.patch.object(disk, "sdk_allocated_bytes", side_effect=[1024, 2048]))
        self.remove = self.stack.enter_context(mock.patch.object(disk.subprocess, "run"))
        self.output = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.usage.side_effect = [SimpleNamespace(free=disk.MIN_FREE_BYTES)] * 2

    def test_every_guard_rejects_before_measurement_or_deletion(self):
        for name, wrong in (("GITHUB_ACTIONS", "false"), ("RUNNER_ENVIRONMENT", "self-hosted"), ("RUNNER_OS", "Windows")):
            for value in (wrong, None):
                with self.subTest(name=name, value=value), mock.patch.dict(os.environ, HOSTED, clear=True):
                    if value is None:
                        os.environ.pop(name)
                    else:
                        os.environ[name] = value
                    with self.assertRaisesRegex(RuntimeError, "refusing to measure or change"):
                        disk.prepare()
        self.usage.assert_not_called()
        self.size.assert_not_called()
        self.remove.assert_not_called()

    def test_sufficient_space_preserves_both_sdks(self):
        disk.prepare()
        self.remove.assert_not_called()
        self.assertIn(f"free_bytes_before={disk.MIN_FREE_BYTES}", self.output.getvalue())
        self.assertIn("sdk=/usr/local/lib/android present=True allocated_bytes=1024", self.output.getvalue())
        self.assertIn(f"free_bytes_after={disk.MIN_FREE_BYTES}", self.output.getvalue())

    def test_low_space_removes_only_present_literal_allowlist(self):
        self.usage.side_effect = [SimpleNamespace(free=19_574_378_496), SimpleNamespace(free=disk.MIN_FREE_BYTES)]
        disk.prepare()
        self.assertEqual(self.remove.call_args_list, [
            mock.call(["sudo", "-n", "rm", "-rf", "--", "/usr/local/lib/android"], check=True),
            mock.call(["sudo", "-n", "rm", "-rf", "--", "/usr/share/dotnet"], check=True),
        ])

    def test_absent_sdks_are_safe_and_insufficient_postcheck_is_explicit(self):
        self.size.side_effect = [None, None]
        self.usage.side_effect = [SimpleNamespace(free=1), SimpleNamespace(free=2)]
        with self.assertRaisesRegex(RuntimeError, f"available_bytes=2 required_bytes={disk.MIN_FREE_BYTES}"):
            disk.prepare()
        self.remove.assert_not_called()
        self.assertIn("present=False allocated_bytes=None", self.output.getvalue())

    def test_one_absent_sdk_is_not_removed(self):
        self.size.side_effect = [None, 2048]
        self.usage.side_effect = [SimpleNamespace(free=1), SimpleNamespace(free=disk.MIN_FREE_BYTES)]
        disk.prepare()
        self.remove.assert_called_once_with(["sudo", "-n", "rm", "-rf", "--", "/usr/share/dotnet"], check=True)

    def test_removal_failure_propagates_and_stops_further_work(self):
        self.usage.side_effect = [SimpleNamespace(free=1)]
        self.remove.side_effect = subprocess.CalledProcessError(1, ["sudo", "-n", "rm"])
        with self.assertRaises(subprocess.CalledProcessError):
            disk.prepare()
        self.assertEqual(self.remove.call_count, 1)
        self.assertEqual(self.usage.call_count, 1)

    def test_cli_reports_failure_as_nonzero(self):
        with mock.patch.object(disk, "prepare", side_effect=RuntimeError("capacity failure")), contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(disk.main(), 1)
        self.assertIn("rust-disk-preflight: capacity failure", stderr.getvalue())

    def test_workflow_preflight_follows_cache_and_precedes_clippy(self):
        workflow = (Path(__file__).resolve().parents[3] / ".github/workflows/ci.yml").read_text()
        quality = workflow.split("  quality:\n", 1)[1].split("\n  msrv:", 1)[0]
        marker = "      - name: Prepare Stable Rust disk capacity\n"
        self.assertEqual(workflow.count("python3 tools/ci/prepare_rust_disk.py"), 1)
        step = quality.split(marker, 1)[1].split("      - ", 1)[0]
        self.assertIn("if: needs.changes.outputs.rust == 'true'", step)
        self.assertLess(quality.index("- name: Restore Stable dependencies"), quality.index(marker))
        self.assertLess(quality.index(marker), quality.index("- name: Clippy\n"))


if __name__ == "__main__":
    unittest.main()
