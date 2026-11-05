"""Driver falsifiers; no native libraries, scientific observations or remote jobs."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv
import zipfile

from run_matched import CampaignError, check_installation, paired_summary, run_campaign, schedule, spread, validate_monitor


class MatchedDriverTests(unittest.TestCase):
    def test_fixed_budget_and_balanced_order(self):
        stages = schedule()
        self.assertEqual(stages[:2], [("exploratory", None, "baseline"), ("exploratory", None, "candidate")])
        self.assertEqual(len(stages), 18)
        for pair in range(1, 9):
            actual = [label for kind, number, label in stages if number == pair]
            self.assertEqual(actual, ["baseline", "candidate"] if pair % 2 else ["candidate", "baseline"])

    def test_pairs_are_units_and_exploratory_outlier_is_excluded(self):
        stages = []
        for kind, pair, label in schedule():
            # Independent hand calculation: paired candidate/baseline is 1/2,
            # despite tenfold chronological drift and extreme exploratory values.
            value = (10 ** pair) * (2 if label == "baseline" else 1) if pair else 1e30
            rows = [{"tool": tool, "cells": cells, "wall": {"median": value}, "cpu": {"median": value / 2}}
                    for tool in ("eqiora", "fenicsx") for cells in (4, 8, 16, 32)]
            stages.append({"kind": kind, "pair": pair, "label": label, "rows": rows})
        result = paired_summary(stages)
        self.assertFalse(result["stable_timing_claim"])
        self.assertEqual(len(result["rows"]), 8)
        for row in result["rows"]:
            self.assertEqual(len(row["pairs"]), 8)
            self.assertEqual(row["wall_ratio_median"], 0.5)
            self.assertEqual(row["wall_ratio_range"], [0.5, 0.5])
            self.assertEqual(row["cpu_ratio_median"], 0.5)
            self.assertEqual(row["wall_ratio_median_by_order"], {"baseline": 0.5, "candidate": 0.5})
            self.assertEqual(row["last_over_first_wall_drift"], {"baseline": 1e7, "candidate": 1e7})
        with self.assertRaisesRegex(CampaignError, "incomplete matched pair"):
            paired_summary(stages[:-1])

    def test_seven_samples_and_spread(self):
        self.assertEqual(spread([1, 2, 3, 4, 5, 6, 7]), {"median": 4, "min": 1, "max": 7, "mad": 2})
        for values in ([1] * 6, [1] * 8, [1] * 6 + [float("nan")], [1] * 6 + [0]):
            with self.assertRaises(CampaignError):
                spread(values)

    def test_busy_monitor_is_descriptive_but_invalid_counters_fail(self):
        samples = [{"monotonic_seconds": 1, "counters": {"cpu23": [0] * 8}},
                   {"monotonic_seconds": 2, "counters": {"cpu23": [100, 0, 0, 0, 0, 0, 0, 0]}}]
        validate_monitor(samples)  # 0% idle is explicitly not a veto.
        for mutation in ("counter", "clock", "topology"):
            invalid = copy.deepcopy(samples)
            if mutation == "counter":
                invalid[1]["counters"]["cpu23"][1] = -1
            elif mutation == "clock":
                invalid[1]["monotonic_seconds"] = 1
            else:
                invalid[1]["counters"] = {}
            with self.assertRaises(CampaignError):
                validate_monitor(invalid)

    def test_failed_command_retains_log_exit_and_monitor(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            with self.assertRaisesRegex(CampaignError, "exit 9"):
                run_campaign([sys.executable, "-c", "print('retained failure', flush=True); raise SystemExit(9)"],
                             directory, {min(os.sched_getaffinity(0))}, dict(os.environ))
            self.assertIn("retained failure", (directory / "campaign.log").read_text())
            record = json.loads((directory / "command.json").read_text())
            self.assertEqual(record["exit_code"], 9)
            self.assertTrue(record["owned_process_waited"])
            self.assertEqual(record["active_owned_processes"], [])
            self.assertGreaterEqual(len(json.loads((directory / "cpu-monitor.json").read_text())), 2)

    def test_installed_payload_mismatch_fails_even_with_same_version(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            environment = directory / "env"
            venv.EnvBuilder(with_pip=False).create(environment)
            site = next((environment / "lib").glob("python*/site-packages"))
            package = site / "eqiora"
            package.mkdir()
            (package / "__init__.py").write_text("value = 1\n")
            (package / "_eqiora.py").write_text("value = 2\n")
            (site / "numpy.py").write_text("# Synthetic package; no numerical execution.\n")
            metadata = site / "eqiora-0.0.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: eqiora\nVersion: 0.0.0\n")
            wheel = directory / "synthetic.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for path in package.glob("*.py"):
                    archive.write(path, "eqiora/" + path.name)
            spec = {"python": environment / "bin/python", "wheel": wheel}
            self.assertEqual(check_installation("matching", spec, directory)["payload_files_checked"], 2)
            (package / "_eqiora.py").write_text("value = 999\n")
            with self.assertRaisesRegex(CampaignError, "installed payload verification failed"):
                check_installation("changed", spec, directory)
            record = json.loads((directory / "changed-probe.json").read_text())
            self.assertNotEqual(record["exit_code"], 0)
            self.assertIn("installed wheel payload differs", record["stderr"])

    def test_monitor_failure_terminates_owned_process_group(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            calls = 0
            def sample(_):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise CampaignError("monitor unavailable")
                return {"monotonic_seconds": 1, "counters": {"cpu0": [0] * 8}}
            with patch("run_matched.cpu_sample", side_effect=sample):
                with self.assertRaisesRegex(CampaignError, "monitor unavailable"):
                    run_campaign([sys.executable, "-c", "import time; time.sleep(60)"], directory, {0}, dict(os.environ))
            record = json.loads((directory / "command.json").read_text())
            self.assertEqual(record["exit_code"], -signal.SIGTERM)
            self.assertTrue(record["owned_process_waited"])
            self.assertEqual(record["active_owned_processes"], [])
            self.assertTrue((directory / "cpu-monitor.json").exists())


if __name__ == "__main__":
    unittest.main()
