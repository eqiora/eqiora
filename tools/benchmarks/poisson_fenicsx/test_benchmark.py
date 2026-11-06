"""Synthetic clocks and independent fields; no native tools or timing campaign."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import common
from accuracy import ComparisonError
from test_protocol import analytic_discrete_solution, driver


class BenchmarkTests(unittest.TestCase):
    def finish_case(self, output, *, dimension=3, cells=4, tool="eqiora",
                    mode="benchmark", phases=True, corrupt=False):
        coordinates, values = analytic_discrete_solution(cells, dimension)
        if corrupt:
            values = values * 1.1
        args = SimpleNamespace(dimension=dimension, cells=cells, mode=mode, output=output)
        events, timed, extracted = [], [], []
        in_clock = False
        calls = 0
        clock_calls = 0

        def execute():
            nonlocal calls
            calls += 1
            timed.append(in_clock)
            events.append("execute")
            return calls

        def extract(result):
            self.assertFalse(in_clock, "output inspection must remain outside the clock")
            extracted.append(result)
            events.append("extract")
            return coordinates, values, {"fixture": "independently derived discrete sine field"}

        def clock(function):
            nonlocal in_clock, clock_calls
            in_clock = True
            result = function()
            in_clock = False
            clock_calls += 1
            # Hand-calculable samples 1..7; cold and diagnostic must not enter them.
            wall = 1000.0 if clock_calls == 1 else float(clock_calls-1)
            events.append("clock-stop")
            return result, {"wall_seconds": wall, "process_cpu_seconds": wall/2}

        def memory():
            self.assertFalse(in_clock)
            events.append("memory")
            return {"process_peak_rss_kib": 123}

        def accuracy(*arguments):
            self.assertFalse(in_clock)
            events.append("accuracy")
            return actual_accuracy(*arguments)

        def profile():
            self.assertFalse(in_clock)
            events.append("diagnostic")
            extract(execute())
            return {"required_phases_available": phases,
                    "fixture_diagnostic_seconds": 10000.0}

        prepared = {"environment": {}, "mesh_measurement": {"fixture": "mesh"},
                    "preparation_measurement": {"fixture": "preparation"}}
        self.last_timed = timed
        actual_accuracy = common.check_accuracy
        with patch.object(common, "clock", clock), patch.object(common, "memory", memory), \
                patch.object(common, "check_accuracy", accuracy), patch.object(common.os, "getloadavg", return_value=(0, 0, 0)):
            common.finish(args, tool, prepared, execute, extract, profile)
        return json.loads(output.read_text()), events, timed, extracted

    def test_both_dimensions_use_the_existing_cold_diagnostic_warm_sample_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            for dimension in (2, 3):
                output = Path(temporary) / str(dimension) / "result.json"
                report, events, timed, extracted = self.finish_case(output, dimension=dimension)
                self.assertEqual(report["dimension"], dimension)
                self.assertEqual(events[:8], ["execute", "clock-stop", "memory", "extract",
                                             "accuracy", "diagnostic", "execute", "extract"])
                self.assertEqual(timed, [True, False, False, False] + [True]*7)
                self.assertEqual(extracted, list(range(1, 12)))
                timing = report["execution_measurement"]
                self.assertEqual(timing["warmups"], 2)
                self.assertEqual(timing["cold"]["wall_seconds"], 1000)
                self.assertEqual([row["wall_seconds"] for row in timing["samples"]], list(range(1, 8)))
                self.assertEqual([row["process_cpu_seconds"] for row in timing["samples"]], [value/2 for value in range(1, 8)])
                self.assertEqual((timing["median_wall_seconds"], timing["min_wall_seconds"],
                                  timing["max_wall_seconds"], timing["median_absolute_deviation_seconds"]),
                                 (4, 1, 7, 2))
                self.assertEqual(report["memory_after_execution_before_accuracy"], {"process_peak_rss_kib": 123})
                self.assertFalse(report["stable_timing_claim"])
                self.assertIn("unqualified", report["measurement_status"])

    def test_missing_3d_phases_and_wrong_field_reject_before_warm_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, options, message in (("telemetry", {"phases": False}, "required_reference_linear_solve_phase_unavailable"),
                                           ("field", {"corrupt": True}, None)):
                output = Path(temporary) / name / "result.json"
                with self.assertRaises(ComparisonError) as error:
                    self.finish_case(output, **options)
                if message:
                    self.assertIn(message, str(error.exception))
                self.assertEqual(self.last_timed, [True, False] if name == "telemetry" else [True])
                self.assertFalse(output.exists(), "failed worker must not publish a completed report")
                self.assertTrue(output.with_name("candidate-field.json").exists())
                candidate = json.loads(output.with_name("candidate-field.json").read_text())
                self.assertEqual(candidate["dimension"], 3)

    def test_correctness_only_has_no_warm_sample_or_stage_timing_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            for dimension in (2, 3):
                report, _, timed, extracted = self.finish_case(
                    Path(temporary) / str(dimension) / "result.json", dimension=dimension,
                    mode="correctness-only", phases=False)
                self.assertEqual(timed, [True, False])
                self.assertEqual(extracted, [1, 2])
                for key in ("execution_measurement", "mesh_measurement", "preparation_measurement"):
                    self.assertNotIn(key, report)

    def driver_arguments(self, output, *, levels=None):
        args = ["run.py", str(output), "--eqiora-python", "/installed/eqiora/python",
                "--fenicsx-python", "/installed/fenicsx/python", "--eqiora-wheel", "/retained/wheel.whl",
                "--eqiora-revision", "a"*40, "--eqiora-build-profile", "synthetic test profile",
                "--dimension", "3", "--mode", "benchmark", "--cpu", str(min(os.sched_getaffinity(0)))]
        return args if levels is None else args + ["--levels", *map(str, levels)]

    def test_3d_driver_collects_both_tools_all_declared_levels_and_convergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"
            commands = []
            def worker(command, **kwargs):
                commands.append(command)
                tool = Path(command[2]).stem.removesuffix("_case")
                dimension = int(command[command.index("--dimension")+1])
                cells = int(command[command.index("--cells")+1])
                mode = command[command.index("--mode")+1]
                target = Path(command[command.index("--output")+1])
                self.finish_case(target, dimension=dimension, cells=cells, tool=tool, mode=mode)
                return SimpleNamespace(returncode=0)
            with patch("sys.argv", self.driver_arguments(output)), patch.object(driver.os, "sched_setaffinity"), \
                    patch.object(driver.subprocess, "run", worker), redirect_stdout(io.StringIO()):
                self.assertEqual(driver.main(), 0)
            self.assertEqual([(Path(row[2]).stem, int(row[row.index("--cells")+1])) for row in commands],
                             [("eqiora_case", 4), ("fenicsx_case", 4), ("fenicsx_case", 8),
                              ("eqiora_case", 8), ("eqiora_case", 16), ("fenicsx_case", 16)])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual((summary["dimension"], summary["levels"]), (3, [4, 8, 16]))
            self.assertTrue(summary["complete_timing_comparison"])
            self.assertFalse(summary["stable_timing_claim"])
            self.assertTrue(all(len(orders) == 2 and all(order > 1.9 for order in orders)
                                for orders in summary["observed_orders"].values()))

    def test_3d_driver_rejects_undeclared_level_and_retains_worker_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"
            with patch("sys.argv", self.driver_arguments(output, levels=[4, 32])), \
                    patch.object(driver.subprocess, "run") as worker, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.main()
            worker.assert_not_called()
            self.assertFalse(output.exists())
            def failed_worker(command, **kwargs):
                kwargs["stdout"].write("required_reference_linear_solve_phase_unavailable\n")
                return SimpleNamespace(returncode=2)
            with patch("sys.argv", self.driver_arguments(output)), patch.object(driver.os, "sched_setaffinity"), \
                    patch.object(driver.subprocess, "run", failed_worker), redirect_stdout(io.StringIO()):
                self.assertEqual(driver.main(), 2)
            self.assertFalse((output / "summary.json").exists())
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual((failure["backend"], failure["cells"], failure["exit_code"]), ("eqiora", 4, 2))
            self.assertIn("required_reference_linear_solve_phase_unavailable", Path(failure["log"]).read_text())


if __name__ == "__main__":
    unittest.main()
