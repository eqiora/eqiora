"""Run isolated installed-library children; no library is imported by this driver."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_measurement import THREAD_KEYS, child_environment
from accuracy import declared_levels

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--eqiora-python", type=Path, required=True)
    parser.add_argument("--fenicsx-python", type=Path, required=True)
    parser.add_argument("--eqiora-wheel", type=Path, required=True)
    parser.add_argument("--eqiora-revision", required=True)
    parser.add_argument("--eqiora-build-profile", required=True)
    parser.add_argument("--mode", choices=("correctness-only", "benchmark"), required=True)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--levels", type=int, nargs="+", choices=(4, 8, 16, 32))
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    levels = sorted(set(args.levels if args.levels is not None else declared_levels(args.dimension)))
    if any(level not in declared_levels(args.dimension) for level in levels):
        parser.error("mesh level has no declared reference for this dimension")
    if len(levels) < 2:
        parser.error("at least two declared levels are required for convergence")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("requested CPU is outside the caller's affinity")
    if args.output.exists():
        parser.error("output directory must not already exist; preserve earlier receipts")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0, {args.cpu})
    reports = []
    for index, cells in enumerate(levels):
        # Alternate launch order, retaining separate environments/processes.
        backends = ("eqiora", "fenicsx") if index % 2 == 0 else ("fenicsx", "eqiora")
        for backend in backends:
            directory = args.output / f"n{cells}" / backend
            directory.mkdir(parents=True)
            executable = (args.eqiora_python if backend == "eqiora" else args.fenicsx_python).absolute()
            output = directory / "result.json"
            command = [str(executable), "-I", str(HERE / f"{backend}_case.py"),
                       "--dimension", str(args.dimension), "--cells", str(cells), "--mode", args.mode, "--output", str(output)]
            if backend == "eqiora":
                command += ["--wheel", str(args.eqiora_wheel.resolve()), "--revision", args.eqiora_revision,
                            "--build-profile", args.eqiora_build_profile]
            with (directory / "worker.log").open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, env=child_environment(backend, executable, directory),
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                error = {"error": "comparison_worker_failed", "backend": backend, "cells": cells,
                         "exit_code": completed.returncode, "log": str(directory / "worker.log"),
                         "complete_comparison": False}
                (args.output / "failure.json").write_text(json.dumps(error, indent=2)+"\n")
                print(json.dumps(error))
                return 2
            reports.append(json.loads(output.read_text()))
    orders = {}
    for backend in ("eqiora", "fenicsx"):
        rows = sorted((row for row in reports if row["tool"] == backend), key=lambda row: row["cells_per_axis"])
        values = []
        for previous, current in zip(rows, rows[1:]):
            error0, error1 = previous["accuracy"]["continuous_l2"], current["accuracy"]["continuous_l2"]
            order = math.log(error0/error1) / math.log(current["cells_per_axis"]/previous["cells_per_axis"])
            if not math.isfinite(order) or not error1 < error0 or order <= 1.9:
                raise ValueError(f"{backend} convergence violates the predeclared order >1.9")
            values.append(order)
        orders[backend] = values
    summary = {"mode": args.mode, "complete_timing_comparison": args.mode == "benchmark",
               "stable_timing_claim": False, "dimension": args.dimension,
               "scope": f"{args.dimension}D tensor Q1 unit-box scalar Poisson only",
               "observed_orders": orders, "levels": levels,
               "required_eqiora_phases_available": all(row["profile"]["required_phases_available"] for row in reports if row["tool"] == "eqiora"),
               "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "protocol_sha256": hashlib.sha256((HERE / "README.md").read_bytes()).hexdigest(),
               "external_lock_sha256": hashlib.sha256((HERE / "pixi.lock").read_bytes()).hexdigest(),
               "reports": [{"tool": row["tool"], "cells_per_axis": row["cells_per_axis"], "accuracy": row["accuracy"]} for row in reports]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(args.output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
