"""Isolated child collection shared by external comparison problems."""
import json
import os
import subprocess
from shared_measurement import ROOT, child_environment


def collect_reports(args, levels, *, problem_root):
    """Collect each tool/mesh in fresh processes; the caller owns scientific gates.

    Callers validate their level/dimension policy and output/CPU arguments first.
    A failed worker retains its log and failure record and returns None.
    """
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
            command = [str(executable), "-I", str(problem_root / f"{backend}_case.py"),
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
                return None
            reports.append(json.loads(output.read_text()))
    return reports
