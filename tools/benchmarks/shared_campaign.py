"""Isolated child collection shared by external comparison problems."""
from dataclasses import dataclass
import re
import json
import os
import subprocess
from shared_measurement import ROOT, ComparisonError, child_environment


@dataclass(frozen=True)
class ComparisonCase:
    """A problem-owned directory label and worker arguments."""

    label: str
    worker_arguments: tuple[str, ...]


def collect_reports(args, cases, *, problem_root):
    """Collect each tool/mesh in fresh processes; the caller owns scientific gates.

    Callers validate their scientific policy and output/CPU arguments first.
    A failed worker retains its log and failure record and returns None.
    """
    cases = tuple(cases)
    labels = set()
    common = {"--mode", "--output", "--wheel", "--revision", "--build-profile"}
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case.label) or case.label in labels:
            raise ComparisonError("comparison case labels must be unique safe directory names")
        if any(argument.split("=", 1)[0] in common or argument == "--"
               for argument in case.worker_arguments):
            raise ComparisonError("comparison case arguments must not override collector arguments")
        labels.add(case.label)
    if not cases:
        raise ComparisonError("comparison requires at least one case")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0, {args.cpu})
    reports = []
    for index, case in enumerate(cases):
        # Alternate launch order, retaining separate environments/processes.
        backends = ("eqiora", "fenicsx") if index % 2 == 0 else ("fenicsx", "eqiora")
        for backend in backends:
            directory = args.output / case.label / backend
            directory.mkdir(parents=True)
            executable = (args.eqiora_python if backend == "eqiora" else args.fenicsx_python).absolute()
            output = directory / "result.json"
            command = [str(executable), "-I", str(problem_root / f"{backend}_case.py"),
                       *case.worker_arguments, "--mode", args.mode, "--output", str(output)]
            if backend == "eqiora":
                command += ["--wheel", str(args.eqiora_wheel.resolve()), "--revision", args.eqiora_revision,
                            "--build-profile", args.eqiora_build_profile]
            with (directory / "worker.log").open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, env=child_environment(backend, executable, directory),
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                error = {"error": "comparison_worker_failed", "backend": backend, "case": case.label,
                         "exit_code": completed.returncode, "log": str(directory / "worker.log"),
                         "complete_comparison": False}
                (args.output / "failure.json").write_text(json.dumps(error, indent=2)+"\n")
                print(json.dumps(error))
                return None
            reports.append(json.loads(output.read_text()))
    return reports
