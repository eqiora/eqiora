"""Measurement bookkeeping for this single Poisson comparison."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

from accuracy import ComparisonError, LEVELS, check_accuracy

ROOT = Path(__file__).resolve().parents[3]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "BLIS_NUM_THREADS")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clock(function):
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    result = function()
    return result, {"wall_seconds": (time.perf_counter_ns()-wall)/1e9,
                    "process_cpu_seconds": (time.process_time_ns()-cpu)/1e9}


def memory():
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return {"current_rss_kib": int(status["VmRSS"].split()[0]),
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "largest_completed_child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "scope": "Linux process high-water marks; child peak is separate, not aggregate simultaneous tree memory"}


def environment(tool, import_seconds):
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise ComparisonError("all declared thread environment variables must be 1")
    if len(os.sched_getaffinity(0)) != 1:
        raise ComparisonError("comparison worker requires exactly one pinned CPU")
    names = ("eqiora", "numpy") if tool == "eqiora" else ("fenics-dolfinx", "fenics-basix", "fenics-ffcx", "fenics-ufl", "numpy", "petsc4py", "mpi4py")
    return {"host": platform.node(), "platform": platform.platform(), "python": sys.version,
            "executable": sys.executable, "affinity": sorted(os.sched_getaffinity(0)),
            "cpu_models": sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")}),
            "versions": {name: importlib.metadata.version(name) for name in names},
            "thread_environment": {key: os.environ[key] for key in THREAD_KEYS},
            "load_at_start": os.getloadavg(), "imports_wall_seconds": import_seconds,
            "numpy_and_tool_imports_included": True,
            "protocol_sha256": digest(Path(__file__).with_name("README.md")),
            "accuracy_driver_sha256": digest(Path(__file__).with_name("accuracy.py")),
            "harness_sha256": {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}}


def finish(args, tool, prepared, execute, extract, profile):
    """Execute fresh numerical work on retained mesh/forms, not retained systems."""
    report = prepared
    report.update(tool=tool, cells_per_axis=args.cells, mode=args.mode,
                  stable_timing_claim=False, measurement_status="shared-host correctness smoke; not a reference performance environment")
    result, cold = clock(execute)
    coordinates, values, solve = extract(result)
    report["memory_after_execution_before_accuracy"] = memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the actual candidate even if an accuracy/export gate rejects.
    # Hex strings remain valid JSON for NaN/Inf and preserve all binary64 bits.
    candidate = {"coordinates_float64_hex": [[float(x).hex() for x in row] for row in coordinates],
                 "values_float64_hex": [float(x).hex() for x in values], "solve": solve}
    args.output.with_name("candidate-field.json").write_text(json.dumps(candidate, indent=2, allow_nan=False)+"\n")
    report["accuracy"] = check_accuracy(coordinates, values, args.cells)
    report["solve"] = solve
    report["profile"] = profile()
    if args.mode == "benchmark":
        if not report["profile"]["required_phases_available"]:
            raise ComparisonError("required_reference_linear_solve_phase_unavailable: use explicit --mode correctness-only; no subtraction or fallback")
        for _ in range(2):
            extract(execute())
        samples = []
        for _ in range(7):
            result, timing = clock(execute)
            extract(result) # Output inspection/protocol diagnostics/destruction are untimed.
            samples.append(timing)
        med = statistics.median(item["wall_seconds"] for item in samples)
        report["execution_measurement"] = {"scope": "fresh assembly, zero-start accepted solve and field return; retained mesh/forms only; output inspection/protocol diagnostics/result destruction excluded",
            "cold": cold, "warmups": 2, "samples": samples, "median_wall_seconds": med,
            "min_wall_seconds": min(x["wall_seconds"] for x in samples),
            "max_wall_seconds": max(x["wall_seconds"] for x in samples),
            "median_absolute_deviation_seconds": statistics.median(abs(x["wall_seconds"]-med) for x in samples)}
    # Correctness-only intentionally emits no stage/execution timing table.
    if args.mode == "correctness-only":
        report.pop("mesh_measurement", None)
        report.pop("preparation_measurement", None)
    report["memory_at_end"] = memory()
    report["environment"]["load_at_end"] = os.getloadavg()
    report["coordinates"] = coordinates.tolist()
    report["values"] = values.tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def worker_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, choices=LEVELS, required=True)
    parser.add_argument("--mode", choices=("correctness-only", "benchmark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--build-profile")
    return parser.parse_args()
