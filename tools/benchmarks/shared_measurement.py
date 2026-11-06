"""Shared execution and measurement bookkeeping for external comparisons.

Problem adapters own geometry, field associations and scientific acceptance.
The common runner owns clocks, candidate retention and execution counts.
"""
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

class ComparisonError(ValueError):
    """The declared mathematical or measurement contract was not met."""

ROOT = Path(__file__).resolve().parents[2]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "BLIS_NUM_THREADS")



def shared_sources():
    """Bind shared implementation bytes as well as each problem's drivers."""
    return tuple(Path(__file__).with_name(name).resolve()
                 for name in ("shared_measurement.py", "shared_accuracy.py", "shared_campaign.py"))


def float64_hex(value):
    """Retain scalar/vector field shape and nonfinite status in candidate JSON."""
    return [float64_hex(item) for item in value] if isinstance(value, list) else float(value).hex()


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


def environment(tool, import_seconds, *, protocol_root):
    protocol_root = Path(protocol_root)
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
            "protocol_sha256": digest(protocol_root / "README.md"),
            "accuracy_driver_sha256": digest(protocol_root / "accuracy.py"),
            "harness_sha256": {**{path.name: digest(path) for path in sorted(protocol_root.glob("*.py"))},
                               **{str(path.relative_to(ROOT)): digest(path) for path in shared_sources()}}}


def finish(args, tool, prepared, execute, extract, profile, *, accuracy, diagnostic_extra_calls=0):
    """Execute fresh numerical work on retained mesh/forms, not retained systems."""
    if type(diagnostic_extra_calls) is not int or diagnostic_extra_calls < 0:
        raise ComparisonError("diagnostic extra-call count must be a nonnegative integer")
    report = prepared
    report.update(tool=tool, dimension=args.dimension, cells_per_axis=args.cells, mode=args.mode,
                  stable_timing_claim=False, measurement_status=("correctness-only smoke; not performance evidence" if args.mode == "correctness-only"
                                      else "unqualified timing collection; no stable reference-environment claim"))
    result, cold = clock(execute)
    report["memory_after_execution_before_accuracy"] = memory()
    coordinates, values, solve = extract(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the actual candidate even if an accuracy/export gate rejects.
    # Hex strings retain non-finite status as valid JSON and finite values exactly.
    candidate = {"dimension": args.dimension, "coordinates_float64_hex": [[float(x).hex() for x in row] for row in coordinates],
                 "values_float64_hex": float64_hex(values.tolist()), "solve": solve}
    args.output.with_name("candidate-field.json").write_text(json.dumps(candidate, indent=2, allow_nan=False)+"\n")
    report["accuracy"] = accuracy(coordinates, values, args.cells)
    report["solve"] = solve
    report["profile"] = profile()
    report["diagnostic_execution"] = {
        "additional_calls_per_tool": 1 + diagnostic_extra_calls,
        "position": "after cold ordinary execution and validation, before any warmups/samples",
        "ordinary_execution_has_phase_instrumentation": False,
        "cold_memory_scope": "process HWM immediately after ordinary field return, before output/protocol inspection, accuracy or profiling",
        "final_memory_scope": "whole child, including untimed inspection/accuracy/reporting/diagnostic profiling"}
    if diagnostic_extra_calls:
        report["diagnostic_execution"]["additional_problem_probe_calls"] = diagnostic_extra_calls
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



def child_environment(backend: str, executable: Path, output: Path) -> dict:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
        env.pop(key, None)
    if backend == "fenicsx":
        # Never load the Eqiora build/test environment's OpenMPI or other
        # foreign libraries ahead of the isolated MPICH conda-forge RPATH.
        for key in ("LD_LIBRARY_PATH", "MPICC", "MPICXX", "PETSC_OPTIONS", "PETSC_OPTIONS_YAML", "SLEPC_OPTIONS",
                    "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LIBRARY_PATH",
                    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            env.pop(key, None)
        for key in list(env):
            if key.startswith(("OMPI_", "PMI_", "PMIX_", "I_MPI_", "MPICH_", "MPIEXEC_")):
                env.pop(key)
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    env.update({key: "1" for key in THREAD_KEYS})
    env.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(output), XDG_CACHE_HOME=str(output / "cache"))
    return env

