"""Measure the ordinary installed Run; preparation and diagnostics are separate."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--edition", choices=("a7", "candidate"), required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--wheel", type=Path, required=True)
parser.add_argument("--workload", choices=("cylinder", "decay"), required=True)
parser.add_argument("--repetitions", type=int, default=7)
parser.add_argument("--warmups", type=int, default=2)
parser.add_argument("--calls-per-sample", type=int, default=1)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
if args.repetitions < 3 or args.warmups < 0 or args.calls_per_sample < 1:
    parser.error("require repetitions >= 3, warmups >= 0, calls-per-sample >= 1")
args.output.mkdir(parents=True, exist_ok=True)
import eqiora
import fixtures
if args.edition == "a7" and importlib.metadata.version("eqiora") != "0.1.0a7":
    parser.error("a7 fixture requires installed eqiora0.1.0a7")
if args.profile and args.edition == "a7":
    parser.error("a7 does not expose Profile; compare unprofiled timings")
if args.workload == "cylinder":
    plan, state, mesh = fixtures.cylinder(args.edition)
    controls = dict(steps=10, output_steps=tuple(range(1, 11)))
else:
    plan, state, field = fixtures.decay(args.edition)
    controls = dict(until_s=1., output_times_s=(.25, .5, 1.))
state_bytes = state.to_bytes()
(args.output / (args.workload + "-measured-plan.json")).write_bytes(plan.to_bytes())
(args.output / (args.workload + "-measured-state.json")).write_bytes(state_bytes)
report = {
    "host": platform.node(), "platform": platform.platform(), "libc": platform.libc_ver(),
    "python": sys.version, "executable": sys.executable,
    "cpu_models": sorted({line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")}),
    "logical_cpu_count": os.cpu_count(),
    "affinity": sorted(os.sched_getaffinity(0)),
    "load_at_start": os.getloadavg(),
    "versions": {n: importlib.metadata.version(n) for n in ("eqiora", "numpy", "gmsh")},
    "thread_environment": {n: os.environ.get(n) for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")},
    "workload": args.workload, "controls": controls,
    "source_preparation": {"declared_source_revision": args.revision, "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(), "edition": args.edition, "initial_data_sha256": fixtures.INITIAL_SHA256 if args.workload == "cylinder" else None},
    "plan_sha256": hashlib.sha256(plan.to_bytes()).hexdigest(),
    "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "measurement": "wall-clock and process CPU around eqiora.run only; import, compile, resolve, initial binding, result inspection and file writes excluded",
    "cold_definition": "first Run in this process after untimed input preparation; no OS/file-cache flushing",
    "warm_definition": "same immutable Plan and State, fresh ordinary Run per call; no retained Result or provider reuse supplied by this driver",
    "profiled": args.profile,
}
if args.workload == "cylinder":
    mesh_artifact = json.loads(mesh.to_bytes())
    mesh_resource = mesh_artifact.get("resources", {}).get("mesh", mesh_artifact)
    native_mesh = json.loads(base64.b64decode(mesh_resource["mesh_base64"]))
    projection = {k: native_mesh[k] for k in ("topology", "geometry", "vertices", "cells", "acceptance")}
    mesh_sha = hashlib.sha256(json.dumps(projection,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    if mesh_sha != "ac046556dc98188923b773e6fa33348f2344c894e74e23282f0978faaf68029a":
        raise ValueError("Mesh differs from original a7 input")
    report["semantic_mesh_sha256"] = mesh_sha
    report.update(vertices=mesh.vertex_count, cells=mesh.cell_count,
                  physical_initial_sha256=hashlib.sha256((fixtures.ROOT / "cylinder-initial-data.json").read_bytes()).hexdigest())

def run_once():
    before_wall, before_cpu = time.perf_counter_ns(), time.process_time_ns()
    result = eqiora.run(plan, state=state, **controls, **({"profile": True} if args.profile else {}))
    wall, cpu = time.perf_counter_ns() - before_wall, time.process_time_ns() - before_cpu
    return result, {"wall_seconds": wall / 1e9, "cpu_seconds": cpu / 1e9}

result, report["cold"] = run_once()
(args.output / (args.workload + "-measured-result.json")).write_bytes(result.to_bytes())
if args.profile:
    profile = result.profile
    report["profile"] = {"total_seconds": profile.total_seconds,
       "phases": [{"path": x.path, "fields": x.fields, "calls": x.calls,
                   "inclusive_seconds": x.inclusive_seconds, "self_seconds": x.self_seconds}
                  for x in profile.phases],
       "events": [{"path": x.path, "fields": x.fields} for x in profile.events]}
else:
    report["warmup_calls"] = args.warmups
    for _ in range(args.warmups):
        run_once()
    samples = []
    for _ in range(args.repetitions):
        calls = [run_once()[1] for _ in range(args.calls_per_sample)]
        samples.append({"calls": calls, "mean_wall_seconds": statistics.mean(x["wall_seconds"] for x in calls)})
    means = [s["mean_wall_seconds"] for s in samples]
    report.update(samples=samples, median_seconds=statistics.median(means),
                  min_seconds=min(means), max_seconds=max(means),
                  median_absolute_deviation_seconds=statistics.median(abs(x-statistics.median(means)) for x in means),
                  uncertainty="raw samples, range and MAD are reported; no stable regression bound or confidence claim is inferred from this run alone")
if state.to_bytes() != state_bytes:
    raise ValueError("Input State mutated")
if args.workload == "decay":
    series = result.series(field)
    times = series.time.numpy().reshape(-1).tolist()
    values = series.values.numpy().reshape(-1).tolist()
    error = max(abs(x - math.exp(-t)) for t, x in zip(times, values, strict=True))
    if times != [.25, .5, 1.] or error >= 1e-8:
        raise ValueError("Decay output cadence or independent analytic bound failed")
    report["decay_max_absolute_error"] = error
report["load_at_end"] = os.getloadavg()
report["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
path = args.output / (args.workload + ("-profile.json" if args.profile else "-timing.json"))
path.write_text(json.dumps(report, indent=2))
print(path)
