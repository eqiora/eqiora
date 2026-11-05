"""Fixed-budget matched revision campaigns using already installed environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

from run import HERE, ROOT, THREAD_KEYS, child_environment

LEVELS = [4, 8, 16, 32]
PAIRS = 8


class CampaignError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def schedule():
    stages = [("exploratory", None, label) for label in ("baseline", "candidate")]
    for pair in range(1, PAIRS + 1):
        order = ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")
        stages.extend(("paired", pair, label) for label in order)
    return stages


def cpu_list(value):
    result = set()
    for part in value.strip().split(","):
        bounds = list(map(int, part.split("-")))
        require(len(bounds) in (1, 2), "invalid CPU topology")
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def controls(cpu):
    paths = {
        "siblings": f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list",
        "boost": "/sys/devices/system/cpu/cpufreq/boost",
        "intel_no_turbo": "/sys/devices/system/cpu/intel_pstate/no_turbo",
    }
    siblings = cpu_list(Path(paths["siblings"]).read_text())
    for sibling in sorted(siblings):
        paths[f"cpu{sibling}_governor"] = f"/sys/devices/system/cpu/cpu{sibling}/cpufreq/scaling_governor"
    values = {}
    for key, path in paths.items():
        try:
            values[key] = Path(path).read_text().strip()
        except FileNotFoundError:
            values[key] = None
    return {"settings": values, "load": os.getloadavg(),
            "supervisor_affinity": sorted(os.sched_getaffinity(0)), "settings_changed_by_runner": False}


def cpu_sample(cpus):
    rows = {}
    for line in Path("/proc/stat").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in {f"cpu{cpu}" for cpu in cpus}:
            # Guest times are already included in user/nice; do not double count.
            values = list(map(int, parts[1:9]))
            require(len(values) == 8, "incomplete CPU monitor counters")
            rows[parts[0]] = values
    require(set(rows) == {f"cpu{cpu}" for cpu in cpus}, "CPU monitor omitted a selected CPU/sibling")
    return {"monotonic_seconds": time.monotonic(), "load": os.getloadavg(), "counters": rows}


def validate_monitor(samples):
    require(len(samples) >= 2, "CPU monitor has no interval")
    for before, after in zip(samples, samples[1:]):
        require(after["monotonic_seconds"] > before["monotonic_seconds"], "CPU monitor clock did not advance")
        require(before["counters"].keys() == after["counters"].keys(), "CPU monitor topology changed")
        for cpu in before["counters"]:
            delta = [b - a for a, b in zip(before["counters"][cpu], after["counters"][cpu], strict=True)]
            require(all(value >= 0 for value in delta), "CPU monitor counters went backwards")
            # Utilization has no acceptance threshold. Retain even busy intervals.


PROBE = r'''
import hashlib, importlib.metadata as m, json, pathlib, sys, zipfile
import eqiora, eqiora._eqiora, numpy
dist = m.distribution("eqiora")
payloads = {}
with zipfile.ZipFile(sys.argv[1]) as wheel:
    members = [name for name in wheel.namelist() if name.startswith("eqiora/") and not name.endswith("/")]
    if not members or "eqiora/__init__.py" not in members:
        raise RuntimeError("wheel has no Eqiora package payload")
    for name in members:
        installed = pathlib.Path(dist.locate_file(name))
        if hashlib.sha256(wheel.read(name)).digest() != hashlib.sha256(installed.read_bytes()).digest():
            raise RuntimeError("installed wheel payload differs: " + name)
        payloads[str(installed)] = hashlib.sha256(installed.read_bytes()).hexdigest()
if pathlib.Path(eqiora.__file__).resolve() != pathlib.Path(dist.locate_file("eqiora/__init__.py")).resolve():
    raise RuntimeError("imported Eqiora differs from installed distribution")
print(json.dumps({"python": sys.version, "versions": {d.metadata["Name"].lower().replace("_", "-"): d.version for d in m.distributions()},
                  "module": eqiora.__file__, "native": eqiora._eqiora.__file__, "payload_files_checked": len(members), "payloads_sha256": payloads}))
'''


def check_installation(label, spec, directory):
    command = [str(spec["python"]), "-I", "-c", PROBE, str(spec["wheel"])]
    process = None
    record = {"command": command, "exit_code": None}
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment("eqiora", spec["python"], directory),
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        record["stdout"], record["stderr"] = process.communicate()
        record["exit_code"] = process.returncode
        require(process.returncode == 0, f"{label} installed payload verification failed; see probe log")
        return json.loads(record["stdout"])
    finally:
        try:
            if process is not None:
                stop_group(process)
                record["exit_code"] = process.returncode
        finally:
            save(directory / f"{label}-probe.json", record)


def active_group(group):
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def stop_group(process):
    # The session was created by this runner. Never signal another process group.
    if process.poll() is None or active_group(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while active_group(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if active_group(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    deadline = time.monotonic() + 5
    while active_group(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    require(not active_group(process.pid), "owned campaign processes remain active")


def run_campaign(command, directory, cpus, env):
    samples = []
    process = None
    record = {"command": command, "exit_code": None, "owned_process_waited": False}
    try:
        samples.append(cpu_sample(cpus))
        with (directory / "campaign.log").open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                try:
                    record["exit_code"] = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    samples.append(cpu_sample(cpus))
            samples.append(cpu_sample(cpus))
        validate_monitor(samples)
        require(record["exit_code"] == 0, f"campaign failed with exit {record['exit_code']}; see {directory / 'campaign.log'}")
        require(not active_group(process.pid), "campaign returned with active owned descendants")
    finally:
        try:
            if process is not None:
                try:
                    stop_group(process)
                finally:
                    record.update(exit_code=process.returncode, owned_process_waited=process.poll() is not None,
                                  active_owned_processes=active_group(process.pid))
        finally:
            save(directory / "command.json", record)
            save(directory / "cpu-monitor.json", samples)


def spread(values):
    require(len(values) == 7 and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in values),
            "campaign must retain seven finite positive timing samples")
    median = statistics.median(values)
    return {"median": median, "min": min(values), "max": max(values),
            "mad": statistics.median(abs(value - median) for value in values)}


def read_campaign(directory, spec, cpu):
    summary = json.loads((directory / "summary.json").read_text())
    require(summary["levels"] == LEVELS and summary["mode"] == "benchmark" and summary["required_eqiora_phases_available"] is True,
            "campaign omitted levels, benchmark mode or required phases")
    require(set(summary["observed_orders"]) == {"eqiora", "fenicsx"} and
            all(len(orders) == 3 and all(math.isfinite(order) and order > 1.9 for order in orders)
                for orders in summary["observed_orders"].values()), "campaign convergence check incomplete")
    rows = []
    for cells in LEVELS:
        for tool in ("eqiora", "fenicsx"):
            report = json.loads((directory / f"n{cells}" / tool / "result.json").read_text())
            env, timing = report["environment"], report["execution_measurement"]
            require(report["tool"] == tool and report["cells_per_axis"] == cells and report["mode"] == "benchmark", "mismatched worker result")
            require(env["affinity"] == [cpu] and all(env["thread_environment"][key] == "1" for key in THREAD_KEYS), "worker affinity/threads differ")
            require(timing["warmups"] == 2 and report["diagnostic_execution"]["additional_calls_per_tool"] == 1,
                    "worker changed warmup/diagnostic count")
            if tool == "eqiora":
                require(env["wheel_sha256"] == spec["sha256"] and env["source_revision_declared"] == spec["revision"]
                        and env["rust_build_profile_declared"] == spec["profile"], "worker provenance differs")
            wall = spread([row["wall_seconds"] for row in timing["samples"]])
            cpu_times = spread([row["process_cpu_seconds"] for row in timing["samples"]])
            rows.append({"tool": tool, "cells": cells, "wall": wall, "cpu": cpu_times,
                         "plan_sha256": report.get("plan_sha256"),
                         "median_cpu_over_wall": cpu_times["median"] / wall["median"],
                         "software": {key: env[key] for key in ("python", "versions")}})
    return rows


def paired_summary(stages):
    rows = []
    for tool in ("eqiora", "fenicsx"):
        for cells in LEVELS:
            pairs = []
            for pair in range(1, PAIRS + 1):
                selected = [stage for stage in stages if stage["pair"] == pair]
                require(len(selected) == 2 and {stage["label"] for stage in selected} == {"baseline", "candidate"}, "incomplete matched pair")
                values = {stage["label"]: next(row for row in stage["rows"] if row["tool"] == tool and row["cells"] == cells)
                          for stage in selected}
                pairs.append({"pair": pair, "order": [stage["label"] for stage in selected],
                              "candidate_over_baseline_wall": values["candidate"]["wall"]["median"] / values["baseline"]["wall"]["median"],
                              "candidate_over_baseline_cpu": values["candidate"]["cpu"]["median"] / values["baseline"]["cpu"]["median"],
                              "campaigns": values})
            ratios = [pair["candidate_over_baseline_wall"] for pair in pairs]
            cpu_ratios = [pair["candidate_over_baseline_cpu"] for pair in pairs]
            rows.append({"tool": tool, "cells": cells, "pairs": pairs,
                         "wall_ratio_median": statistics.median(ratios), "wall_ratio_range": [min(ratios), max(ratios)],
                         "cpu_ratio_median": statistics.median(cpu_ratios), "cpu_ratio_range": [min(cpu_ratios), max(cpu_ratios)],
                         "wall_ratio_median_by_order": {order: statistics.median(pair["candidate_over_baseline_wall"] for pair in pairs
                             if pair["order"][0] == order) for order in ("baseline", "candidate")},
                         "last_over_first_wall_drift": {label: pairs[-1]["campaigns"][label]["wall"]["median"] /
                             pairs[0]["campaigns"][label]["wall"]["median"] for label in ("baseline", "candidate")}})
    return {"independent_units": "eight matched pairs; inner calls are not independent replicates",
            "stable_timing_claim": False, "inference": "Descriptive only; no #802 non-regression or cylinder-speed claim", "rows": rows}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fenicsx-python", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--supervisor-cpu", type=int, required=True)
    for label in ("baseline", "candidate"):
        for field in ("python", "wheel", "build-record"):
            parser.add_argument(f"--{label}-{field}", type=Path, required=True)
        for field in ("sha256", "revision", "build-profile"):
            parser.add_argument(f"--{label}-{field}", required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output must be new; preserve earlier attempts")
    require({args.cpu, args.supervisor_cpu} <= os.sched_getaffinity(0), "worker/supervisor CPU outside caller affinity")
    require(args.supervisor_cpu not in cpu_list(Path(f"/sys/devices/system/cpu/cpu{args.cpu}/topology/thread_siblings_list").read_text()),
            "supervisor must use a separate physical core")
    specs = {}
    for label in ("baseline", "candidate"):
        spec = {field: getattr(args, f"{label}_{field}") for field in ("python", "wheel", "sha256", "revision", "build_record")}
        spec["profile"] = getattr(args, f"{label}_build_profile")
        require(re.fullmatch("[0-9a-f]{40}", spec["revision"]) is not None and bool(spec["profile"].strip()), "full source revision and build profile required")
        require(re.fullmatch("[0-9a-f]{64}", spec["sha256"]) is not None and digest(spec["wheel"]) == spec["sha256"], "wheel SHA-256 differs")
        for key in ("python", "wheel", "build_record"):
            spec[key] = spec[key].absolute()
            require(spec[key].is_file(), f"{label} {key} missing")
        require(spec["build_record"].stat().st_size > 0, "build record must not be empty")
        specs[label] = spec
    return args, specs


def interrupt_campaign(*_):
    raise KeyboardInterrupt("SIGTERM")


def main(argv=None):
    args, specs = arguments(argv)
    args.output = args.output.absolute()
    args.output.mkdir(parents=True)
    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {args.supervisor_cpu})
    previous_term = signal.signal(signal.SIGTERM, interrupt_campaign)
    stages = []
    try:
        inputs = {str(path): digest(path) for path in [*HERE.glob("*.py"), HERE / "README.md", HERE / "pixi.lock",
                   ROOT / "examples/inverse-poisson.eqi", ROOT / "verify/numerics/cartesian-poisson-fem-fvm/expected/convergence.csv",
                   *(spec[key] for spec in specs.values() for key in ("wheel", "build_record"))]}
        save(args.output / "plan.json", {"schedule": schedule(), "levels": LEVELS, "cpu": args.cpu,
             "supervisor_cpu": args.supervisor_cpu, "fenicsx_python": str(args.fenicsx_python.absolute()), "inputs_sha256": inputs,
             "installations": {label: {key: str(value) for key, value in spec.items()} for label, spec in specs.items()},
             "provenance_note": "Build records retained as supplied; runner does not authenticate source/build declarations or installed-family evidence."})
        probes = {}
        for label, spec in specs.items():
            shutil.copyfile(spec["build_record"], args.output / f"{label}-build-record")
            probes[label] = check_installation(label, spec, args.output)
        require(probes["baseline"]["python"] == probes["candidate"]["python"] and
                {k: v for k, v in probes["baseline"]["versions"].items() if k != "eqiora"} ==
                {k: v for k, v in probes["candidate"]["versions"].items() if k != "eqiora"}, "installed Python/dependency versions differ")
        for probe in probes.values():
            inputs.update(probe["payloads_sha256"])
        for index, (kind, pair, label) in enumerate(schedule(), 1):
            require(all(digest(path) == value for path, value in inputs.items()), "declared inputs changed during attempt")
            directory = args.output / f"{index:02d}-{kind}-{label}"
            directory.mkdir()
            before = controls(args.cpu)
            save(directory / "controls-before.json", before)
            require(before["supervisor_affinity"] == [args.supervisor_cpu], "supervisor affinity changed")
            spec = specs[label]
            command = ["taskset", "-c", str(args.cpu), str(spec["python"]), "-I", str(HERE / "run.py"), str(directory / "results"),
                       "--eqiora-python", str(spec["python"]), "--fenicsx-python", str(args.fenicsx_python.absolute()),
                       "--eqiora-wheel", str(spec["wheel"]), "--eqiora-revision", spec["revision"],
                       "--eqiora-build-profile", spec["profile"], "--mode", "benchmark", "--cpu", str(args.cpu)]
            try:
                run_campaign(command, directory, cpu_list(before["settings"]["siblings"]), child_environment("eqiora", spec["python"], directory))
            finally:
                save(directory / "controls-after.json", controls(args.cpu))
            rows = read_campaign(directory / "results", spec, args.cpu)
            if stages:
                for row, first in zip(rows, stages[0]["rows"], strict=True):
                    actual, expected = row["software"], first["software"]
                    require(actual["python"] == expected["python"] and
                            {key: value for key, value in actual["versions"].items() if key != "eqiora"} ==
                            {key: value for key, value in expected["versions"].items() if key != "eqiora"}, "worker Python/library versions changed")
            stages.append({"kind": kind, "pair": pair, "label": label, "directory": str(directory), "rows": rows})
            save(args.output / "stages.json", stages)
        require(all(digest(path) == value for path, value in inputs.items()), "declared inputs changed during attempt")
        save(args.output / "summary.json", paired_summary(stages))
    except BaseException as error:
        save(args.output / "failure.json", {"error": f"{type(error).__name__}: {error}", "complete_comparison": False,
                                            "completed_campaigns": len(stages)})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        os.sched_setaffinity(0, original_affinity)
    print(args.output / "summary.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CampaignError, OSError, ValueError, KeyError) as error:
        print(f"matched_campaign_failed: {error}", file=sys.stderr)
        raise SystemExit(2)
