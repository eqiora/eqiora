#!/usr/bin/env python3
"""Installed-wheel six-case gate through the existing ordinary producer/oracle."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    # Set controls before importing the installed native package or NumPy.
    allowed = os.sched_getaffinity(0)
    if not allowed:
        raise RuntimeError("no allowed CPU for the serial verification worker")
    cpu = min(allowed)
    os.sched_setaffinity(0, {cpu})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in tuple(os.environ):
        if name.startswith(("OMPI_", "PMI_", "PMIX_", "MV2_")):
            del os.environ[name]
    scratch = Path.home() / ".cache/eqiora/affine-q2-scalar-registered"
    scratch.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="attempt-", dir=scratch))
    print(f"affine-q2-registered: retained output {output}", flush=True)
    # This script runs in a fresh interpreter. Exact file loaders keep its generic
    # producer/reference filenames out of the parent pytest module namespace.
    sys.path.insert(0, str(HERE))
    producer = load("eqiora_affine_q2_producer", HERE / "check.py")
    sys.path.insert(0, str(HERE / "reference"))
    comparator = load("eqiora_affine_q2_comparator", HERE / "reference/compare.py")
    from types import SimpleNamespace
    states = []
    try:
        for dimension in (2, 3):
            candidates = []
            for n in (2, 4, 8):
                directory = output / f"d{dimension}-n{n}"
                directory.mkdir()
                state = {"dimension": dimension, "N": n, "ordinary_run_attempted": 0, "ordinary_run_completed": 0}
                states.append(state)
                args = SimpleNamespace(dimension=dimension, cells=n, started_at=datetime.now(timezone.utc).isoformat())
                producer.run(args, directory, state)
                candidate = json.loads((directory / "candidate.json").read_text())
                comparison = comparator.compare(candidate)
                producer.write(directory / "comparison.json", comparison)
                if not comparison["accepted"]:
                    raise RuntimeError(f"independent comparator rejected {dimension}D N={n}")
                candidates.append(candidate)
            levels = comparator.compare_levels(candidates)
            producer.write(output / f"levels-d{dimension}.json", levels)
            if not levels["accepted"]:
                raise RuntimeError(f"independent finite-level gate rejected {dimension}D")
    except Exception as error:
        producer.write(output / "failure.json", {"type": type(error).__name__, "message": str(error), "states": states})
        raise
    producer.write(output / "completed.json", {"states": states, "ordinary_solves": sum(s["ordinary_run_completed"] for s in states), "profiles": 0, "warmups": 0, "samples": 0})
    print("affine-q2-registered: six complete ordinary fields and both finite-level gates PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
