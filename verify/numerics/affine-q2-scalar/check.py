#!/usr/bin/env python3
"""One installed-Python ordinary Q2 sine solve and complete physical-field export.

Run only after the production protocol/source/artifact review. This is a
correctness witness; it performs no warmups, samples, or performance comparison.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

from production import authenticate_plan, extract_candidate, require, sha256


HERE = Path(__file__).resolve().parent


def source(dimension):
    require(dimension in (2, 3), "only the declared 2D/3D boxes are supported")
    axes = "xyz"[:dimension]
    names = [f"{axis}_{side}" for axis in axes for side in ("lower", "upper")]
    parameters = [f"parameter L{axis}: m" for axis in axes] + ["parameter pi_value: 1", "parameter source_scale: 1/m^2"]
    supports = [f"support body: volume(ambient_dimension={dimension})"] + [f"support {name}: boundary(parent=body)" for name in names]
    forcing = "*".join(f"math.sin(pi_value*coordinate({i})/L{axis})" for i, axis in enumerate(axes))
    boundaries = "\n".join(f"  relation fixed_{name} on {name} {{ trace(u)=0; }}" for name in names)
    return "public component AffineQ2Sine(\n  "+",\n  ".join(supports+parameters)+") {\n  variable u: 1 on body in h1;\n  relation balance on body { -div(grad(u))=source_scale*"+forcing+"; }\n"+boundaries+"\n}\n"


def write(path, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True)+"\n").encode()
    with path.open("xb") as handle:
        handle.write(raw)
    return sha256(raw)


def utc():
    return datetime.now(timezone.utc).isoformat()


def run(args, directory, state):
    # Import the installed package only after the launch inputs/controls are checked.
    import eqiora
    import eqiora._eqiora as native

    lengths = (2, 3, 4)[:args.dimension]
    axes = "xyz"[:args.dimension]
    names = [f"{axis}_{side}" for axis in axes for side in ("lower", "upper")]
    graph = eqiora.geometry.GeometryGraph()
    primitive = (graph.rectangle if args.dimension == 2 else graph.cuboid)(**{f"{axis}_bounds": (0., float(length)) for axis, length in zip(axes, lengths)})
    geometry = graph.build(primitive, named_topology={"body": primitive.region, **dict(zip(names, primitive.boundaries, strict=True))})
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(args.cells,)*args.dimension)))
    authored = source(args.dimension)
    write(directory/"model.eqi", authored.encode())
    values = {f"L{axis}": float(length) for axis, length in zip(axes, lengths)}
    # The independent comparator encloses exact-pi reference residuals, so it also
    # includes any error due to these ordinary binary64 authored parameter values.
    values.update(pi_value=math.pi, source_scale=math.pi**2*sum(1./length**2 for length in lengths))
    model = eqiora.compile(source=authored, filename="affine-q2-sine.eqi", entry="AffineQ2Sine", geometry=geometry,
                           bindings={"body": geometry.selection("body"), **{name: (geometry.selection(name), geometry.selection("body")) for name in names}, **values})
    field = model.field("definition.u")
    plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q2(), quadrature=eqiora.fem.GaussLegendre(4),
                          solve=eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
                                                   preconditioner=eqiora.solve.Preconditioner.Identity,
                                                   reduction=eqiora.solve.Reduction.Reproducible,
                                                   provider=eqiora.solve.SolverProvider.reference(),
                                                   relative_tolerance=1e-13, absolute_tolerance=1e-14, maximum_iterations=100000))
    require(plan.fields == (field,), "source-selected field differs from actual Plan inventory")
    require(plan.mesh_digest == mesh.digest, "resolve did not retain the actual caller Mesh")
    require(plan.execution.workers == 1 and plan.execution.placement == "host-serial", "unexpected actual execution policy")
    original_plan = plan.to_bytes()
    write(directory/"plan.json", original_plan)
    write(directory/"model.json", model.to_bytes())
    write(directory/"mesh.json", mesh.to_bytes())
    reopened_plan = eqiora.Plan.from_bytes(original_plan)
    require(reopened_plan.to_bytes() == original_plan and reopened_plan.identity == plan.identity, "Plan replay changed canonical effective settings")
    require(tuple(reopened_plan.field_coefficient_entities(field)) == tuple(plan.field_coefficient_entities(field)), "Plan replay changed the complete coefficient entity order")
    # Reuse the complete public preflight before any candidate execution.
    authenticate_plan(reopened_plan, lengths, args.cells)
    state["ordinary_run_attempted"] = 1
    result = eqiora.run(reopened_plan)
    state["ordinary_run_completed"] = 1
    result_bytes = result.to_bytes()
    write(directory/"result.json", result_bytes)
    reopened_result = eqiora.Result.from_bytes(reopened_plan, result_bytes)
    require(reopened_result.to_bytes() == result_bytes, "Result replay changed canonical bytes")
    candidate, identity = extract_candidate(reopened_plan, result, field, lengths, args.cells)
    candidate_sha256 = write(directory/"candidate.json", candidate)
    replayed_candidate, replayed_identity = extract_candidate(reopened_plan, reopened_result, field, lengths, args.cells)
    require(replayed_candidate == candidate and replayed_identity == identity, "Result replay changed a vertex/edge/face/cell coefficient")
    identity["candidate_sha256"] = candidate_sha256
    export_identity_sha256 = write(directory/"export-identity.json", identity)
    native_path = Path(native.__file__).resolve()
    write(directory/"execution.json", {**state, "status": "exported-not-yet-compared", "eqiora_version": importlib.metadata.version("eqiora"),
                                        "native_module": str(native_path), "native_module_sha256": sha256(native_path.read_bytes()),
                                        "python": sys.executable, "python_version": sys.version, "started_at": args.started_at, "finished_at": utc(),
                                        "cpu_affinity": sorted(os.sched_getaffinity(0)), "authored_parameters": values,
                                        "reference_table_sha256": sha256((HERE/"reference"/"intervals-gauss4-audit.json").read_bytes()),
                                        "source_sha256": sha256(Path(__file__).read_bytes()), "production_export_sha256": sha256((HERE/"production.py").read_bytes()),
                                        "shared_quadratic_export_sha256": sha256((HERE.parent/"quadratic_output.py").read_bytes()),
                                        "plan_identity": plan.identity, "result_sha256": sha256(result_bytes), "candidate_sha256": candidate_sha256, "export_identity_sha256": export_identity_sha256,
                                        "pde_solves": 1, "profile_calls": 0, "warmups": 0, "timed_samples": 0,
                                        "plan_reopens": 1, "result_reopens": 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--cells", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist; preserve previous attempts")
    if args.cpu < 0 or os.sched_getaffinity(0) != {args.cpu}:
        parser.error("caller must pin this process to exactly the declared CPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CUDA_VISIBLE_DEVICES must be empty")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        if os.environ.get(name) != "1":
            parser.error(f"{name} must be 1")
    args.started_at = utc()
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"ordinary_run_attempted": 0, "ordinary_run_completed": 0}
    write(args.output/"launch.json", {"schema": 1, "command": sys.argv, "started_at": args.started_at, "dimension": args.dimension,
                                     "N": args.cells, "cpu": args.cpu, "source_sha256": sha256(Path(__file__).read_bytes()),
                                     "production_export_sha256": sha256((HERE/"production.py").read_bytes()),
                                        "shared_quadratic_export_sha256": sha256((HERE.parent/"quadratic_output.py").read_bytes()),
                                     "declared_pde_solves": 1, "profile_calls": 0, "warmups": 0, "timed_samples": 0})
    try:
        run(args, args.output, state)
    except Exception as error:
        write(args.output/"failure.json", {**state, "status": "failed", "type": type(error).__name__, "message": str(error), "finished_at": utc()})
        print(f"q2-production: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
