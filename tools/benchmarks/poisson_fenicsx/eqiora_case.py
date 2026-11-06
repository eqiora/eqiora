"""Ordinary installed Eqiora half of the declared Poisson comparison."""
import time
IMPORT_START = time.perf_counter_ns()
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eqiora
import numpy as np
from shared_measurement import ROOT, ComparisonError, clock, digest, environment, finish, memory
from worker_arguments import worker_arguments
from accuracy import check_accuracy
IMPORT_SECONDS = (time.perf_counter_ns()-IMPORT_START)/1e9


def main(args):
    if not args.wheel or not args.revision or not args.build_profile:
        raise ComparisonError("Eqiora requires wheel path, declared build revision and build profile")
    if Path(eqiora.__file__).resolve().is_relative_to(ROOT):
        raise ComparisonError("use an installed wheel, not repository Python source")
    env = environment("eqiora", IMPORT_SECONDS, protocol_root=Path(__file__).parent)
    env.update(wheel_sha256=digest(args.wheel), source_revision_declared=args.revision,
               rust_build_profile_declared=args.build_profile, module_path=eqiora.__file__,
               blas=np.show_config(mode="dicts"))
    baseline = memory()

    dimension = args.dimension
    region = "square" if dimension == 2 else "cube"

    def mesh_setup():
        graph = eqiora.geometry.GeometryGraph()
        primitive = (graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
                     if dimension == 2 else graph.cuboid(
                         x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), z_bounds=(0.0, 1.0)))
        names = tuple(f"{axis}_{side}" for axis in "xyz"[:dimension] for side in ("lower", "upper"))
        geometry = graph.build(primitive, named_topology={region: primitive.region,
            **dict(zip(names, primitive.boundaries, strict=True))})
        mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
            geometry, eqiora.meshing.CartesianMesher(cells=(args.cells,)*dimension)))
        return geometry, mesh, names

    (geometry, mesh, names), mesh_time = clock(mesh_setup)
    source = (ROOT / "examples/inverse-poisson.eqi" if dimension == 2
              else Path(__file__).with_name("poisson_3d.eqi"))

    def prepare():
        model = eqiora.compile(path=source, geometry=geometry, entry="InversePoisson" if dimension == 2 else "PoissonCube", bindings={
            region: geometry.selection(region),
            **{name: (geometry.selection(name), geometry.selection(region)) for name in names},
            "diffusion": 1.0, "wave_number": np.pi, "source_scale": dimension*np.pi**2, "boundary_offset": 0.0})
        plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
            preconditioner=eqiora.solve.Preconditioner.Identity, reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(), relative_tolerance=1e-12,
            absolute_tolerance=1e-14, maximum_iterations=10000))
        if plan.execution.workers != 1 or plan.execution.placement != "host-serial":
            raise ComparisonError("Eqiora execution differs from host serial one-worker policy")
        return model, plan

    (model, plan), preparation = clock(prepare)
    field = plan.capability.fields[0]
    env["resolved_execution"] = {name: getattr(plan.execution, name) for name in
                                 ("scalar_type", "provider", "provider_version", "placement", "workers")}
    env["linear_provider"] = {"id": plan.solve.provider.id,
                              "implementation_version": plan.solve.provider.implementation_version}

    def extract(result):
        summary = result.solve
        fields = ("algorithm", "completed_iterations", "initial_residual_norm",
                  "reported_residual_norm", "true_residual_norm", "residual_target")
        solve = {name: getattr(summary, name) for name in fields}
        solve["preconditioner"] = summary.preconditioner.kind
        for name in ("backend", "adapter", "verification_adapter", "reduction",
                     "relative_tolerance", "absolute_tolerance", "maximum_iterations"):
            solve[name] = getattr(summary, name)
        solve["convergence_reason"] = str(summary.reason)
        if (solve["algorithm"] != "bicgstab" or solve["preconditioner"] != "identity"
                or solve["relative_tolerance"] != 1e-12 or solve["absolute_tolerance"] != 1e-14
                or solve["maximum_iterations"] != 10000):
            raise ComparisonError("Eqiora solve report differs from the declared exact policy")
        if not np.isfinite(solve["true_residual_norm"]) or solve["true_residual_norm"] > solve["residual_target"]:
            raise ComparisonError("Eqiora true residual exceeds its accepted target")
        return np.asarray(plan.mesh.coordinates), np.asarray(result.output(field).values("vertex")), solve

    def profile():
        result = eqiora.run(plan, profile=True)
        phases = [{"path": list(item.path), "fields": dict(item.fields), "calls": item.calls,
                   "inclusive_seconds": item.inclusive_seconds, "self_seconds": item.self_seconds}
                  for item in result.profile.phases]
        assembly = [x for x in phases if x["path"][-1] == "assembly" and x["fields"].get("role") == "scalar_region_finalization"]
        linear = [x for x in phases if x["path"][-1] == "linear_solve"]
        if args.mode == "correctness-only":
            return {"required_phases_available": bool(assembly and linear),
                    "phase_inventory": [{key: row[key] for key in ("path", "fields", "calls")} for row in phases]}
        return {"source": "separate profile=True diagnostic run; inclusive wall and exclusive self are not unprofiled wall partitions",
                "required_phases_available": bool(assembly and linear), "phases": phases}

    report = {"environment": env, "memory_after_import": baseline, "memory_after_preparation": memory(),
              "mesh_measurement": mesh_time, "preparation_measurement": preparation,
              "model_sha256": digest(source), "plan_sha256": digest_bytes(plan.to_bytes()),
              "package_compilation_digest": model.package_compilation_digest,
              "dofs": {"total": (args.cells+1)**dimension, "free": (args.cells-1)**dimension, "stored_algebraic": (args.cells-1)**dimension}}
    finish(args, "eqiora", report, lambda: eqiora.run(plan), extract, profile,
           accuracy=lambda coordinates, values, cells: check_accuracy(coordinates, values, cells, args.dimension))


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    args = worker_arguments()
    try:
        main(args)
    except (ComparisonError, ImportError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        raise SystemExit(2)
