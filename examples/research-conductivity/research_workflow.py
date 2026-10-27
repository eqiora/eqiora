"""Layered heat conduction: forward, bounded inverse, and reusable evaluation data.

Run beside conductivity.eqi with installed eqiora, numpy and scipy. All equations,
spatial observations and derivative products execute in Eqiora; Python composes
requests, scaled least squares, dataset splits and reporting.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import resource
import shutil
import time
from pathlib import Path

import eqiora as q
import numpy as np

MODEL_SOURCE = Path(__file__).with_name("conductivity.eqi")
SENSOR_NAMES = ("left_mean", "right_mean", "mean_flux")
SENSOR_UNITS = ("K", "K", "W/m^2")
SCALES = np.array([0.02, 0.02, 0.05])
NOISE_SEED = 20261026
BOUNDS = tuple((np.log(0.5), np.log(10.0)) for _ in range(2))


class IncompleteEvaluation(RuntimeError):
    """A failed reduction retains the common exact collection for inspection."""

    def __init__(self, report):
        self.collection = report.record()
        super().__init__("Incomplete multi-load evaluation; inspect exception.collection before retry.")


def log_conductivities(values):
    """Convert positive SI W/(m K) inputs to the dimensionless log coordinates."""
    values = np.asarray(values)
    if (values.shape != (2,) or values.dtype.kind not in "fiu"
            or not np.all(np.isfinite(values)) or np.any(values <= 0)):
        raise ValueError("Supply two finite, positive conductivities in W/(m K).")
    return np.log(values.astype(np.float64))


def prepare(cells=16, solver="direct", *, source=MODEL_SOURCE,
            conductivities=(2.0, 5.0), heating=20.0, right_temperature=310.0):
    """One metre rod, fixed interface at 0.5 m, and ordinary immutable owners."""
    if isinstance(cells, bool) or cells < 2 or cells % 2:
        raise ValueError("Use a positive even cell count so the interface is a mesh node.")
    graph = q.geometry.GeometryGraph()
    rod = graph.interval(bounds=(0.0, 1.0))
    geometry = graph.build(rod, named_topology={
        "body": rod.region, "left": rod.boundaries[0], "right": rod.boundaries[1],
    })
    body = geometry.selection("body")
    model = q.compile(path=source, entry="LayeredRod", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
        "log_k": log_conductivities(conductivities).tolist(),
        "heating": heating, "right_temperature": right_temperature,
    })
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.CartesianMesher(cells=(cells,))))
    if solver == "direct":
        algorithm, provider, reduction = (q.solve.LinearSolver.SparseLu,
            q.solve.SolverProvider.faer(), q.solve.Reduction.Fast)
    elif solver == "iterative":
        algorithm, provider, reduction = (q.solve.LinearSolver.BiConjugateGradientStabilized,
            q.solve.SolverProvider.reference(), q.solve.Reduction.Reproducible)
    else:
        raise ValueError("solver must be direct or iterative")
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=q.solve.Linear(
        algorithm=algorithm, provider=provider, reduction=reduction,
        preconditioner=q.solve.Preconditioner.Identity,
        relative_tolerance=1e-12, absolute_tolerance=1e-12, maximum_iterations=10000))
    observations = tuple(model.observable("definition." + name) for name in SENSOR_NAMES)
    program = q.diff.compile(plan,
        inputs=tuple(model.parameter(name) for name in ("log_k", "heating", "right_temperature")),
        outputs=(plan.fields[0], *observations), quadrature_points=2)
    return model, plan, program


def evaluate_loads(program, log_k, loads):
    """The native ordered map owns evaluation, attribution and partial failure."""
    loads = np.asarray(loads, dtype=np.float64)
    mapped = program.map((np.ascontiguousarray(np.broadcast_to(log_k, (len(loads), 2))),
        *(np.ascontiguousarray(loads[:, i]) for i in range(2))), point_shape=(len(loads),))
    report = mapped.execute()
    if not isinstance(report, q.CompleteEvaluationMap):
        raise IncompleteEvaluation(report)
    return report


def sensor_values(report):
    return np.stack(report.primal()[1:], axis=-1)


def objective_gradient(program, log_k, loads, observed, scales=SCALES):
    report = evaluate_loads(program, log_k, loads)
    residual = (sensor_values(report) - observed) / scales
    seeds = (np.zeros_like(report.primal()[0]),
             *(np.ascontiguousarray(residual[:, i] / scales[i]) for i in range(3)))
    gradient = report.vjp(seeds).mapped_cotangents[0].sum(axis=0)
    return float(np.sum(residual * residual) / 2), np.asarray(gradient)


def fit(program, loads, observed, initial=(3.0, 3.0)):
    """No regularizer: bounds enforce positivity; data must provide rank itself."""
    # SciPy is optional and does not bundle type information.
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    result = least_squares(
        lambda eta: ((sensor_values(evaluate_loads(program, eta, loads)) - observed) / SCALES).ravel(),
        log_conductivities(initial), jac=lambda eta: sensitivity(program, eta, loads),
        method="trf", bounds=np.array(BOUNDS).T, ftol=1e-10, xtol=1e-10, gtol=1e-8,
        max_nfev=100)
    if not result.success:
        raise RuntimeError(f"Bounded fit failed: {result.message}")
    return result


def sensitivity(program, log_k, loads, *, temperatures_only=False):
    report = evaluate_loads(program, log_k, loads)
    columns = []
    for direction in np.eye(2):
        action = report.jvp((np.ascontiguousarray(np.broadcast_to(direction, (len(loads), 2))),
                            np.zeros(len(loads)), np.zeros(len(loads))))
        values = np.stack(action.tangent[1:], axis=-1)
        columns.append((values[:, :2] / SCALES[:2] if temperatures_only else values / SCALES).ravel())
    return np.stack(columns, axis=-1)


def taylor_remainders(program, log_k, loads, observed, *, wrong_gradient=False):
    direction = np.array([0.3, -0.2])
    base, gradient = objective_gradient(program, log_k, loads, observed)
    if wrong_gradient:
        gradient = -gradient  # Deliberately wrong adjoint sign; it must fail the Taylor check.
    steps = np.array([0.04, 0.02, 0.01, 0.005])
    errors = []
    for step in steps:
        value, _ = objective_gradient(program, log_k + step * direction, loads, observed)
        errors.append(abs(value - base - step * np.dot(gradient, direction)))
    return steps, np.array(errors)


def write_dataset(program, reference, output):
    # A whole physical parameter case stays in one split across every load.
    cases = [("train", 1.5, 4.0), ("train", 2.0, 5.0), ("train", 2.5, 6.0),
             ("validation", 1.75, 5.5), ("test", 1.0, 7.0), ("test", 3.0, 3.0)]
    loads = [("calibration-a", 20.0, 310.0), ("calibration-b", 40.0, 302.0),
             ("withheld", 15.0, 307.0)]
    points, labels = [], []
    for case_index, (split, left, right) in enumerate(cases):
        for load_name, heating, temperature in loads:
            points.append((np.log(left), np.log(right), heating, temperature))
            labels.append((f"case-{case_index}/{load_name}", f"case-{case_index}", split, load_name))
    points = np.asarray(points)
    mapped = program.map((np.ascontiguousarray(points[:, :2]),
        np.ascontiguousarray(points[:, 2]), np.ascontiguousarray(points[:, 3])),
        point_shape=(len(points),))
    collection = mapped.execute_recorded()
    collection.write(output / "dataset.eqcollection")
    targets = np.full((len(points), 3), np.nan)
    fields = np.full((len(points), *program.outputs[0].shape), np.nan)
    accepted = np.array([member.status == "accepted" for member in collection[0]])
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "parameter_case", "split", "load", "seed",
            "log_k_left", "log_k_right", "heating_W_per_m3", "right_temperature_K",
            "occurrence_id", "status", "result_reference", "diagnostic"])
        for i, (member, label) in enumerate(zip(collection[0], labels, strict=True)):
            writer.writerow([*label, "none (explicit grid)", *points[i], member.occurrence_id,
                member.status, member.reference,
                "; ".join(d.code + ": " + d.message for d in member.diagnostics)])
            if member.status == "accepted":
                fields[i] = member.result.output[0]
                targets[i] = [float(value) for value in member.result.output[1:]]
    if not collection.complete:
        raise RuntimeError("Dataset has unsuccessful members; inspect its saved collection and CSV before reducing.")
    train = np.array([label[2] == "train" for label in labels]) & accepted
    if not train.any():
        raise RuntimeError("No accepted training cases; inspect dataset.eqcollection.")
    # Withheld loads are kept separate from training statistics as well as evaluation.
    train &= np.array([label[3] != "withheld" for label in labels])
    np.savez(output / "targets.npz", sample_ids=np.array([label[0] for label in labels]),
        accepted=accepted, temperature_K=fields, observations=targets,
        observation_units=np.array(SENSOR_UNITS),
        training_mean=targets[train].mean(axis=0), training_std=targets[train].std(axis=0))
    test_indices = np.array([i for i, label in enumerate(labels) if label[2] == "test"])
    selected = points[test_indices]
    heldout = reference.map((np.ascontiguousarray(selected[:, :2]),
        np.ascontiguousarray(selected[:, 2]), np.ascontiguousarray(selected[:, 3])),
        point_shape=(len(selected),)).execute_recorded()
    heldout.write(output / "parameter-holdout-reference.eqcollection")
    if not heldout.complete:
        raise RuntimeError("Held-out reference failed; inspect its saved collection.")
    reference_targets = np.array([[float(value) for value in member.result.output[1:]]
                                  for member in heldout[0]])
    errors = targets[test_indices] - reference_targets
    np.savez(output / "parameter-holdout-errors.npz",
        sample_ids=np.array([labels[i][0] for i in test_indices]),
        errors=errors, units=np.array(SENSOR_UNITS))
    return collection, np.max(np.abs(errors), axis=0)


def failure_and_retry(model, plan, output):
    """An ordinary ordered study and explicit retries retain accepted history."""
    parameters = tuple(model.parameter(name) for name in
        ("log_k", "heating", "right_temperature"))
    baseline = tuple(parameter.value for parameter in parameters)
    changed = (log_conductivities((1.5, 4.5)).tolist(), 40.0, 302.0)
    # exp(1000) cannot produce a finite conductivity. Keep the failed occurrence
    # and its unstarted sibling; neither is an accepted observation or a dataset target.
    invalid = ([1000.0, 0.0], baseline[1], baseline[2])
    points = [baseline, changed, baseline, invalid, changed]
    first = q.EvaluationMapPlan.from_parameters(plan, parameters, points).execute_recorded()
    first.write(output / "interrupted.eqcollection")
    # Offline reopening preserves accepted history before a fresh request is made.
    first = q.EvaluationCollection.read(output / "interrupted.eqcollection")
    group = first[0]
    unsuccessful = [member.point for member in group if member.status != "accepted"]
    # An exact retry of the invalid log input still fails; it must not be reported as success.
    exact = q.EvaluationMapPlan.from_parameters(
        group.plan, group.parameters, unsuccessful).execute_recorded()
    # Correction is an explicitly changed request, with new reference and occurrence identities.
    corrected = [(baseline[0], *unsuccessful[0][1:]), *unsuccessful[1:]]
    amended = q.EvaluationMapPlan.from_parameters(
        group.plan, group.parameters, corrected).execute_recorded()
    history = q.EvaluationCollection.combine([first, exact, amended])
    history.write(output / "retry-history.eqcollection")
    return history


def _study_summary(group):
    """Project the saved ordinary Results for user inspection; no solve is performed."""
    return {"parameter_ids": [parameter.id for parameter in group.parameters],
        "members": [{"point": [np.asarray(value).tolist() for value in member.point],
            "status": member.status,
            "diagnostics": [d.code + ": " + d.message for d in member.diagnostics],
            "plan_identity": member.plan.identity if member.status == "accepted" else None,
            "temperature_K": member.result.output(member.plan.fields[0]).values("vertex").numpy().tolist()
                if member.status == "accepted" else None} for member in group]}


def reopen(output):
    """Inspect exact accepted artifacts without compilation or a solver invocation."""
    plan = q.Plan.from_bytes((output / "plan.json").read_bytes())
    result = q.Result.from_bytes(plan, (output / "result.json").read_bytes())
    dataset = q.EvaluationCollection.read(output / "dataset.eqcollection")
    history = q.EvaluationCollection.read(output / "retry-history.eqcollection")
    return {"plan_identity": plan.identity, "result_bytes": len(result.to_bytes()),
        "dataset_members": len(dataset[0]), "dataset_complete": dataset.complete,
        "history": [[member.status for member in group] for group in history],
        "ordered_parameter_study": _study_summary(history[0])}


def run(output, cells=16, solver="direct", conductivities=(2.0, 5.0),
        heating=20.0, right_temperature=310.0):
    truth = log_conductivities(conductivities)
    if any(value < lower or value > upper for value, (lower, upper) in zip(truth, BOUNDS, strict=True)):
        raise ValueError("The inverse profile requires conductivities between 0.5 and 10 W/(m K).")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    loads = np.array([[heating, right_temperature], [2 * heating, 302.0]])
    withheld = np.array([[15.0, 307.0]])
    timings = {}
    start = time.perf_counter()
    model, plan, program = prepare(cells, solver, conductivities=conductivities,
        heating=heating, right_temperature=right_temperature)
    timings["preparation_s"] = time.perf_counter() - start
    (output / "model.json").write_bytes(model.to_bytes())
    (output / "plan.json").write_bytes(plan.to_bytes())
    # Save one ordinary Result for labeled physical field and observation inspection.
    start = time.perf_counter()
    result = q.run(plan)
    baseline = [result.observe(model.observable("definition." + name), quadrature_points=2).value
                for name in SENSOR_NAMES]
    (output / "result.json").write_bytes(result.to_bytes())
    timings["forward_and_observation_s"] = time.perf_counter() - start
    # Distinct fine-grid data, never the same inverse operator's outputs as observations.
    _, reference_plan, reference = prepare(cells * 4, solver, conductivities=conductivities,
        heating=heating, right_temperature=right_temperature)
    (output / "reference-plan.json").write_bytes(reference_plan.to_bytes())
    reference_report = evaluate_loads(reference, truth, loads)
    reference_report.record().write(output / "reference-observations.eqcollection")
    observed = sensor_values(reference_report)
    noisy = observed + np.random.default_rng(NOISE_SEED).normal(size=observed.shape) * SCALES
    np.savez(output / "measurements.npz", noise_free=observed, noisy=noisy,
        scales=SCALES, units=np.array(SENSOR_UNITS), seed=NOISE_SEED, loads=loads,
        log_conductivities=truth)
    start = time.perf_counter()
    probe_cost, probe_gradient = objective_gradient(program, np.log([3.0, 3.0]), loads, observed)
    timings["objective_and_gradient_s"] = time.perf_counter() - start
    clean_fit, noisy_fit = fit(program, loads, observed), fit(program, loads, noisy)
    start = time.perf_counter()
    fitted = evaluate_loads(program, clean_fit.x, loads)
    timings["multi_load_s"] = time.perf_counter() - start
    fitted.record().write(output / "fitted.eqcollection")
    prediction = evaluate_loads(program, clean_fit.x, withheld)
    prediction.record().write(output / "heldout.eqcollection")
    heldout_reference = evaluate_loads(reference, truth, withheld)
    heldout_reference.record().write(output / "heldout-reference.eqcollection")
    start = time.perf_counter()
    dataset, parameter_holdout_error = write_dataset(program, reference, output)
    timings["dataset_s"] = time.perf_counter() - start
    history = failure_and_retry(model, plan, output)
    steps, errors = taylor_remainders(program, np.log([3.0, 3.0]), loads, observed)
    _, wrong = taylor_remainders(program, np.log([3.0, 3.0]), loads, observed, wrong_gradient=True)
    singular_values = np.linalg.svd(sensitivity(program, truth, loads), compute_uv=False)
    deficient = np.linalg.svd(sensitivity(program, truth, np.array([[0.0, 310.0]]),
        temperatures_only=True), compute_uv=False)
    # Examine the declared bounded parameter box, not only the recovered point.
    domain_smin = [float(np.linalg.svd(sensitivity(program, np.log([left, right]), loads),
        compute_uv=False)[-1]) for left in (0.5, 2.0, 10.0) for right in (0.5, 5.0, 10.0)]
    evidence = fitted[0].primal().evidence.steps[0].primal_solves[0]
    summary = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("eqiora", "numpy", "scipy")}},
        "model_digest": model.digest, "plan_identity": plan.identity,
        "mesh_digest": program.outputs[0].mesh_digest,
        "parameter_ids": [item.id for item in program.inputs],
        "observation_ids": [item.id for item in program.outputs[1:]],
        "calibration_loads_W_per_m3_and_K": loads.tolist(),
        "withheld_loads_W_per_m3_and_K": withheld.tolist(),
        "reference_plan_identity": reference_plan.identity,
        "cells": cells, "reference_cells": cells * 4, "solver": solver,
        "reference_temperature_mean_error_bound_K": float(np.max(np.abs(loads[:, 0]))
            / (12 * (cells * 4)**2 * np.min(conductivities))),
        "sensor_names": SENSOR_NAMES, "sensor_units": SENSOR_UNITS,
        "measurement_scales": SCALES.tolist(), "noise_model": "independent Gaussian, fixed seed",
        "noise_seed": NOISE_SEED, "baseline_observations": baseline,
        "true_conductivities_W_per_m_K": np.exp(truth).tolist(),
        "clean_fit_W_per_m_K": np.exp(clean_fit.x).tolist(),
        "noisy_fit_W_per_m_K": np.exp(noisy_fit.x).tolist(),
        "clean_parameter_relative_error": (np.exp(clean_fit.x - truth) - 1).tolist(),
        "noisy_parameter_relative_error": (np.exp(noisy_fit.x - truth) - 1).tolist(),
        "probe_scaled_objective": probe_cost,
        "probe_gradient_wrt_log_conductivities": probe_gradient.tolist(),
        "clean_scaled_objective": float(clean_fit.cost), "noisy_scaled_objective": float(noisy_fit.cost),
        "optimizer_function_evaluations": [int(clean_fit.nfev), int(noisy_fit.nfev)],
        "singular_values_at_truth": singular_values.tolist(),
        "sampled_domain_minimum_singular_value": min(domain_smin),
        "domain_rank_nonclaim": "nine sampled points do not certify rank throughout the box",
        "deficient_temperature_only_singular_values": deficient.tolist(),
        "calibration_observation_error": (sensor_values(fitted) - observed).tolist(),
        "heldout_observation_error": (sensor_values(prediction) - sensor_values(heldout_reference)).tolist(),
        "heldout_parameter_region_max_abs_observation_error": parameter_holdout_error.tolist(),
        "taylor_steps": steps.tolist(), "taylor_remainders": errors.tolist(),
        "wrong_gradient_remainders": wrong.tolist(),
        "solver_true_residual_norm": evidence.true_residual_norm,
        "solver_residual_target": evidence.residual_target,
        "dataset_members": len(dataset[0]),
        "ordered_parameter_study": _study_summary(history[0]),
        "retry_history": [[member.status for member in group] for group in history],
        "timings": timings, "peak_process_rss_KiB_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copyfile(MODEL_SOURCE, output / MODEL_SOURCE.name)
    shutil.copyfile(__file__, output / "research_workflow.py")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reopen", action="store_true")
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--solver", choices=("direct", "iterative"), default="direct")
    parser.add_argument("--conductivities", type=float, nargs=2, default=(2.0, 5.0),
                        metavar=("LEFT_W_PER_M_K", "RIGHT_W_PER_M_K"))
    parser.add_argument("--heating", type=float, default=20.0, help="known source in W/m^3")
    parser.add_argument("--right-temperature", type=float, default=310.0, help="K")
    args = parser.parse_args()
    if args.reopen:
        value = reopen(args.output)
    else:
        value = run(args.output, args.cells, args.solver, args.conductivities,
                    args.heating, args.right_temperature)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
