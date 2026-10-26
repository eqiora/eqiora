"""Layered-rod research source: independent ODE, inverse and artifact checks."""
from __future__ import annotations

import csv
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import eqiora as q
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[3] / "examples/research-conductivity"


@pytest.fixture(scope="module")
def workflow():
    # The tutorial and installed-user checks execute the same source.
    return runpy.run_path(str(SOURCE / "research_workflow.py"))


def continuum_temperature(x, conductivity, heating, right_temperature):
    """Integrate j'=heating and T'=-j/k, imposing both boundary values.

    R=integral(1/k)=1/(2*kL)+1/(2*kR), A=integral(x/k)=1/(8*kL)+3/(8*kR).
    Thus j(x)=heating*x+C, C=-(right_temperature-300+heating*A)/R.
    This reference never calls an Eqiora solver or derivative product.
    """
    left, right = conductivity
    resistance = 0.5 / left + 0.5 / right
    first_moment = 0.125 / left + 0.375 / right
    constant = -(right_temperature - 300 + heating * first_moment) / resistance
    x = np.asarray(x)
    lower = np.minimum(x, 0.5)
    temperature = (300 - (heating * lower**2 / 2 + constant * lower) / left
        - np.where(x > 0.5, (heating * (x**2 - 0.25) / 2 + constant * (x - 0.5)) / right, 0))
    return temperature, constant


def continuum_observations(conductivity, heating, right_temperature):
    # Two-point Gauss integrates each quadratic half-rod exactly. These are
    # continuum quadratics, not samples of the computed discrete Field.
    z = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    left, constant = continuum_temperature((z + 1) / 4, conductivity, heating, right_temperature)
    right, _ = continuum_temperature((z + 3) / 4, conductivity, heating, right_temperature)
    return np.array([left.mean(), right.mean(), heating / 2 + constant])


@pytest.mark.parametrize("conductivity,heating,boundary", [((2.0, 5.0), 40.0, 302.0),
    ((0.5, 10.0), 20.0, 310.0), ((3.0, 3.0), -10.0, 307.0)])
def test_forward_reference_spatial_error_and_solver_acceptance(workflow, conductivity, heating, boundary):
    expected = continuum_observations(conductivity, heating, boundary)
    errors = []
    for cells in (8, 16, 32):
        _, _, program = workflow["prepare"](cells)
        evaluation = program.evaluate((np.log(conductivity), heating, boundary))
        primal = evaluation.primal()
        exact, _ = continuum_temperature(np.linspace(0, 1, cells + 1), conductivity, heating, boundary)
        np.testing.assert_allclose(primal.output[0], exact, rtol=0, atol=2e-9)
        observed = np.array([float(value) for value in primal.output[1:]])
        # The Q1 interpolant of T''=-q/k has regional mean error -q*h²/(12*k).
        expected_error = np.array([-heating / (12 * cells**2 * k) for k in conductivity] + [0.0])
        np.testing.assert_allclose(observed - expected, expected_error, rtol=0, atol=2e-9)
        errors.append(np.abs(observed[:2] - expected[:2]))
        evidence = primal.evidence.steps[0].primal_solves[0]
        assert evidence.true_residual_norm <= evidence.residual_target
        assert program.outputs[1].value_type == q.ValueType.real(q.Dimension(temperature=1))
        assert program.outputs[3].value_type == q.ValueType.real(q.Dimension(mass=1, time=-3))
    np.testing.assert_allclose(np.array(errors[:-1]) / np.array(errors[1:]), 4, rtol=0, atol=3e-5)


def test_identifiability_gradient_taylor_and_wrong_gradient_falsifier(workflow):
    _, _, program = workflow["prepare"]()
    _, _, reference = workflow["prepare"](64)
    loads = np.array([[20.0, 310.0], [40.0, 302.0]])
    truth, point = np.log([2.0, 5.0]), np.log([3.0, 3.0])
    observed = workflow["sensor_values"](workflow["evaluate_loads"](reference, truth, loads))
    matrix = workflow["sensitivity"](program, truth, loads)
    expected_columns = []
    for index, direction in enumerate(np.eye(2)):
        # Complex-step differentiates the independent closed-form ODE reference.
        # Include the independently derived Q1 regional-mean interpolation bias.
        values = []
        for heating, boundary in loads:
            derivative = continuum_observations(np.exp(truth + 1j * 1e-30 * direction),
                heating, boundary).imag / 1e-30
            derivative[index] += heating / (12 * 16**2 * np.exp(truth[index]))
            values.append(derivative / workflow["SCALES"])
        expected_columns.append(np.asarray(values).ravel())
    expected_matrix = np.stack(expected_columns, axis=-1)
    np.testing.assert_allclose(matrix, expected_matrix, rtol=0, atol=2e-7)
    assert np.linalg.matrix_rank(expected_matrix) == 2
    # No source and temperature-only data admit simultaneous scaling of both k's.
    insufficient = workflow["sensitivity"](program, truth, np.array([[0.0, 310.0]]),
                                               temperatures_only=True)
    np.testing.assert_allclose(insufficient @ np.ones(2), 0, rtol=0, atol=1e-8)
    assert np.linalg.svd(insufficient, compute_uv=False)[-1] < 1e-8
    _, gradient = workflow["objective_gradient"](program, point, loads, observed)
    step = 1e-5
    finite_difference = []
    for direction in np.eye(2):
        plus = workflow["objective_gradient"](program, point + step * direction, loads, observed)[0]
        minus = workflow["objective_gradient"](program, point - step * direction, loads, observed)[0]
        finite_difference.append((plus - minus) / (2 * step))
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-7, atol=2e-4)
    _, correct = workflow["taylor_remainders"](program, point, loads, observed)
    _, wrong = workflow["taylor_remainders"](program, point, loads, observed, wrong_gradient=True)
    assert np.all(correct[:-1] / correct[1:] > 3.8)
    assert np.all((wrong[:-1] / wrong[1:] > 1.9) & (wrong[:-1] / wrong[1:] < 2.1))


@pytest.mark.parametrize("conductivity,loads", [
    ((2.0, 5.0), [[20.0, 310.0], [40.0, 302.0]]),
    ((2.5, 4.0), [[25.0, 308.0], [50.0, 302.0]]),
])
def test_bounded_recovery_uses_distinct_reference_discretization(workflow, conductivity, loads):
    pytest.importorskip("scipy")
    _, inverse_plan, program = workflow["prepare"]()
    _, reference_plan, reference = workflow["prepare"](64)
    assert inverse_plan.identity != reference_plan.identity
    loads = np.asarray(loads)
    truth = np.log(conductivity)
    observed = workflow["sensor_values"](workflow["evaluate_loads"](reference, truth, loads))
    # Reference mean error from the independently derived h² formula, < 0.5 mK.
    assert np.max(np.abs(loads[:, 0])) / (12 * 64**2 * min(conductivity)) < 0.0005
    for noisy, tolerance in ((False, 0.005), (True, 0.02)):
        values = observed.copy()
        if noisy:
            values += np.random.default_rng(workflow["NOISE_SEED"]).normal(size=values.shape) * workflow["SCALES"]
        fit = workflow["fit"](program, loads, values)
        assert fit.success
        assert np.max(np.abs(np.exp(fit.x - truth) - 1)) < tolerance
        # Prediction uses a load that the optimizer never sees.
        heldout = np.array([[15.0, 307.0]])
        prediction = workflow["sensor_values"](workflow["evaluate_loads"](program, fit.x, heldout))
        reference_value = workflow["sensor_values"](workflow["evaluate_loads"](reference, truth, heldout))
        assert np.max(np.abs(prediction - reference_value) / workflow["SCALES"]) < (1 if noisy else 0.2)


def test_compatible_solver_swap_keeps_discrete_problem(workflow, tmp_path):
    cells = 8
    coordinates = np.linspace(0, 1, cells + 1)
    nominal = np.array([*np.log([2.0, 5.0]), 40.0, 302.0])
    direction = np.array([0.3, -0.2, 0.7, -0.4])
    cotangent = (np.linspace(0.25, 1.25, cells + 1), 0.7, -0.4, 0.2)

    def flat(values):
        return np.concatenate([np.asarray(value).ravel() for value in values])

    def leaves(values):
        return (np.ascontiguousarray(values[:2]), values[2], values[3])

    def reference(values):
        conductivity = np.exp(values[:2])
        field, _ = continuum_temperature(coordinates, conductivity, values[2], values[3])
        observations = continuum_observations(conductivity, values[2], values[3])
        # Exact Q1 regional-mean bias, independently derived above; flux is exact.
        observations[:2] -= values[2] / (12 * cells**2 * conductivity)
        return (field, *observations)

    def jacobian(values):
        return np.stack([flat(reference(values + 1e-30j * unit)).imag / 1e-30
                         for unit in np.eye(4)], axis=-1)

    def accepted(summary, provider, orientation="normal"):
        for name, expected in provider.items():
            assert getattr(summary, name) == expected
        if orientation == "reverse":
            # Both orientations are the transpose of this real operator.
            assert summary.orientation in ("transposed", "conjugate-transposed")
        else:
            assert summary.orientation == orientation
        assert summary.true_residual_norm <= summary.residual_target

    providers = [
        dict(backend="eqiora.faer", algorithm="sparse-lu", reduction="fast",
             preconditioner="identity", adapter="eqiora.host.serial", verification_adapter="eqiora.host.serial"),
        dict(backend="eqiora.reference", algorithm="bicgstab", reduction="reproducible",
             preconditioner="identity", adapter="eqiora.host.serial", verification_adapter="eqiora.host.serial"),
    ]
    points = np.array([nominal, [*np.log([1.5, 4.5]), 20.0, 310.0], nominal])
    directions = np.array([direction, 2 * direction, -direction])
    weights = tuple(np.ascontiguousarray(np.array([1.0, 0.5, -0.25]).reshape(
        (3,) + (1,) * np.ndim(value)) * np.asarray(value)) for value in cotangent)
    outputs, plans, collections = [], [], []
    for solver, provider in zip(("direct", "iterative"), providers, strict=True):
        model, plan, program = workflow["prepare"](cells, solver, heating=40.0, right_temperature=302.0)
        plans.append(plan)
        assert model.digest == plans[0].model.digest
        assert model.to_bytes() == plans[0].model.to_bytes()
        assert plan.mesh.digest == plans[0].mesh.digest
        requested_provider = (q.solve.SolverProvider.faer() if solver == "direct"
                              else q.solve.SolverProvider.reference())
        assert plan.solve.provider == requested_provider  # Includes implementation and library versions.
        for name in ("backend", "algorithm", "preconditioner", "reduction"):
            assert getattr(plan.solve, name) == provider[name]

        ordinary = q.run(plan)
        accepted(ordinary.solve, provider)
        ordinary_outputs = (ordinary.output(plan.fields[0]).values("vertex").numpy(),
            *(ordinary.observe(model.observable("definition." + name), quadrature_points=2).value
              for name in workflow["SENSOR_NAMES"]))
        np.testing.assert_allclose(flat(ordinary_outputs), flat(reference(nominal)), rtol=0, atol=2e-9)
        (tmp_path / (solver + ".eqresult")).write_bytes(ordinary.to_bytes())

        point = program.evaluate(leaves(nominal))
        primal, forward = point.primal(), point.jvp(leaves(direction))
        reverse = point.vjp(cotangent)
        accepted(primal.evidence.steps[0].primal_solves[0], provider)
        accepted(forward.evidence.steps[0].derivative_solve, provider)
        accepted(reverse.evidence.steps[0].derivative_solve, provider, "reverse")
        np.testing.assert_allclose(flat(primal.output), flat(reference(nominal)), rtol=0, atol=2e-9)
        np.testing.assert_allclose(flat(forward.tangent), jacobian(nominal) @ direction, rtol=0, atol=2e-7)
        np.testing.assert_allclose(flat(reverse.input_cotangent), jacobian(nominal).T @ flat(cotangent),
                                   rtol=0, atol=2e-7)
        np.testing.assert_allclose(np.dot(flat(cotangent), flat(forward.tangent)),
            np.dot(direction, flat(reverse.input_cotangent)), rtol=0, atol=2e-7)
        outputs.append((primal.output, forward.tangent, reverse.input_cotangent))

        mapped = program.map((np.ascontiguousarray(points[:, :2]),
            np.ascontiguousarray(points[:, 2]), np.ascontiguousarray(points[:, 3])), point_shape=(3,)).execute()
        assert isinstance(mapped, q.CompleteEvaluationMap)
        mapped_forward = mapped.jvp((np.ascontiguousarray(directions[:, :2]),
            np.ascontiguousarray(directions[:, 2]), np.ascontiguousarray(directions[:, 3])))
        mapped_reverse = mapped.vjp(weights)
        for index, values in enumerate(points):
            expected = jacobian(values)
            np.testing.assert_allclose(flat(tuple(value[index] for value in mapped.primal())),
                flat(reference(values)), rtol=0, atol=2e-9)
            np.testing.assert_allclose(flat(tuple(value[index] for value in mapped_forward.tangent)),
                expected @ directions[index], rtol=0, atol=2e-7)
            np.testing.assert_allclose(flat(tuple(value[index] for value in mapped_reverse.mapped_cotangents)),
                expected.T @ flat(tuple(value[index] for value in weights)), rtol=0, atol=2e-7)
            accepted(mapped[index].primal().evidence.steps[0].primal_solves[0], provider)
            accepted(mapped_forward.member(index).evidence.steps[0].derivative_solve, provider)
            accepted(mapped_reverse.member(index).evidence.steps[0].derivative_solve, provider, "reverse")
        np.testing.assert_allclose(sum(np.sum(seed * tangent) for seed, tangent in
            zip(weights, mapped_forward.tangent, strict=True)), sum(np.sum(seed * gradient) for seed, gradient in
            zip((directions[:, :2], directions[:, 2], directions[:, 3]),
                mapped_reverse.mapped_cotangents, strict=True)), rtol=0, atol=2e-7)
        collections.append(mapped.record())

    assert plans[0].identity != plans[1].identity
    for direct, iterative in zip(outputs[0], outputs[1], strict=True):
        np.testing.assert_allclose(flat(direct), flat(iterative), rtol=0, atol=2e-7)
    saved = tmp_path / "solver-comparison.eqcollection"
    q.EvaluationCollection.combine(collections).write(saved)
    script = """
import json, pathlib, sys
import eqiora as q
import numpy as np
def forbidden(*args, **kwargs):
    raise AssertionError('offline solver comparison invoked execution')
q.run = q.submit = q.resolve = q.compile = q.diff.compile = forbidden
path = pathlib.Path(sys.argv[1])
providers, identities, expected = json.loads(sys.argv[2])
collection = q.EvaluationCollection.read(path)
assert collection.to_bytes() == path.read_bytes() and collection.complete and len(collection) == 2
for index, (group, provider) in enumerate(zip(collection, providers, strict=True)):
    assert group.plan.identity == identities[index]
    requested_provider = q.solve.SolverProvider.faer() if index == 0 else q.solve.SolverProvider.reference()
    assert group.plan.solve.provider == requested_provider
    assert group[0].reference == group[2].reference
    assert group[0].occurrence_id != group[2].occurrence_id
    result_path = path.parent / (('direct' if index == 0 else 'iterative') + '.eqresult')
    result = q.Result.from_bytes(group.plan, result_path.read_bytes())
    for key, value in provider.items():
        assert getattr(result.solve, key) == value
    field = result.output(group.plan.fields[0]).values('vertex').numpy()
    np.testing.assert_allclose(field, expected[0][0], rtol=0, atol=2e-9)
    for signature, value in zip(group.outputs[1:], expected[0][1:], strict=True):
        actual = result.observe(group.plan.model.observable(signature.id), quadrature_points=2)
        np.testing.assert_allclose(actual.value, value, rtol=0, atol=2e-9)
    for member, wanted in zip(group, expected, strict=True):
        assert member.status == 'accepted' and not hasattr(member.result, 'jvp')
        assert member.result.evidence.plan_identity == group.plan.identity
        for key, value in provider.items():
            assert getattr(member.result.evidence.steps[0].primal_solves[0], key) == value
        for actual, target in zip(member.result.output, wanted, strict=True):
            np.testing.assert_allclose(actual, target, rtol=0, atol=2e-9)
"""
    expected = [[np.asarray(value).tolist() for value in reference(values)] for values in points]
    subprocess.run([sys.executable, "-I", "-c", script, str(saved),
        json.dumps([providers, [plan.identity for plan in plans], expected])],
        cwd=tmp_path, check=True, timeout=120)


def test_dataset_splits_normalization_identity_and_failed_history(workflow, tmp_path):
    _, _, program = workflow["prepare"](8)
    _, _, reference = workflow["prepare"](32)
    collection, heldout_error = workflow["write_dataset"](program, reference, tmp_path)
    rows = list(csv.DictReader((tmp_path / "samples.csv").open()))
    assert len(rows) == len(collection[0]) == 18
    assert len({row["sample_id"] for row in rows}) == 18
    cases = {}
    for row, member in zip(rows, collection[0], strict=True):
        cases.setdefault(row["parameter_case"], set()).add(row["split"])
        assert row["occurrence_id"] == member.occurrence_id
        assert row["status"] == "accepted"
        assert row["result_reference"] == member.reference
    assert all(len(splits) == 1 for splits in cases.values())
    data = np.load(tmp_path / "targets.npz")
    mask = np.array([row["split"] == "train" and row["load"] != "withheld" for row in rows])
    np.testing.assert_array_equal(data["training_mean"], data["observations"][mask].mean(axis=0))
    np.testing.assert_array_equal(data["training_std"], data["observations"][mask].std(axis=0))
    assert np.max(heldout_error[:2]) < 0.06  # Independent q*h²/(12*min(k)) bound, n=8.
    assert heldout_error[2] < 1e-8
    statuses = workflow["failure_and_retry"](program, tmp_path)
    assert statuses == [["accepted", "failed", "not_started"],
                        ["failed", "not_started"], ["accepted", "accepted"]]
    original = q.EvaluationCollection.read(tmp_path / "interrupted.eqcollection")
    history = q.EvaluationCollection.read(tmp_path / "retry-history.eqcollection")
    for retained, before in zip(history[0][0].result.output, original[0][0].result.output, strict=True):
        np.testing.assert_array_equal(retained, before)
    assert history[0][0].reference == original[0][0].reference
    assert history[0][0].occurrence_id == original[0][0].occurrence_id
    assert history[0][1].diagnostics[0].code == "EQ0505"
    assert history[2][1].reference == history[0][0].reference
    for retried, requested in zip(history[2][1].point, history[0][2].point, strict=True):
        np.testing.assert_array_equal(retried, requested)
    assert history[2][1].occurrence_id != history[0][2].occurrence_id
    # Common collection ownership rejects stale/reordered sample attribution.
    wire = json.loads(collection.to_bytes())
    wire["groups"][0]["members"].reverse()
    with pytest.raises(q.ValidationError, match="attribution"):
        q.EvaluationCollection.from_bytes(json.dumps(wire).encode())


def test_invalid_coefficients_units_and_stale_observations(workflow, tmp_path):
    for conductivity in ((0, 1), (-1, 1), (np.inf, 1), (np.nan, 1),
                         np.array([1 + 2j, 1]), np.array([True, True])):
        with pytest.raises(ValueError, match="positive conductivities"):
            workflow["log_conductivities"](conductivity)
    bad_source = tmp_path / "wrong_units.eqi"
    bad_source.write_text((SOURCE / "conductivity.eqi").read_text().replace("parameter heating:W/m^3", "parameter heating:K"))
    with pytest.raises(q.ValidationError):
        workflow["prepare"](8, source=bad_source)
    model, plan, _ = workflow["prepare"](8)
    changed, _, _ = workflow["prepare"](8, heating=21.0)
    result = q.run(plan)
    with pytest.raises(ValueError, match="different exact Model artifact"):
        result.observe(changed.observable("definition.left_mean"), quadrature_points=2)
    with pytest.raises(q.ValidationError):
        result.observe_at(model.observable("definition.profile"), [(0.25, q.Dimension(time=1))])
    _, _, program = workflow["prepare"](8)
    with pytest.raises(workflow["IncompleteEvaluation"]) as failed:
        workflow["objective_gradient"](program, np.array([1000.0, 0.0]),
            np.array([[20.0, 310.0]]), np.zeros((1, 3)))
    assert failed.value.collection[0][0].status == "failed"
    assert failed.value.collection[0][0].diagnostics[0].code == "EQ0505"


def test_installed_research_sources_match_the_canonical_pair():
    from importlib.resources import files

    installed = files("eqiora").joinpath("examples")
    assert installed.joinpath("research_conductivity.py").read_bytes() == (SOURCE / "research_workflow.py").read_bytes()
    assert installed.joinpath("conductivity.eqi").read_bytes() == (SOURCE / "conductivity.eqi").read_bytes()


def test_installed_research_module_runs_outside_checkout(tmp_path):
    pytest.importorskip("scipy")
    output = tmp_path / "accepted"
    command = [sys.executable, "-I", "-m", "eqiora.examples.research_conductivity",
               "--output", str(output)]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                               check=True, timeout=120)
    summary = json.loads(completed.stdout)
    assert summary["dataset_members"] == 18
    assert len(list(tmp_path.iterdir())) == 1  # No downloaded or repository-owned source fixtures.
    reopened = subprocess.run([*command, "--reopen"], cwd=tmp_path,
                              capture_output=True, text=True, check=True, timeout=120)
    retained = json.loads(reopened.stdout)
    assert retained["plan_identity"] == summary["plan_identity"]
    assert retained["dataset_complete"] and retained["dataset_members"] == 18


@pytest.mark.parametrize("solver", ["direct", "iterative"])
def test_downloaded_source_and_fresh_process_offline_reopening(workflow, tmp_path, solver):
    pytest.importorskip("scipy")
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    for name in ("conductivity.eqi", "research_workflow.py"):
        shutil.copyfile(SOURCE / name, downloaded / name)
    output = tmp_path / "accepted"
    completed = subprocess.run([sys.executable, "-I", str(downloaded / "research_workflow.py"),
        "--output", str(output), "--solver", solver], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=120)
    summary = json.loads(completed.stdout)
    assert summary["dataset_members"] == 18
    assert np.asarray(summary["probe_gradient_wrt_log_conductivities"]).shape == (2,)
    assert summary["probe_scaled_objective"] > summary["clean_scaled_objective"]
    assert summary["reference_temperature_mean_error_bound_K"] < 0.0005
    # Disable live entry points: offline reopening must use the persisted owners.
    script = """
import json, pathlib, runpy, sys
import eqiora as q
def forbidden(*args, **kwargs):
    raise AssertionError('offline opening invoked a solver/compiler')
q.run = q.submit = q.compile = q.resolve = q.diff.compile = forbidden
source = runpy.run_path(sys.argv[1])
print(json.dumps(source['reopen'](pathlib.Path(sys.argv[2]))))
"""
    reopened = subprocess.run([sys.executable, "-I", "-c", script,
        str(output / "research_workflow.py"), str(output)], cwd=tmp_path,
        capture_output=True, text=True, check=True, timeout=120)
    retained = json.loads(reopened.stdout)
    assert retained["plan_identity"] == summary["plan_identity"]
    assert retained["dataset_complete"] and retained["dataset_members"] == 18
    assert retained["history"] == summary["retry_history"]
