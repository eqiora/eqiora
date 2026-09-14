"""Independent Q1 heat balance and exact offline package/Plan replay."""
import importlib.util
from pathlib import Path
import shutil

import numpy as np
import pytest
import eqiora


SPECIMEN = Path(__file__).resolve().parents[3] / "examples/heated-body"
spec = importlib.util.spec_from_file_location("heated_body_example", SPECIMEN / "run.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def check_heat(model, result, heating=1):
    form, = model.authored_formulations
    trial_id, = form.trial_field_ids
    (_, restriction_trial_id, zero_on), = form.test_restrictions
    assert restriction_trial_id == trial_id
    assert len(zero_on) == len(set(zero_on)) == 4
    field = model.field(trial_id)
    temperatures = np.sort(result.output(field).values("vertex").numpy().reshape(-1))
    assert temperatures.shape == (9,)
    # Four h=1/2 Q1 squares: each central-hat stiffness contribution is 2/3.
    # The source integral against that hat is (1/2)^2=1/4 W/m (per depth).
    # Thus (8/3)*(T_center-300)=1/4, independently of the implementation.
    # 1e-10 K covers the requested residual divided by 8/3 and cancellation at 300 K.
    np.testing.assert_allclose(temperatures[:8], 300.0, atol=1e-10, rtol=0)
    assert temperatures[8] == pytest.approx(300 + heating*3/32, abs=1e-10, rel=0)
    assert (8/3)*(temperatures[8]-300) == pytest.approx(heating/4, abs=3e-10, rel=0)


def test_steady_heated_body_package_runs_and_moves_offline(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(SPECIMEN, project)
    store = tmp_path / "store"
    store.mkdir()
    geometry, bindings = example.geometry_and_bindings()
    lock = eqiora.resolve_local_project(project, store)
    model = eqiora.compile_package(store, lock, entry="HeatedBody",
                                  geometry=geometry, bindings=bindings)
    plan = example.resolve(model, geometry)
    assert plan.formulation.requested == eqiora.FormulationSelectionMode.Authored
    result = eqiora.run(plan)
    check_heat(model, result)
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    assert restored.to_bytes() == plan.to_bytes()
    check_heat(model, eqiora.run(restored))
    assert eqiora.Result.from_bytes(restored, result.to_bytes()).to_bytes() == result.to_bytes()
    direct = eqiora.compile(path=project / "src/main.eqi", entry="HeatedBody",
                            geometry=geometry, bindings=bindings)
    check_heat(direct, eqiora.run(example.resolve(direct, geometry)))
    assert direct.structural_fingerprint == model.structural_fingerprint
    assert model.package_compilation_digest is not None
    assert direct.package_compilation_digest is None
    vendor = project / "vendor"
    vendor.mkdir()
    assert eqiora.vendor_project(project, store, vendor) == lock
    shutil.rmtree(store)
    moved = tmp_path / "moved"
    project.rename(moved)
    assert eqiora.open_project(moved, moved / "vendor") == lock
    offline = eqiora.compile_package(moved / "vendor", lock, entry="HeatedBody",
                                    geometry=geometry, bindings=bindings)
    assert offline.digest == model.digest
    assert offline.package_compilation_digest == model.package_compilation_digest
    check_heat(offline, eqiora.run(example.resolve(offline, geometry)))
    stronger = eqiora.compile_package(moved / "vendor", lock, entry="HeatedBody",
                                     geometry=geometry, bindings={**bindings, "heating": 2.0})
    stronger_plan = example.resolve(stronger, geometry)
    check_heat(stronger, eqiora.run(stronger_plan), heating=2)
    assert stronger_plan.identity != plan.identity
    with pytest.raises(eqiora.ValidationError):
        eqiora.Result.from_bytes(stronger_plan, result.to_bytes())


def transient_plan(model, geometry, step=1/24):
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(2, 2))))
    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100,
    )
    return eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear,
                          temporal=eqiora.time.BackwardEuler(step))


def check_transient(model, result, capacity=1, heating=1, step=1/24, start=0):
    field = model.field("definition.temperature")
    previous = 300 + heating*3/32*(1-(capacity/(capacity+24*step))**start)
    for n, state in enumerate(result.trajectory.states, start+1):
        values = state.field(field).values("vertex").reshape(-1)
        # On the 3x3 Cartesian vertex grid only index 4 is interior; preserve
        # association order so a misplaced heated coefficient cannot pass.
        assert values.shape == (9,)
        np.testing.assert_allclose(np.delete(values, 4), 300, atol=1e-10, rtol=0)
        expected = 300 + heating*3/32*(1-(capacity/(capacity+24*step))**n)
        assert values[4] == pytest.approx(expected, abs=1e-10, rel=0)
        # Four h=1/2 cells give M=4*h^2/9=1/9, K=8/3, F=1/4.
        # With dt=1/24 and c=1, each step halves the steady-state deficit.
        balance = capacity/9*(values[4]-previous)/step + 8/3*(values[4]-300)
        assert balance == pytest.approx(heating/4, abs=2e-9, rel=0)
        previous = values[4]


def test_transient_heat_storage_runs_and_replays_offline(tmp_path):
    project = tmp_path / "heat"
    shutil.copytree(SPECIMEN, project)
    store = tmp_path / "store"
    store.mkdir()
    geometry, bindings = example.geometry_and_bindings()
    bindings["capacity"] = 1.0
    lock = eqiora.resolve_local_project(project, store)
    model = eqiora.compile_package(store, lock, entry="TransientHeatedBody",
                                   geometry=geometry, bindings=bindings)
    plan = transient_plan(model, geometry)
    initial = eqiora.State.initial(plan)
    np.testing.assert_array_equal(initial.field(model.field("definition.temperature")).values("vertex"), 300)
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    assert restored.to_bytes() == plan.to_bytes()
    state = eqiora.State.from_bytes(restored, initial.to_bytes())
    result = eqiora.run(restored, state=state, steps=3, output_steps=(1, 2, 3))
    check_transient(model, result)
    replayed = eqiora.Result.from_bytes(restored, result.to_bytes())
    assert replayed.to_bytes() == result.to_bytes()
    check_transient(model, replayed)
    grid = eqiora.run(plan, state=initial, steps=16, output_steps=(16,))
    assert grid.trajectory.states[0].time_s == 16*(1/24)
    check_transient(model, grid, start=15)
    restart = eqiora.State.from_result(restored, replayed, time_s=1/24)
    check_transient(model, eqiora.run(restored, state=restart, steps=2, output_steps=(1, 2)), start=1)
    direct = eqiora.compile(path=project / "src/main.eqi", entry="TransientHeatedBody",
                            geometry=geometry, bindings=bindings)
    assert direct.structural_fingerprint == model.structural_fingerprint
    assert direct.package_compilation_digest is None
    assert model.package_compilation_digest is not None
    direct_plan = transient_plan(direct, geometry)
    check_transient(direct, eqiora.run(direct_plan, state=eqiora.State.initial(direct_plan),
                                     steps=3, output_steps=(1, 2, 3)))
    vendor = project / "vendor"
    vendor.mkdir()
    assert eqiora.vendor_project(project, store, vendor) == lock
    shutil.rmtree(store)
    moved = tmp_path / "moved"
    project.rename(moved)
    assert eqiora.open_project(moved, moved / "vendor") == lock
    offline = eqiora.compile_package(moved / "vendor", lock, entry="TransientHeatedBody",
                                     geometry=geometry, bindings=bindings)
    assert offline.package_compilation_digest == model.package_compilation_digest
    offline_plan = transient_plan(offline, geometry)
    check_transient(offline, eqiora.run(offline_plan, state=eqiora.State.initial(offline_plan),
                                      steps=3, output_steps=(1, 2, 3)))
    for capacity, heating, step in [(2, 1, 1/24), (1, 2, 1/24), (1, 1, 1/48)]:
        # Isolate the step-policy falsifier on the very same exact Model.
        changed = model if capacity == heating == 1 else eqiora.compile(
            path=moved / "src/main.eqi", entry="TransientHeatedBody", geometry=geometry,
            bindings={**bindings, "capacity": capacity, "heating": heating},
        )
        changed_plan = transient_plan(changed, geometry, step)
        assert changed_plan.identity != plan.identity
        check_transient(changed, eqiora.run(changed_plan, state=eqiora.State.initial(changed_plan),
                                          steps=3, output_steps=(1, 2, 3)), capacity, heating, step)
        with pytest.raises((ValueError, eqiora.ValidationError)):
            eqiora.run(changed_plan, state=initial, steps=1, output_steps=(1,))
        with pytest.raises(eqiora.ValidationError):
            eqiora.Result.from_bytes(changed_plan, result.to_bytes())
    for outputs in [(0,), (2, 1), (4,)]:
        with pytest.raises(eqiora.ValidationError):
            eqiora.run(plan, state=initial, steps=3, output_steps=outputs)


@pytest.mark.parametrize("mutation", ["missing-capacity", "negative-capacity", "missing-storage", "missing-initial", "changed-initial", "missing-boundary", "changed-boundary"])
def test_transient_heat_rejects_incomplete_or_incompatible_data(mutation):
    source = (SPECIMEN / "src/main.eqi").read_text()
    geometry, bindings = example.geometry_and_bindings()
    bindings["capacity"] = -1.0 if mutation == "negative-capacity" else 1.0
    if mutation == "missing-capacity":
        del bindings["capacity"]
    elif mutation == "missing-storage":
        source = source.replace("storage capacity * temperature;", "")
    elif mutation == "missing-initial":
        source = source.replace("initial { temperature = 300[K]; }", "")
    elif mutation == "changed-initial":
        source = source.replace("initial { temperature = 300[K]; }", "initial { temperature = 299[K]; }")
    elif mutation == "missing-boundary":
        source = source.replace("relation prescribed_x_lower on x_lower { trace(temperature) = 300[K]; }", "")
    elif mutation == "changed-boundary":
        source = source.replace("relation prescribed_x_lower on x_lower { trace(temperature) = 300[K]; }", "relation prescribed_x_lower on x_lower { trace(temperature) = 301[K]; }")
    with pytest.raises(eqiora.ValidationError):
        model = eqiora.compile(source=source, entry="TransientHeatedBody", geometry=geometry, bindings=bindings)
        eqiora.State.initial(transient_plan(model, geometry))
