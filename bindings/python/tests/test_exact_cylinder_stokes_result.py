"""Installed-wheel contract for the root exact-cylinder Stokes lifecycle."""

from __future__ import annotations

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

import eqiora
from _signature_bindings import support_bindings


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DEMO = REPOSITORY_ROOT / "examples" / "python" / "exact_cylinder_stokes.py"


def geometry_and_mesh() -> tuple[eqiora.geometry.Geometry, eqiora.meshing.Mesh]:
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
    circle = graph.circle(center=(0.2, 0.2), radius=0.05)
    fluid = graph.subtract(rectangle, circle)
    geometry = graph.build(
        fluid,
        named_topology={
            "fluid": fluid.region,
            "inlet": rectangle.boundaries[0],
            "outlet": rectangle.boundaries[1],
            "walls": rectangle.boundaries[2:],
            "cylinder": circle.boundaries[0],
        },
    )
    request = eqiora.meshing.GmshMesher(
        maximum_boundary_error=1.0e-4,
        minimum_mean_ratio=1.0e-5,
        maximum_boundary_facets=50,
    )
    mesh_plan = eqiora.meshing.resolve(geometry, request)
    return geometry, eqiora.meshing.generate(mesh_plan)


def accepted(*, profile: bool = False) -> tuple[eqiora.geometry.Geometry, eqiora.Model, eqiora.Plan, eqiora.Result]:
    geometry, mesh = geometry_and_mesh()
    model = eqiora.compile(path=files(eqiora).joinpath('examples', 'steady-flow-past-cylinder.eqi'), geometry=geometry, entry='SteadyFlowPastCylinder', bindings={**support_bindings(geometry, ['fluid'], [('inlet', 'fluid'), ('outlet', 'fluid'), ('walls', 'fluid'), ('cylinder', 'fluid')]), **{'dynamic_viscosity': 0.001, 'zero_pressure': 0.0, 'inlet_speed': 0.3, 'channel_height': geometry.bounds[1][1] - geometry.bounds[1][0]}})
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.MiniP1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.SparseLu,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Fast,
            provider=eqiora.solve.SolverProvider.faer(),
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-13,
            maximum_iterations=10_000,
        ),
        scaling=None,
    )
    return geometry, model, plan, eqiora.run(plan, profile=profile)


def test_root_plan_result_and_observation_close_exact_lineage() -> None:
    geometry, model, plan, result = accepted()
    pressure = result.output(plan.capability.pressure)
    evidence = eqiora.fluid.steady_stokes_evidence(result)

    assert result.model_digest == model.digest == plan.model_digest
    assert result.plan_key == plan.identity == evidence.plan_key
    assert pressure.field == plan.capability.pressure
    assert pressure.mesh is plan.mesh
    assert pressure.coefficient_count("vertex") == plan.mesh.vertex_count
    assert pressure.value_shape == ()
    values = pressure.values("vertex").numpy(copy=False)
    assert values.shape == (plan.mesh.vertex_count,)
    assert not values.flags.writeable
    assert np.isfinite(values).all()
    assert float(values.min()) == evidence.pressure_minimum
    assert float(values.max()) == evidence.pressure_maximum

    cylinder_force = result.boundary_force(geometry.selection("cylinder"))
    inlet_flux = result.boundary_flux(geometry.selection("inlet"))
    outlet_flux = result.boundary_flux(geometry.selection("outlet"))
    assert cylinder_force.on_domain == evidence.cylinder_force_on_fluid
    assert cylinder_force.source_digest == result.plan_key
    assert cylinder_force.source_kind == "result"
    assert inlet_flux.value == evidence.inlet_flux
    assert outlet_flux.value == evidence.outlet_flux
    assert inlet_flux.value + outlet_flux.value == evidence.net_flux

    assert evidence.exact_bounds == ((0.0, 2.2), (0.0, 0.41))
    assert evidence.net_flux == evidence.inlet_flux + evidence.outlet_flux
    assert np.isfinite(evidence.net_flux)
    assert np.isfinite(evidence.momentum_closure).all()
    assert evidence.solve.true_residual_norm <= evidence.solve.residual_target


def test_fresh_and_replayed_models_use_the_same_root_resolver() -> None:
    _, model, plan, _ = accepted()
    replayed = eqiora.Model.from_bytes(model.to_bytes())
    again = eqiora.resolve(
        replayed,
        mesh=plan.mesh,
        spatial=eqiora.fem.MiniP1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.SparseLu,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Fast,
            provider=eqiora.solve.SolverProvider.faer(),
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-13,
            maximum_iterations=10_000,
        ),
        scaling=None,
    )
    assert again.identity == plan.identity
    assert again.model_digest == replayed.digest == model.digest
    assert eqiora.run(again).plan_key == again.identity


def test_displaced_fluid_lifecycle_is_absent_and_cross_physics_fails() -> None:
    for name in ("SteadyStokes", "SteadyStokesPlan", "resolve"):
        assert not hasattr(eqiora.fluid, name)
        assert name not in eqiora.fluid.__all__

    ode = eqiora.compile(source="""
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow { derivative(x) + rate * x = 0; }
}
""")
    field = ode.field(ode.field_ids[0])
    ode_plan = eqiora.resolve(
        ode,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={field: 1.0e-11},
        ),
    )
    ode_result = eqiora.run(
        ode_plan,
        state=eqiora.State.initial(ode_plan),
        until_s=0.1,
        output_times_s=(0.1,),
    )
    with pytest.raises(eqiora.CapabilityError):
        eqiora.fluid.steady_stokes_evidence(ode_result)


def test_checked_in_python_demo_runs_with_packaged_component_resource() -> None:
    completed = subprocess.run(
        [sys.executable, str(PYTHON_DEMO)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "cylinder force on fluid" in completed.stdout
    assert "net flux" in completed.stdout


def test_profile_reports_common_and_sparse_lu_phases_without_changing_the_plan() -> None:
    geometry, _, steady_plan, steady_result = accepted(profile=True)
    result = steady_result
    assert result.profile is not None
    paths = {tuple(phase.path) for phase in result.profile.phases}
    assert ("run",) in paths
    assert ("run", "setup") in paths
    assert ("run", "solve") in paths
    assert ("run", "solve", "linear_solve") in paths
    assert ("run", "solve", "linear_solve", "symbolic_factorization") in paths
    assert ("run", "solve", "linear_solve", "numeric_factorization") in paths
    assert ("run", "solve", "linear_solve", "backsolve") in paths
    assert result.profile.total_seconds >= 0.0
    summary = result.profile.summary()
    assert "total inclusive" in summary
    assert "self" in summary
    assert "calls" in summary
    assert "mean per call" in summary
    assert "numeric_factorization" in summary
    assert "×" not in summary
    linear_events = [
        event.fields
        for event in result.profile.events
        if event.fields.get("phase") == "linear_solve"
    ]
    assert any(
        event.get("linear_solver") == "SparseLu"
        and event.get("solver_provider") == "eqiora.faer"
        for event in linear_events
    ), linear_events
    replayed = eqiora.Result.from_bytes(steady_plan, result.to_bytes())
    assert replayed.profile is None
    assert replayed.plan_key == result.plan_key

    model = eqiora.compile(
        path=files(eqiora).joinpath('examples', 'transient-flow-past-cylinder.eqi'),
        geometry=geometry,
        entry='TransientFlowPastCylinder',
        bindings={
            **support_bindings(geometry, ['fluid'], [('inlet', 'fluid'), ('outlet', 'fluid'), ('walls', 'fluid'), ('cylinder', 'fluid')]),
            **{'density': 1.0, 'dynamic_viscosity': 0.001, 'zero_pressure': 0.0, 'inlet_speed': 0.3, 'channel_height': geometry.bounds[1][1] - geometry.bounds[1][0]},
        },
    )
    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-9,
        maximum_iterations=20_000,
    )
    plan = eqiora.resolve(
        model,
        mesh=steady_plan.mesh,
        spatial=eqiora.fem.MiniP1(),
        temporal=eqiora.time.BackwardEuler(0.0001),
        solve=eqiora.solve.Newton(linear=linear),
        scaling=eqiora.fluid.IncompressibleScaling(
            length_m=0.41,
            velocity_m_per_s=0.3,
            pressure_pa=0.09,
        ),
    )
    velocity = steady_result.output(steady_plan.capability.velocity)
    pressure = steady_result.output(steady_plan.capability.pressure)
    state = eqiora.State.initial(
        plan,
        time_s=0.0,
        fields=(
            eqiora.InitialField(
                plan.capability.velocity,
                vertex_values=np.asarray(velocity.values("vertex")).reshape(
                    plan.mesh.vertex_count, 2
                ),
                cell_values=np.asarray(velocity.values("cell-bubble")).reshape(
                    plan.mesh.cell_count, 2
                ),
            ),
            eqiora.InitialField(
                plan.capability.pressure,
                vertex_values=np.asarray(pressure.values("vertex")),
            ),
        ),
    )
    transient = eqiora.run(
        plan, state=state, steps=2, output_steps=(2,), profile=True
    )
    assert transient.profile is not None
    events = transient.profile.events
    time_steps = [
        event for event in events if event.fields.get("phase") == "time_step"
    ]
    assert [step.fields["step"] for step in time_steps] == ["1", "2"]
    assert [float(step.fields["time_s"]) for step in time_steps] == [
        0.0001,
        0.0002,
    ]
    assert all(float(step.fields["dt_s"]) == 0.0001 for step in time_steps)
    assert any(
        event.fields.get("event") == "nonlinear_status"
        and event.fields.get("nonlinear_solver") == "newton"
        and "residual_norm" in event.fields
        and "converged" in event.fields
        for event in events
    )
    assert (
        "run",
        "solve",
        "time_step",
        "nonlinear_iteration",
        "linear_solve",
        "numeric_factorization",
    ) in {tuple(phase.path) for phase in transient.profile.phases}
    phase_identities = {
        (tuple(phase.path), tuple(sorted(phase.fields.items())))
        for phase in transient.profile.phases
    }
    assert (
        ("run", "solve", "setup"),
        (("role", "discretization_preparation"),),
    ) in phase_identities
    assert (
        ("run", "solve", "time_step", "assembly"),
        (("role", "initial_linearization"),),
    ) in phase_identities
    assert (
        ("run", "solve", "time_step", "nonlinear_iteration", "assembly"),
        (("role", "line_search_trial"),),
    ) in phase_identities
    prepared_phases = {
        name: [phase for phase in transient.profile.phases if phase.name == name]
        for name in (
            "assembly_preparation",
            "assembly_local_evaluation",
            "assembly_scatter_update",
            "assembly_finalization",
        )
    }
    assert sum(phase.calls for phase in prepared_phases["assembly_preparation"]) == 1
    scatter_calls = sum(
        phase.calls for phase in prepared_phases["assembly_scatter_update"]
    )
    assert scatter_calls == 3 * len(time_steps)
    assert (
        sum(phase.calls for phase in prepared_phases["assembly_finalization"])
        == scatter_calls
    )
    assert (
        sum(phase.calls for phase in prepared_phases["assembly_local_evaluation"])
        > scatter_calls
    )
    for phases in prepared_phases.values():
        for phase in phases:
            assert phase.calls > 0
            assert phase.inclusive_seconds >= phase.self_seconds >= 0.0
            assert phase.mean_seconds == phase.inclusive_seconds / phase.calls
    phase_calls = {
        path: sum(
            phase.calls
            for phase in transient.profile.phases
            if tuple(phase.path) == path
        )
        for path in {tuple(phase.path) for phase in transient.profile.phases}
    }
    linear_path = (
        "run",
        "solve",
        "time_step",
        "nonlinear_iteration",
        "linear_solve",
    )
    symbolic_path = (*linear_path, "symbolic_factorization")
    numeric_path = (*linear_path, "numeric_factorization")
    backsolve_path = (*linear_path, "backsolve")
    assert phase_calls[linear_path] > 1
    assert phase_calls[symbolic_path] == 1
    assert phase_calls[numeric_path] == phase_calls[linear_path]
    assert phase_calls[backsolve_path] == phase_calls[linear_path]
