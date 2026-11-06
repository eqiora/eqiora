"""Installed-wheel contract for the root exact-cylinder Stokes lifecycle."""

from __future__ import annotations

import subprocess
import sys
import copy
import json
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
    model = eqiora.compile(path=files(eqiora).joinpath('examples', 'steady-flow-past-cylinder.eqi'), geometry=geometry, entry='SteadyFlowPastCylinder', bindings={**support_bindings(geometry, ['fluid'], [('inlet', 'fluid'), ('outlet', 'fluid'), ('walls', 'fluid'), ('cylinder', 'fluid')]), **{'dynamic_viscosity': 0.001, 'inlet_speed': 0.3, 'channel_height': geometry.bounds[1][1] - geometry.bounds[1][0]}})
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
    assert pressure.value_type.shape == []
    values = pressure.values("vertex").numpy(copy=False)
    assert values.shape == (plan.mesh.vertex_count,)
    assert not values.flags.writeable
    assert np.isfinite(values).all()
    assert float(values.min()) == evidence.pressure_minimum
    assert float(values.max()) == evidence.pressure_maximum

    cylinder_force = result.boundary_force(geometry.selection("cylinder"))
    inlet_flux = result.boundary_flux(geometry.selection("inlet"))
    outlet_flux = result.boundary_flux(geometry.selection("outlet"))
    assert np.isfinite(cylinder_force.on_domain).all()
    assert cylinder_force.source_digest == json.loads(result.to_bytes())["identity"]
    assert cylinder_force.output_index is None
    assert cylinder_force.source_kind == "result"
    assert inlet_flux.value + outlet_flux.value == evidence.net_flux

    assert evidence.exact_bounds == ((0.0, 2.2), (0.0, 0.41))
    assert evidence.net_flux == inlet_flux.value + outlet_flux.value
    assert np.isfinite(evidence.net_flux)
    assert np.isfinite(evidence.momentum_closure).all()
    assert evidence.solve.true_residual_norm <= evidence.solve.residual_target


def test_transient_equations_accept_renamed_and_split_boundaries() -> None:
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
    circle = graph.circle(center=(0.2, 0.2), radius=0.05)
    region = graph.subtract(rectangle, circle)
    geometry = graph.build(region, named_topology={
        "region": region.region,
        "feed": rectangle.boundaries[0],
        "exit": rectangle.boundaries[1],
        "upper": rectangle.boundaries[2],
        "lower": rectangle.boundaries[3],
        "obstacle": circle.boundaries[0],
    })
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.GmshMesher(
            maximum_boundary_error=1.0e-4,
            minimum_mean_ratio=1.0e-5,
            maximum_boundary_facets=50,
        ),
    ))
    names = {"fluid": "region", "inlet": "feed", "outlet": "exit",
             "cylinder": "obstacle", "upper": "upper", "lower": "lower"}
    bindings = {
        "fluid": geometry.selection("region"),
        **{name: (geometry.selection(selection), geometry.selection("region"))
           for name, selection in names.items() if name != "fluid"},
        "dynamic_viscosity": 0.001,
        "inlet_speed": 0.3, "channel_height": 0.41,
    }
    def compile_equations(filename, entry, parameters):
        source = files(eqiora).joinpath("examples", filename).read_text()
        source = source.replace(
            "support walls: boundary(parent = fluid),",
            "support upper: boundary(parent = fluid),\n"
            "  support lower: boundary(parent = fluid),",
        ).replace(
            "relation wall_velocity on walls { trace(velocity) = 0; }",
            "relation upper_velocity on upper { trace(velocity) = 0; }\n"
            "  relation lower_velocity on lower { trace(velocity) = 0; }",
        )
        if filename == "steady-flow-past-cylinder.eqi":
            source = source.replace(
                "boundaries(inlet, outlet, walls, cylinder)",
                "boundaries(inlet, outlet, upper, lower, cylinder)",
            ).replace(
                "instance wall: stokes.NoSlip2d(body = fluid, face = walls);",
                "instance upper_wall: stokes.NoSlip2d(body = fluid, face = upper);\n"
                "  instance lower_wall: stokes.NoSlip2d(body = fluid, face = lower);",
            ).replace(
                "connect flow.mechanical[boundary = walls], wall.mechanical;",
                "connect flow.mechanical[boundary = upper], upper_wall.mechanical;\n"
                "  connect flow.mechanical[boundary = lower], lower_wall.mechanical;",
            )
        return eqiora.compile(source=source, geometry=geometry, entry=entry,
                              bindings={**bindings, **parameters})

    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1.0e-6, absolute_tolerance=1.0e-13,
        maximum_iterations=10_000,
    )
    steady = eqiora.resolve(
        compile_equations("steady-flow-past-cylinder.eqi", "SteadyFlowPastCylinder", {}),
        mesh=mesh, spatial=eqiora.fem.MiniP1(), solve=linear, scaling=None,
    )
    result = eqiora.run(steady)
    replayed = eqiora.Result.from_bytes(steady, result.to_bytes())
    assert replayed.boundary_force(geometry.selection("obstacle")).on_domain == result.boundary_force(geometry.selection("obstacle")).on_domain
    for name in ("feed", "exit"):
        assert replayed.boundary_flux(geometry.selection(name)).value == result.boundary_flux(geometry.selection(name)).value
    wire = json.loads(result.to_bytes())
    observation = wire["content"]["payload"]["acceptance"]
    assert set(observation) == {"family", "acceptance"}
    assert set(observation["acceptance"]) == {
        "continuity_residual_norm", "gauge_multiplier",
    }
    # The natural outlet traction fixes pressure, so this Plan has no auxiliary gauge.
    assert observation["acceptance"]["gauge_multiplier"] is None
    assert result.gauge_multiplier is None
    assert replayed.gauge_multiplier is None
    mutant = copy.deepcopy(wire)
    mutant["content"]["payload"]["acceptance"]["acceptance"]["gauge_multiplier"] = 1.0
    with pytest.raises(eqiora.ValidationError, match="gauge multiplier"):
        eqiora.Result.from_bytes(steady, json.dumps(mutant).encode())
    for name in ("constrained_reaction", "integrated_body_force", "integrated_boundary_traction"):
        mutant = copy.deepcopy(wire)
        mutant["content"]["payload"]["acceptance"]["acceptance"][name] = [0.0, 0.0]
        with pytest.raises(eqiora.ValidationError, match="unknown field"):
            eqiora.Result.from_bytes(steady, json.dumps(mutant).encode())
    for name, removed in (("reactions", [["unselected", [0.0, 0.0]]]),
                          ("fluxes", [["feed", 0.0]])):
        mutant = copy.deepcopy(wire)
        mutant["content"]["payload"]["acceptance"][name] = removed
        with pytest.raises(eqiora.ValidationError, match="unknown field"):
            eqiora.Result.from_bytes(steady, json.dumps(mutant).encode())
    plan = eqiora.resolve(
        compile_equations("transient-flow-past-cylinder.eqi", "TransientFlowPastCylinder",
                          {"density": 1.0}),
        mesh=mesh, spatial=eqiora.fem.MiniP1(),
        temporal=eqiora.time.BackwardEuler(1.0e-4),
        solve=eqiora.solve.Newton(linear=linear),
        scaling=eqiora.fluid.IncompressibleScaling(
            length_m=0.41, velocity_m_per_s=0.3, pressure_pa=0.09),
    )
    velocity = result.output(steady.capability.velocity)
    pressure = result.output(steady.capability.pressure)
    initial = eqiora.State.initial(plan, time_s=0.0, fields=(
        eqiora.InitialField(plan.capability.velocity,
                           vertex_values=np.asarray(velocity.values("vertex")).reshape(-1, 2),
                           cell_values=np.asarray(velocity.values("cell-bubble")).reshape(-1, 2)),
        eqiora.InitialField(plan.capability.pressure,
                           vertex_values=np.asarray(pressure.values("vertex"))),
    ))
    state = eqiora.run(plan, state=initial, steps=1, output_steps=(1,)).trajectory.states[-1]
    values = np.asarray(state.field(plan.capability.velocity).values("vertex")).reshape(-1, 2)
    assert state.time_s == 1.0e-4
    assert np.isfinite(values).all()
    coordinates = np.asarray(mesh.coordinates)
    feed = coordinates[:, 0] == 0.0
    y = coordinates[feed, 1]
    expected = 4.0 * 0.3 * y * (0.41 - y) / 0.41**2
    assert np.max(expected) > 0.1
    np.testing.assert_allclose(values[feed, 0], expected, rtol=0.0, atol=64 * np.finfo(float).eps)
    np.testing.assert_allclose(values[feed, 1], 0.0, rtol=0.0, atol=64 * np.finfo(float).eps)
    assert np.isfinite(np.asarray(state.curl(plan.capability.velocity).values("cell"))).all()
    assert np.isfinite(result.boundary_force(geometry.selection("obstacle")).on_domain).all()


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
            absolute_tolerances={(field, 0, 0, False): 1.0e-11},
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


def test_steady_mini_profile_reports_one_assembly_and_preserves_artifact() -> None:
    _, _, plan, profiled = accepted(profile=True)
    ordinary = eqiora.run(plan)
    assert ordinary.profile is None
    # Result envelopes include elapsed time in their identity. Compare all
    # numerical content and retain exact artifact replay for each execution.
    ordinary_content = json.loads(ordinary.to_bytes())["content"]
    profiled_content = json.loads(profiled.to_bytes())["content"]
    for content in (ordinary_content, profiled_content):
        content.pop("elapsed_seconds")
    assert profiled_content == ordinary_content
    for result in (ordinary, profiled):
        replayed = eqiora.Result.from_bytes(plan, result.to_bytes())
        assert replayed.to_bytes() == result.to_bytes()
        assert replayed.profile is None
    assert profiled.profile is not None
    phases = profiled.profile.phases
    assembly = [phase for phase in phases if phase.name == "assembly"]
    assert len(assembly) == 1
    assert tuple(assembly[0].path) == ("run", "solve", "assembly")
    assert assembly[0].fields["role"] == "mini_stokes_finalization"
    assert assembly[0].calls == 1
    linear = [phase for phase in phases if phase.name == "linear_solve"]
    assert len(linear) == 1
    assert tuple(linear[0].path) == ("run", "solve", "linear_solve")
    assert linear[0].calls == 1
    paths = {tuple(phase.path) for phase in phases}
    for name in ("symbolic_factorization", "numeric_factorization", "backsolve"):
        assert ("run", "solve", "linear_solve", name) in paths
    assert all(phase.path.count("assembly") <= 1 for phase in phases)


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
            **{'density': 1.0, 'dynamic_viscosity': 0.001, 'inlet_speed': 0.3, 'channel_height': geometry.bounds[1][1] - geometry.bounds[1][0]},
        },
    )
    maximum_backsolves = 20_000
    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-12,
        maximum_iterations=maximum_backsolves,
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
    # Each numeric factorization serves an original backsolve and any
    # safeguarded residual corrections within the requested per-solve budget.
    assert phase_calls[linear_path] <= phase_calls[backsolve_path]
    assert phase_calls[backsolve_path] <= maximum_backsolves * phase_calls[linear_path]


IMPORTED_TRANSLATION_SOURCE = """
import Eqiora.Fluid.fluid as fluid;
public component Translation(
  support body: volume(ambient_dimension=2),
  support inlet: boundary(parent=body),
  support outlet: boundary(parent=body),
  support walls: boundary(parent=body),
  support cylinder: boundary(parent=body),
  parameter speed: m/s
) {
  variable velocity: vector<m/s,2> on body in smooth;
  variable pressure: Pa on body in h1;
  instance flow: fluid.SteadyStokes2d(
    body=body, exterior=boundaries(inlet,outlet,walls,cylinder),
    velocity=velocity, pressure=pressure, dynamic_viscosity=1[Pa*s]
  );
  instance a: fluid.PrescribedVelocity2d(body=body,face=inlet,
    velocity=speed*tensor_value(frame=body,components=[1,0]));
  instance b: fluid.PrescribedVelocity2d(body=body,face=outlet,
    velocity=speed*tensor_value(frame=body,components=[1,0]));
  instance c: fluid.PrescribedVelocity2d(body=body,face=walls,
    velocity=speed*tensor_value(frame=body,components=[1,0]));
  instance d: fluid.PrescribedVelocity2d(body=body,face=cylinder,
    velocity=speed*tensor_value(frame=body,components=[1,0]));
  connect flow.mechanical[boundary=inlet], a.mechanical;
  connect flow.mechanical[boundary=outlet], b.mechanical;
  connect flow.mechanical[boundary=walls], c.mechanical;
  connect flow.mechanical[boundary=cylinder], d.mechanical;
}
"""


@pytest.mark.parametrize("speed", [0.7, 1.3])
def test_imported_translation_on_curved_and_grouped_boundaries(speed):
    # Uniform u=(U,0) has zero strain and divergence, with p=0 under the
    # all-Dirichlet zero-mean gauge. It is represented exactly by MINI/P1 on
    # any triangular mesh. The circular boundary moves with the flow here;
    # this is not a stationary no-slip obstacle or a reaction-force oracle.
    # Reuse the physical reconstruction tolerance of
    # verify/fluid/port-closed-si-mini-stokes-2d, not observed Eqiora output.
    geometry, mesh = geometry_and_mesh()
    model = eqiora.compile(
        source=IMPORTED_TRANSLATION_SOURCE, entry="Translation",
        geometry=geometry,
        bindings={"speed": speed, "body": geometry.selection("fluid"), **{
            name: geometry.selection(name)
            for name in ("inlet", "outlet", "walls", "cylinder")
        }},
    )
    assert len(model.field_ids) == 2
    assert model.parameter_ids == [model.parameter("definition.speed").id]
    plan = eqiora.resolve(
        model, mesh=mesh, spatial=eqiora.fem.MiniP1(), scaling=None,
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.SparseLu,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Fast,
            provider=eqiora.solve.SolverProvider.faer(),
            relative_tolerance=1e-10, absolute_tolerance=1e-13,
            maximum_iterations=10_000,
        ),
    )
    result = eqiora.run(plan)
    for candidate in (result, eqiora.Result.from_bytes(plan, result.to_bytes())):
        velocity = candidate.output(plan.capability.velocity)
        vertices = np.asarray(velocity.values("vertex")).reshape(-1, 2)
        np.testing.assert_allclose(vertices[:, 0], speed, atol=2e-10, rtol=0)
        np.testing.assert_allclose(vertices[:, 1], 0, atol=2e-10, rtol=0)
        np.testing.assert_allclose(velocity.values("cell-bubble"), 0, atol=2e-10, rtol=0)
        pressure = candidate.output(plan.capability.pressure)
        np.testing.assert_allclose(pressure.values("vertex"), 0, atol=2e-10, rtol=0)
