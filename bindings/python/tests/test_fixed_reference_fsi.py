"""Installed-wheel contract for Model-first common fixed-reference FSI."""

from __future__ import annotations

import gc
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

import eqiora
from _signature_bindings import support_bindings


PARAMETERS = {
    "fluid_density": 2.0,
    "fluid_viscosity": 0.5,
    "solid_density": 3.0,
    "solid_mu": 4.0,
    "solid_lambda": 2.0,
    "zero_pressure": 0.0,
}
EXPECTED_CELLS = np.array(
    [
        [0, 3, 4],
        [0, 4, 1],
        [1, 4, 5],
        [1, 5, 2],
        [3, 6, 7],
        [3, 7, 4],
        [4, 7, 8],
        [4, 8, 5],
    ],
    dtype=np.uint32,
)


def geometry_and_mesh() -> tuple[eqiora.geometry.Geometry, eqiora.meshing.Mesh]:
    graph = eqiora.geometry.GeometryGraph()
    fluid = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    solid = graph.rectangle(x_bounds=(1.0, 2.0), y_bounds=(0.0, 1.0))
    partition = graph.partition(
        fluid, solid, interface=(fluid.boundaries[1], solid.boundaries[0])
    )
    geometry = graph.build(
        partition,
        named_topology={
            "fluid": fluid.region,
            "fluid_x_lower": fluid.boundaries[0],
            "fluid_x_upper": fluid.boundaries[1],
            "fluid_y_lower": fluid.boundaries[2],
            "fluid_y_upper": fluid.boundaries[3],
            "solid": solid.region,
            "solid_x_lower": solid.boundaries[0],
            "solid_x_upper": solid.boundaries[1],
            "solid_y_lower": solid.boundaries[2],
            "solid_y_upper": solid.boundaries[3],
        },
    )
    request = eqiora.meshing.AffineTriangleMesher(cells=(2, 2))
    return geometry, eqiora.meshing.generate(eqiora.meshing.resolve(geometry, request))


def admitted() -> tuple[eqiora.Model, eqiora.meshing.Mesh, eqiora.Plan]:
    geometry, mesh = geometry_and_mesh()
    model = eqiora.compile(path=files(eqiora).joinpath('examples', 'fixed-reference-fsi.eqi'), geometry=geometry, entry='FixedReferenceFsi2d', bindings={**support_bindings(geometry, ['fluid', 'solid'], [('fluid_x_lower', 'fluid'), ('fluid_x_upper', 'fluid'), ('fluid_y_lower', 'fluid'), ('fluid_y_upper', 'fluid'), ('solid_x_lower', 'solid'), ('solid_x_upper', 'solid'), ('solid_y_lower', 'solid'), ('solid_y_upper', 'solid')]), **PARAMETERS})
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=(
            eqiora.fem.MiniP1().at(model.domain("fluid")),
            eqiora.fem.P1().at(model.domain("solid")),
        ),
        temporal=eqiora.time.BackwardEuler(step_s=0.05),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.MinimumResidual,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-13,
            maximum_iterations=20_000,
        ),
        scaling=None,
    )
    return model, mesh, plan


def linear(**overrides: float | int) -> eqiora.solve.Linear:
    values = {
        "algorithm": eqiora.solve.LinearSolver.MinimumResidual,
        "preconditioner": eqiora.solve.Preconditioner.Identity,
        "reduction": eqiora.solve.Reduction.Reproducible,
        "provider": eqiora.solve.SolverProvider.reference(),
        "relative_tolerance": 1.0e-11,
        "absolute_tolerance": 1.0e-13,
        "maximum_iterations": 20_000,
    }
    values.update(overrides)
    return eqiora.solve.Linear(**values)


def initial(
    model: eqiora.Model, mesh: eqiora.meshing.Mesh, plan: eqiora.Plan
) -> eqiora.State:
    coordinates = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.cells)
    fluid_vertices = np.flatnonzero(coordinates[:, 0] <= 1.0)
    solid_vertices = np.flatnonzero(coordinates[:, 0] >= 1.0)
    fluid_cells = np.flatnonzero(coordinates[cells, 0].mean(axis=1) < 1.0)
    displacement = np.zeros((solid_vertices.size, 2))
    interface_midpoint = np.flatnonzero(
        (coordinates[solid_vertices, 0] == 1.0)
        & (coordinates[solid_vertices, 1] == 0.5)
    )
    assert interface_midpoint.size == 1
    displacement[interface_midpoint[0], 0] = 0.02
    fluid_velocity, fluid_pressure, solid_velocity, solid_displacement = plan.fields
    return eqiora.State.initial(
        plan,
        time_s=0.0,
        fields=(
            eqiora.InitialField(
                fluid_velocity,
                vertex_values=np.zeros((fluid_vertices.size, 2)),
                cell_values=np.zeros((fluid_cells.size, 2)),
            ),
            eqiora.InitialField(
                fluid_pressure,
                vertex_values=np.full(fluid_vertices.size, 0.25),
            ),
            eqiora.InitialField(
                solid_velocity,
                vertex_values=np.zeros((solid_vertices.size, 2)),
            ),
            eqiora.InitialField(solid_displacement, vertex_values=displacement),
        ),
    )


def solved() -> tuple[eqiora.Model, eqiora.Plan, eqiora.Run, eqiora.Result]:
    model, mesh, plan = admitted()
    run = eqiora.submit(
        plan, state=initial(model, mesh, plan), steps=2, output_steps=(1, 2)
    )
    return model, plan, run, run.result()


def test_cancellation_exposes_the_last_exact_accepted_state() -> None:
    model, mesh, plan = admitted()
    run = eqiora.submit(
        plan,
        state=initial(model, mesh, plan),
        steps=100,
        output_steps=(100,),
    )
    assert run.cancel() is True
    with pytest.raises(eqiora.CancellationError):
        run.result()

    cancellation = run.cancellation
    assert cancellation is not None
    assert cancellation.request_identity == run.plan_key
    assert cancellation.state.step == cancellation.progress.accepted_steps
    assert cancellation.state.time_s == cancellation.progress.model_time_s
    assert cancellation.state.model == model
    assert cancellation.state.source_kind == "cancellation"
    assert cancellation.state.source_request_identity == cancellation.request_identity


def test_only_common_model_first_fsi_surface_is_public() -> None:
    assert not hasattr(eqiora.fsi, "FixedMeshMonolithic")
    assert not hasattr(eqiora.fsi, "FixedMeshMonolithicPlan")
    assert not hasattr(eqiora.fsi, "resolve")
    assert eqiora.fsi.__all__ == [
        "FixedReferenceFsiPlanView",
        "FsiEvidence",
        "FsiStateEvidence",
        "evidence",
    ]


def test_plan_binds_exact_model_mesh_scopes_provider_and_scaling_receipt() -> None:
    model, mesh, plan = admitted()
    plan_bytes = plan.to_bytes()
    portable = eqiora.Plan.from_bytes(plan_bytes)
    assert plan.model is model
    assert plan.mesh is mesh
    assert plan.model_digest == model.digest
    assert plan.geometry_digest == mesh.source_digest
    assert plan.mesh_digest == mesh.digest
    assert plan.correspondence_digest == mesh.correspondence_digest
    assert plan.production_digest == mesh.production_lineage_digest
    assert plan.identity.startswith("common-fsi:") and len(plan.identity) == 75
    assert len(plan.realization_digest) == 64
    assert tuple(binding.domain for binding in plan.spatial) == (
        model.domain("fluid"),
        model.domain("solid"),
    )
    assert plan.temporal.step_s == 0.05
    assert plan.solve.maximum_iterations == 20_000
    assert isinstance(plan.capability, eqiora.fsi.FixedReferenceFsiPlanView)
    assert plan.capability.scaling is not None
    assert plan.capability.scaling_receipt.production_digest == mesh.production_lineage_digest
    assert not hasattr(plan.capability, "pressure_gauge")
    assert plan.solve.algorithm == "minimum-residual"
    assert plan.solve.backend == "eqiora.reference"
    assert plan.execution.provider == "eqiora.host.serial"
    assert plan.execution.placement == "host-serial"
    assert len(set(plan.fields)) == 4
    assert plan.fields[:2] == (plan.capability.fluid_velocity, plan.capability.pressure)
    assert portable.identity == plan.identity
    assert portable.to_bytes() == plan_bytes
    assert portable.model.to_bytes() == model.to_bytes()
    assert portable.mesh.to_bytes() == mesh.to_bytes()
    assert tuple(binding.domain.id for binding in portable.spatial) == tuple(
        binding.domain.id for binding in plan.spatial
    )
    assert portable.temporal.step_s == plan.temporal.step_s
    assert portable.requested_solve.maximum_iterations == plan.requested_solve.maximum_iterations
    assert portable.capability.scaling.length_m == plan.capability.scaling.length_m
    assert (
        portable.capability.scaling_receipt.provenance_digest
        == plan.capability.scaling_receipt.provenance_digest
    )


def test_initial_state_is_exact_field_bound_complete_and_gauge_free() -> None:
    model, mesh, plan = admitted()
    state = initial(model, mesh, plan)
    assert state.model is model
    assert state.mesh is mesh
    assert state.time_s == 0.0
    state_bytes = state.to_bytes()
    replayed = eqiora.State.from_bytes(plan, state_bytes)
    assert replayed == state
    assert replayed.to_bytes() == state_bytes
    assert replayed.source_kind == "artifact"
    np.testing.assert_array_equal(
        state.field(plan.capability.pressure).values("vertex"), 0.25
    )
    expected_displacement = np.zeros((6, 2))
    expected_displacement[1, 0] = 0.02
    np.testing.assert_array_equal(
        state.field(plan.fields[3]).values("vertex"), expected_displacement
    )
    with pytest.raises(ValueError, match="time_s"):
        eqiora.State.initial(plan, fields=())
    with pytest.raises(eqiora.ValidationError):
        eqiora.State.initial(plan, time_s=0.0, fields=())
    with pytest.raises(TypeError):
        eqiora.InitialField(plan.capability.pressure, values=[0.0])


def test_common_worker_run_outputs_restart_and_observation_evidence() -> None:
    model, mesh, plan = admitted()
    state = initial(model, mesh, plan)
    run = eqiora.submit(plan, state=state, steps=2, output_steps=(1, 2))
    result = run.result()
    assert run.status is eqiora.RunStatus.Completed
    assert run.model_digest == model.digest
    assert state.source_plan_identity == plan.identity
    assert state.source_request_identity is None
    trajectory = result.trajectory
    outputs = trajectory.states
    assert tuple(value.step for value in outputs) == (1, 2)
    assert tuple(value.time_s for value in outputs) == (0.05, 0.10)
    output_bytes = outputs[0].to_bytes()
    replayed_output = eqiora.State.from_bytes(plan, output_bytes)
    assert replayed_output == outputs[0]
    assert replayed_output.to_bytes() == output_bytes
    assert replayed_output.source_kind == "artifact"
    replayed_result = eqiora.run(
        plan, state=replayed_output, steps=1, output_steps=(1,)
    )
    assert replayed_result.trajectory.states[0].time_s == 0.10
    assert trajectory.plan_identity == plan.identity
    assert trajectory.request_identity == run.plan_key
    assert trajectory.run_digest == run.plan_key
    trajectory_bytes = trajectory.to_bytes()
    replayed_trajectory = eqiora.trajectory.Trajectory.from_bytes(plan, trajectory_bytes)
    assert replayed_trajectory == trajectory
    assert replayed_trajectory.to_bytes() == trajectory_bytes
    assert tuple(state.digest for state in replayed_trajectory.states) == tuple(
        state.digest for state in trajectory.states
    )
    for output in outputs:
        assert output.source_plan_identity == plan.identity
        assert output.source_request_identity == run.plan_key
        assert output.source_trajectory_identity == trajectory.digest
        assert tuple(snapshot.field for snapshot in output.fields) == plan.fields
        for field, snapshot in zip(plan.fields, output.fields, strict=True):
            assert output.field(field) is snapshot
    evidence = eqiora.fsi.evidence(result)
    assert len(evidence.states) == 2
    assert evidence.states[-1].solve.algorithm == "minimum-residual"
    assert (
        evidence.states[-1].solve.true_residual_norm
        <= evidence.states[-1].solve.residual_target
    )
    assert not evidence.fluid_cells.flags.writeable
    assert not evidence.solid_cells.flags.writeable
    assert not evidence.interface_facets.flags.writeable

    fresh_plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=(
            eqiora.fem.MiniP1().at(model.domain("fluid")),
            eqiora.fem.P1().at(model.domain("solid")),
        ),
        temporal=eqiora.time.BackwardEuler(step_s=0.05),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.MinimumResidual,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10_000,
        ),
        scaling=None,
    )
    restarted = eqiora.run(
        fresh_plan, state=result.trajectory.states[-1], steps=1, output_steps=(1,)
    )
    assert restarted.trajectory.states[-1].time_s == pytest.approx(0.15)


def test_scoped_domain_handles_reject_foreign_models() -> None:
    model, mesh, _ = admitted()
    foreign_geometry, _ = geometry_and_mesh()
    foreign = eqiora.compile(path=files(eqiora).joinpath('examples', 'fixed-reference-fsi.eqi'), geometry=foreign_geometry, entry='FixedReferenceFsi2d', bindings={**support_bindings(foreign_geometry, ['fluid', 'solid'], [('fluid_x_lower', 'fluid'), ('fluid_x_upper', 'fluid'), ('fluid_y_lower', 'fluid'), ('fluid_y_upper', 'fluid'), ('solid_x_lower', 'solid'), ('solid_x_upper', 'solid'), ('solid_y_lower', 'solid'), ('solid_y_upper', 'solid')]), **{**PARAMETERS, 'fluid_density': 4.0}})
    with pytest.raises(eqiora.ValidationError):
        eqiora.resolve(
            model,
            mesh=mesh,
            spatial=(
                eqiora.fem.MiniP1().at(foreign.domain("fluid")),
                eqiora.fem.P1().at(model.domain("solid")),
            ),
            temporal=eqiora.time.BackwardEuler(step_s=0.05),
            solve=linear(),
        )


def test_observation_evidence_is_complete_immutable_and_state_bound() -> None:
    _, plan, run, result = solved()
    trajectory = result.trajectory
    evidence = eqiora.fsi.evidence(result)
    assert run.result() is result
    assert result.trajectory is trajectory
    assert eqiora.fsi.evidence(result) is evidence
    assert trajectory.coordinates is trajectory.coordinates
    assert trajectory.cells is trajectory.cells
    assert evidence.fluid_cells is evidence.fluid_cells
    assert evidence.solid_cells is evidence.solid_cells
    assert evidence.interface_facets is evidence.interface_facets
    assert trajectory.coordinates.shape == (9, 2)
    assert trajectory.cells.shape == (8, 3)
    np.testing.assert_array_equal(trajectory.cells, EXPECTED_CELLS)
    np.testing.assert_array_equal(evidence.fluid_cells, [0, 1, 2, 3])
    np.testing.assert_array_equal(evidence.solid_cells, [4, 5, 6, 7])
    np.testing.assert_array_equal(evidence.interface_facets, [[3, 4], [4, 5]])
    assert tuple(state.step for state in trajectory.states) == (1, 2)
    assert tuple(state.time_s for state in trajectory.states) == (0.05, 0.10)
    assert trajectory.state(1) is trajectory.states[0]
    assert trajectory.state(2) is trajectory.states[1]
    with pytest.raises(IndexError):
        trajectory.state(0)
    assert (
        tuple(evidence.state(state) for state in trajectory.states) == evidence.states
    )

    displacements: list[np.ndarray] = []
    for state in trajectory.states:
        state_evidence = evidence.state(state)
        assert evidence.state(state) is state_evidence
        assert state_evidence.state_digest == state.digest
        velocity = state.field(plan.fields[0])
        assert state.field(plan.fields[0]) is velocity
        pressure = state.field(plan.fields[1])
        displacement = state.field(plan.fields[3])
        velocity_vertices = velocity.values("vertex")
        velocity_bubbles = velocity.values("cell")
        pressure_support = pressure.support_indices("vertex")
        pressure_values = pressure.values("vertex")
        displacement_values = displacement.values("vertex")
        displacements.append(displacement_values)
        arrays = (
            velocity_vertices,
            velocity_bubbles,
            pressure_support,
            pressure_values,
            displacement_values,
            state_evidence.interface_vertices,
            state_evidence.fluid_action,
            state_evidence.solid_action,
            state_evidence.action_imbalance,
        )
        assert velocity.values("vertex") is velocity_vertices
        assert velocity.values("cell") is velocity_bubbles
        assert pressure.support_indices("vertex") is pressure_support
        assert pressure.values("vertex") is pressure_values
        assert displacement.values("vertex") is displacement_values
        assert state_evidence.interface_vertices is state_evidence.interface_vertices
        assert state_evidence.fluid_action is state_evidence.fluid_action
        assert state_evidence.solid_action is state_evidence.solid_action
        assert state_evidence.action_imbalance is state_evidence.action_imbalance
        assert velocity_vertices.shape == (6, 2)
        assert velocity_bubbles.shape == (4, 2)
        assert pressure_support.shape == (6,)
        assert pressure_values.shape == (6,)
        assert displacement_values.shape == (6, 2)
        assert state_evidence.interface_vertices.shape == (1,)
        assert state_evidence.fluid_action.shape == (1, 2)
        assert state_evidence.solid_action.shape == (1, 2)
        assert state_evidence.action_imbalance.shape == (1, 2)
        np.testing.assert_array_equal(pressure_support, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(state_evidence.interface_vertices, [4])
        np.testing.assert_array_equal(
            displacement.support_indices("vertex"), [3, 4, 5, 6, 7, 8]
        )
        np.testing.assert_array_equal(
            state_evidence.fluid_action + state_evidence.solid_action,
            state_evidence.action_imbalance,
        )
        assert all(array.flags.c_contiguous for array in arrays)
        assert all(not array.flags.writeable for array in arrays)
        assert all(np.isfinite(array).all() for array in arrays)
        assert state_evidence.solve.algorithm == "minimum-residual"
        assert state_evidence.solve.preconditioner == "identity"
        assert state_evidence.solve.reduction == "reproducible"
        assert (
            state_evidence.solve.true_residual_norm
            <= state_evidence.solve.residual_target
        )
        assert state_evidence.assembly_packets > 0
        assert state_evidence.assembly_targets > 0
        assert np.isfinite(
            [
                state_evidence.previous_kinetic_energy_j_per_m,
                state_evidence.next_kinetic_energy_j_per_m,
                state_evidence.previous_elastic_energy_j_per_m,
                state_evidence.next_elastic_energy_j_per_m,
                state_evidence.kinetic_increment_j_per_m,
                state_evidence.elastic_increment_j_per_m,
                state_evidence.viscous_dissipation_j_per_m,
                state_evidence.energy_defect_j_per_m,
                state_evidence.numerical_residual_norm,
                state_evidence.continuity_residual_norm,
                state_evidence.kinematic_residual_norm,
                state_evidence.interface_velocity_jump_norm,
                state_evidence.interface_action_imbalance_n_per_m,
            ]
        ).all()
    assert not np.array_equal(displacements[0], displacements[1])


def test_evidence_state_lookup_is_bound_to_exact_result_occurrence() -> None:
    model, plan, _, result = solved()
    evidence = eqiora.fsi.evidence(result)
    mesh = plan.mesh
    assert mesh is not None
    repeated = eqiora.run(
        plan,
        state=initial(model, mesh, plan),
        steps=2,
        output_steps=(1, 2),
    )
    for foreign_state in repeated.trajectory.states:
        with pytest.raises(ValueError, match="different exact|occurrence|trajectory"):
            evidence.state(foreign_state)
    foreign_model = eqiora.compile(
        source="""
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow { derivative(x) + rate * x = 0; }
}
"""
    )
    foreign_field = foreign_model.field(foreign_model.field_ids[0])
    foreign_plan = eqiora.resolve(
        foreign_model,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={foreign_field: 1.0e-11},
        ),
    )
    with pytest.raises(ValueError, match="different exact|occurrence|trajectory"):
        evidence.state(eqiora.State.initial(foreign_plan))
    for wrong_type in (1, object(), result.trajectory):
        with pytest.raises(TypeError):
            evidence.state(wrong_type)


def test_unrelated_common_result_rejects_fsi_evidence() -> None:
    model = eqiora.compile(
        source="""
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow { derivative(x) + rate * x = 0; }
}
"""
    )
    field = model.field(model.field_ids[0])
    plan = eqiora.resolve(
        model,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={field: 1.0e-11},
        ),
    )
    unrelated = eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=0.1,
        output_times_s=(0.1,),
    )
    with pytest.raises(eqiora.CapabilityError):
        eqiora.fsi.evidence(unrelated)
    for wrong_type in (object(), model):
        with pytest.raises(TypeError):
            eqiora.fsi.evidence(wrong_type)


@pytest.mark.parametrize(
    "objective",
    [None, eqiora.solve.Robust, eqiora.solve.Fast, eqiora.solve.LowMemory],
)
def test_manual_and_planned_runs_agree_without_sharing_observation_storage(
    objective: eqiora.solve.SolverPlanningObjective | None,
) -> None:
    model, plan, _, first = solved()
    mesh = plan.mesh
    assert mesh is not None
    selected = (
        plan
        if objective is None
        else eqiora.resolve(
            model,
            mesh=mesh,
            spatial=plan.spatial,
            temporal=plan.temporal,
            solve=eqiora.solve.Linear(
                objective=objective,
                relative_tolerance=plan.solve.relative_tolerance,
                absolute_tolerance=plan.solve.absolute_tolerance,
                maximum_iterations=plan.solve.maximum_iterations,
            ),
        )
    )
    selected_bytes = selected.to_bytes()
    replayed = eqiora.Plan.from_bytes(selected_bytes)
    assert replayed.to_bytes() == selected_bytes
    assert replayed.solve.objective == objective
    assert replayed.solve.planning_policy_id == selected.solve.planning_policy_id
    assert replayed.solve.selected_candidate_id == selected.solve.selected_candidate_id
    assert replayed.solve.selected_evidence_case == selected.solve.selected_evidence_case
    assert replayed.solve.planning_reasons == selected.solve.planning_reasons
    if objective is not None:
        assert replayed.solve.planning_policy_id is not None
        assert replayed.solve.selected_candidate_id is not None
        # Reuse the native FSI reduction falsifier: fast LU is excluded before
        # ranking, so every objective retains the admitted reference MINRES.
        assert (
            "eqiora.faer.sparse-lu-indefinite-identity-fast-f64",
            "profile.required-reduction-mismatch",
        ) in replayed.solve.planning_reasons
    for attribute in ("algorithm", "preconditioner", "reduction", "operator", "provider"):
        assert getattr(replayed.solve, attribute) == getattr(plan.solve, attribute)
    assert replayed.execution.provider == plan.execution.provider
    assert replayed.execution.placement == plan.execution.placement
    second = eqiora.run(
        replayed,
        state=initial(model, mesh, replayed),
        steps=2,
        output_steps=(1, 2),
    )
    first_evidence = eqiora.fsi.evidence(first)
    second_evidence = eqiora.fsi.evidence(second)
    for evidence in (first_evidence, second_evidence):
        for accepted in evidence.states:
            assert accepted.solve.true_residual_norm <= accepted.solve.residual_target
    for field, association in (
        (plan.capability.fluid_velocity, "vertex"),
        (plan.capability.fluid_velocity, "cell"),
        (plan.capability.pressure, "vertex"),
        (plan.fields[2], "vertex"),
        (plan.fields[3], "vertex"),
    ):
        for left_state, right_state in zip(
            first.trajectory.states, second.trajectory.states, strict=True
        ):
            left = left_state.field(field).values(association)
            right = right_state.field(field).values(association)
            np.testing.assert_array_equal(left, right)
            assert not np.shares_memory(left, right)
    for left, right in (
        (first_evidence.fluid_cells, second_evidence.fluid_cells),
        (
            first_evidence.states[-1].fluid_action,
            second_evidence.states[-1].fluid_action,
        ),
    ):
        np.testing.assert_array_equal(left, right)
        assert not np.shares_memory(left, right)


def test_observation_arrays_survive_result_deletion() -> None:
    model, plan, _, result = solved()
    trajectory = result.trajectory
    state = trajectory.states[-1]
    evidence = eqiora.fsi.evidence(result)
    state_evidence = evidence.state(state)
    arrays = (
        trajectory.coordinates,
        trajectory.cells,
        state.field(plan.capability.fluid_velocity).values("vertex"),
        state.field(plan.capability.pressure).values("vertex"),
        state.field(plan.fields[3]).values("vertex"),
        state_evidence.fluid_action,
    )
    del state_evidence, evidence, state, trajectory, result, plan, model
    gc.collect()
    assert all(array.size > 0 and not array.flags.writeable for array in arrays)


def test_numpy_import_is_lazy_until_observation_projection(tmp_path: Path) -> None:
    script = tmp_path / "lazy_common_fsi_projection.py"
    script.write_text(
        """
import sys
from importlib.resources import files
import eqiora

assert "numpy" not in sys.modules
graph = eqiora.geometry.GeometryGraph()
fluid = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
solid = graph.rectangle(x_bounds=(1.0, 2.0), y_bounds=(0.0, 1.0))
partition = graph.partition(fluid, solid, interface=(fluid.boundaries[1], solid.boundaries[0]))
geometry = graph.build(partition, named_topology={
    "fluid": fluid.region, "fluid_x_lower": fluid.boundaries[0],
    "fluid_x_upper": fluid.boundaries[1], "fluid_y_lower": fluid.boundaries[2],
    "fluid_y_upper": fluid.boundaries[3], "solid": solid.region,
    "solid_x_lower": solid.boundaries[0], "solid_x_upper": solid.boundaries[1],
    "solid_y_lower": solid.boundaries[2], "solid_y_upper": solid.boundaries[3],
})
request = eqiora.meshing.AffineTriangleMesher(cells=(2, 2))
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, request))
model = eqiora.compile(
    path=files(eqiora).joinpath("examples", "fixed-reference-fsi.eqi"),
    geometry=geometry, entry="FixedReferenceFsi2d",
    bindings={**{r: geometry.selection(r) for r in ("fluid", "solid")},
              **{r+"_"+side: (geometry.selection(r+"_"+side), geometry.selection(r))
                 for r in ("fluid", "solid") for side in ("x_lower", "x_upper", "y_lower", "y_upper")},"fluid_density": 2.0, "fluid_viscosity": 0.5,
                "solid_density": 3.0, "solid_mu": 4.0,
                "solid_lambda": 2.0, "zero_pressure": 0.0},
)
plan = eqiora.resolve(
    model, mesh=mesh,
    spatial=(eqiora.fem.MiniP1().at(model.domain("fluid")),
             eqiora.fem.P1().at(model.domain("solid"))),
    temporal=eqiora.time.BackwardEuler(step_s=0.05),
    solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.MinimumResidual,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(),
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
        maximum_iterations=20000,
    ),
)
fv, fp, sv, sd = plan.fields
state = eqiora.State.initial(plan, time_s=0.0, fields=(
    eqiora.InitialField(fv, vertex_values=[[0.0, 0.0]] * 6,
                        cell_values=[[0.0, 0.0]] * 4),
    eqiora.InitialField(fp, vertex_values=[0.25] * 6),
    eqiora.InitialField(sv, vertex_values=[[0.0, 0.0]] * 6),
    eqiora.InitialField(
        sd,
        vertex_values=[
            [0.0, 0.0], [0.02, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        ],
    ),
))
result = eqiora.run(plan, state=state, steps=2, output_steps=(1, 2))
evidence = eqiora.fsi.evidence(result)
assert "numpy" not in sys.modules
_ = evidence.fluid_cells
assert "numpy" in sys.modules
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-I", str(script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
