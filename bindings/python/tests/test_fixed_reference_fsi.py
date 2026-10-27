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


def admitted(
    algorithm: eqiora.solve.LinearSolver = eqiora.solve.LinearSolver.MinimumResidual,
    *, faer: bool = False,
) -> tuple[eqiora.Model, eqiora.meshing.Mesh, eqiora.Plan]:
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
            algorithm=algorithm,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=(eqiora.solve.Reduction.Fast if faer else eqiora.solve.Reduction.Reproducible),
            provider=(eqiora.solve.SolverProvider.faer() if faer else eqiora.solve.SolverProvider.reference()),
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
    flow_cell_indices = np.flatnonzero(coordinates[cells, 0].mean(axis=1) < 1.0)
    displacement = np.zeros((solid_vertices.size, 2))
    interface_midpoint = np.flatnonzero(
        (coordinates[solid_vertices, 0] == 1.0)
        & (coordinates[solid_vertices, 1] == 0.5)
    )
    assert interface_midpoint.size == 1
    displacement[interface_midpoint[0], 0] = 0.02
    fluid_velocity = model.field("definition.fluid_velocity")
    fluid_pressure = model.field("definition.fluid_pressure")
    solid_velocity = model.field("definition.solid_velocity")
    solid_displacement = model.field("definition.solid_displacement")
    return eqiora.State.initial(
        plan,
        time_s=0.0,
        fields=(
            eqiora.InitialField(
                fluid_velocity,
                vertex_values=np.zeros((fluid_vertices.size, 2)),
                cell_values=np.zeros((flow_cell_indices.size, 2)),
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


def test_coupled_model_uses_the_ordinary_plan_surface() -> None:
    _, _, plan = admitted()
    assert isinstance(plan.capability, eqiora.SpatialPlanView)
    assert not hasattr(eqiora, "fsi")
    assert not hasattr(eqiora, "FixedReferenceFsiPlanView")


def test_plan_binds_exact_model_mesh_scopes_provider_and_realization() -> None:
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
    assert plan.identity and portable.identity == plan.identity
    assert len(plan.realization_digest) == 64
    assert tuple(binding.domain for binding in plan.spatial) == (
        model.domain("fluid"),
        model.domain("solid"),
    )
    assert plan.temporal.step_s == 0.05
    assert plan.solve.maximum_iterations == 20_000
    assert isinstance(plan.capability, eqiora.SpatialPlanView)
    assert tuple(plan.capability.fields) == plan.fields
    assert not any(
        hasattr(plan.capability, name)
        for name in ("fluid_velocity", "pressure", "solid_velocity", "displacement")
    )
    assert not hasattr(plan.capability, "pressure_gauge")
    assert plan.solve.algorithm == "minimum-residual"
    assert plan.solve.backend == "eqiora.reference"
    assert plan.execution.provider == "eqiora.host.serial"
    assert plan.execution.placement == "host-serial"
    assert len(set(plan.fields)) == 4
    assert set(plan.fields) == {
        model.field("definition.fluid_velocity"),
        model.field("definition.fluid_pressure"),
        model.field("definition.solid_velocity"),
        model.field("definition.solid_displacement"),
    }
    assert portable.identity == plan.identity
    assert portable.to_bytes() == plan_bytes
    assert portable.model.to_bytes() == model.to_bytes()
    assert portable.mesh.to_bytes() == mesh.to_bytes()
    assert {binding.domain.id for binding in portable.spatial} == {
        binding.domain.id for binding in plan.spatial
    }
    assert portable.temporal.step_s == plan.temporal.step_s
    assert portable.requested_solve.maximum_iterations == plan.requested_solve.maximum_iterations
    assert portable.realization_digest == plan.realization_digest


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
        state.field(model.field("definition.fluid_pressure")).values("vertex"), 0.25
    )
    expected_displacement = np.zeros((6, 2))
    expected_displacement[1, 0] = 0.02
    np.testing.assert_array_equal(
        state.field(model.field("definition.solid_displacement")).values("vertex"), expected_displacement
    )
    with pytest.raises(eqiora.ValidationError, match="omits an exact stored Field"):
        eqiora.State.initial(plan, fields=())
    with pytest.raises(eqiora.ValidationError):
        eqiora.State.initial(plan, time_s=0.0, fields=())
    with pytest.raises(TypeError):
        eqiora.InitialField(model.field("definition.fluid_pressure"), values=[0.0])


@pytest.mark.parametrize("algorithm,faer", [
    (eqiora.solve.LinearSolver.MinimumResidual, False),
    (eqiora.solve.LinearSolver.BiConjugateGradientStabilized, False),
    (eqiora.solve.LinearSolver.BiConjugateGradientStabilized, True),
    (eqiora.solve.LinearSolver.SparseLu, True),
])
def test_common_worker_run_outputs_restart_and_observation_evidence(
    algorithm: eqiora.solve.LinearSolver, faer: bool,
) -> None:
    model, mesh, plan = admitted(algorithm, faer=faer)
    assert plan.solve.algorithm == {
        eqiora.solve.LinearSolver.MinimumResidual: "minimum-residual",
        eqiora.solve.LinearSolver.BiConjugateGradientStabilized: "bicgstab",
        eqiora.solve.LinearSolver.SparseLu: "sparse-lu",
    }[algorithm]
    assert plan.solve.backend == ("eqiora.faer" if faer else "eqiora.reference")
    assert plan.solve.reduction == ("fast" if faer else "reproducible")
    assert plan.solve.relative_tolerance == 1e-11
    assert plan.solve.absolute_tolerance == 1e-13
    assert plan.solve.maximum_iterations == 20_000
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
    observations = tuple(result.spatial_step(i) for i in range(len(outputs)))
    assert len(observations) == 2
    for observation, output in zip(observations, outputs, strict=True):
        assert observation.state_identity == output.digest
        assert observation.trajectory_identity == trajectory.digest
        assert observation.solve.algorithm == plan.solve.algorithm
        assert observation.solve.backend == plan.solve.backend
        assert observation.solve.reduction == plan.solve.reduction
        assert observation.solve.maximum_iterations == plan.solve.maximum_iterations
        assert observation.solve.completed_iterations <= plan.solve.maximum_iterations
        assert observation.solve.relative_tolerance == plan.solve.relative_tolerance
        assert observation.solve.absolute_tolerance == plan.solve.absolute_tolerance
        assert observation.solve.true_residual_norm <= observation.solve.residual_target
        assert observation.balance_error is None

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
    last = result.trajectory.states[-1]
    with pytest.raises(eqiora.ValidationError):
        eqiora.run(fresh_plan, state=last, steps=1, output_steps=(1,))
    # A different Plan requires an explicit physical-field import into its State.
    imported = eqiora.State.initial(
        fresh_plan, time_s=last.time_s,
        fields=tuple(eqiora.InitialField(
            snapshot.field,
            vertex_values=snapshot.values("vertex"),
            **({"cell_values": snapshot.values("cell")} if "cell" in snapshot.associations else {}),
        ) for snapshot in last.fields),
    )
    restarted = eqiora.run(fresh_plan, state=imported, steps=1, output_steps=(1,))
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


def test_observations_retain_exact_topology_fields_and_physical_work() -> None:
    model, plan, run, result = solved()
    trajectory = result.trajectory
    mesh = plan.mesh
    assert mesh is not None
    assert run.result() is result and result.trajectory is trajectory
    assert trajectory.coordinates is trajectory.coordinates
    assert trajectory.cells is trajectory.cells
    assert trajectory.coordinates.shape == (9, 2)
    np.testing.assert_array_equal(trajectory.cells, EXPECTED_CELLS)
    domain_cells = {
        model.domain("fluid").id: mesh.selection_entities("fluid"),
        model.domain("solid").id: mesh.selection_entities("solid"),
    }
    assert domain_cells[model.domain("fluid").id] == tuple((2, i) for i in range(4))
    assert domain_cells[model.domain("solid").id] == tuple((2, i) for i in range(4, 8))
    facets = mesh.selection_entities("fluid_x_upper")
    assert facets == mesh.selection_entities("solid_x_lower")
    assert tuple(mesh.entity_vertices(entity) for entity in facets) == ((3, 4), (4, 5))
    assert tuple(state.step for state in trajectory.states) == (1, 2)
    assert tuple(state.time_s for state in trajectory.states) == (0.05, 0.10)
    assert trajectory.state(1) is trajectory.states[0]
    assert trajectory.state(2) is trajectory.states[1]
    with pytest.raises(IndexError):
        trajectory.state(0)

    previous = initial(model, mesh, plan)
    displacements = []
    endpoint_pairs = {
        (model.domain("fluid").id, model.field("definition.fluid_velocity").id),
        (model.domain("solid").id, model.field("definition.solid_velocity").id),
    }
    for output_index, state in enumerate(trajectory.states):
        observation = result.spatial_step(output_index)
        assert observation.output_index == output_index
        assert observation.step_index == state.step
        assert observation.state_identity == state.digest
        assert observation.previous_state_identity == previous.digest
        assert observation.trajectory_identity == trajectory.digest
        assert observation.balance_error is None
        for domain, entities in domain_cells.items():
            assert {term.cell_index for term in observation.volume_terms if term.domain_id == domain} == {
                index for dimension, index in entities
            }
        velocity = state.field(model.field("definition.fluid_velocity"))
        pressure = state.field(model.field("definition.fluid_pressure"))
        displacement = state.field(model.field("definition.solid_displacement"))
        solid_velocity = state.field(model.field("definition.solid_velocity")).values("vertex")
        velocity_vertices = velocity.values("vertex")
        velocity_bubbles = velocity.values("cell")
        pressure_support = pressure.support_indices("vertex")
        pressure_values = pressure.values("vertex")
        displacement_values = displacement.values("vertex")
        displacements.append(displacement_values)
        arrays = (velocity_vertices, velocity_bubbles, pressure_support, pressure_values,
                  displacement_values, solid_velocity, trajectory.coordinates, trajectory.cells)
        assert state.field(model.field("definition.fluid_velocity")) is velocity
        assert velocity.values("vertex") is velocity_vertices
        assert velocity.values("cell") is velocity_bubbles
        assert pressure.support_indices("vertex") is pressure_support
        assert pressure.values("vertex") is pressure_values
        assert displacement.values("vertex") is displacement_values
        assert velocity_vertices.shape == (6, 2)
        assert velocity_bubbles.shape == (4, 2)
        assert pressure_support.shape == pressure_values.shape == (6,)
        assert displacement_values.shape == (6, 2)
        np.testing.assert_array_equal(pressure_support, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(displacement.support_indices("vertex"), [3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(velocity.support_indices("cell"), [0, 1, 2, 3])
        assert all(array.flags.c_contiguous and not array.flags.writeable for array in arrays)
        assert all(np.isfinite(array).all() for array in arrays)

        # Independently observe backward Euler and equality of interface velocity.
        previous_displacement = previous.field(model.field("definition.solid_displacement")).values("vertex")
        np.testing.assert_allclose(displacement_values - previous_displacement,
                                   0.05 * solid_velocity, rtol=0., atol=1e-12)
        np.testing.assert_allclose(velocity_vertices[3:6], solid_velocity[:3], rtol=0., atol=1e-12)
        # On each triangle integral(lambda_i)=A/3 and integral(27*l0*l1*l2)=9*A/20.
        # Integrate MINI velocity divergence against each continuous P1 pressure basis.
        continuity = np.zeros(6)
        for cell_index, vertices in enumerate(EXPECTED_CELLS[:4]):
            points = trajectory.coordinates[vertices]
            affine = np.column_stack((np.ones(3), points))
            area = abs(np.linalg.det(affine)) / 2
            gradients = np.linalg.inv(affine)[1:, :].T
            divergence = np.sum(velocity_vertices[vertices] * gradients)
            continuity[vertices] += area / 3 * divergence - 9 * area / 20 * (gradients @ velocity_bubbles[cell_index])
        np.testing.assert_allclose(continuity, 0., rtol=0., atol=1e-10)

        actions = tuple(term for term in observation.boundary_work if term.kind == "interface")
        assert len(actions) == 4  # Two components at the free midpoint, on both sides.
        assert len({term.connection_id for term in actions}) == 1
        assert all(term.connection_id and term.entity == (0, 4) and term.slot == 0 for term in actions)
        assert {(term.domain_id, term.field_id) for term in actions} == endpoint_pairs
        assert {term.component for term in actions} == {0, 1}
        assert all(term.action_dimension == eqiora.Dimension(mass=1, time=-2) for term in actions)
        assert all(term.work_dimension == eqiora.Dimension(mass=1, length=1, time=-2) for term in actions)
        for component in (0, 1):
            assert abs(sum(term.action for term in actions if term.component == component)) < 1e-10
        assert all(np.isfinite(term.action) and np.isfinite(term.work) for term in actions)
        with pytest.raises(AttributeError):
            actions[0].work = 0.
        with pytest.raises(AttributeError):
            observation.volume_terms[0].previous = 0.
        assert observation.solve.algorithm == "minimum-residual"
        assert observation.solve.preconditioner == "identity"
        assert observation.solve.reduction == "reproducible"
        assert observation.solve.true_residual_norm <= observation.solve.residual_target
        assert all(count > 0 for count in observation.assembly_counts)
        assert {term.role for term in observation.volume_terms} >= {"storage", "state-potential", "dissipation"}
        assert all(np.isfinite([term.previous, term.current, term.increment, term.work]).all()
                   for term in observation.volume_terms)
        assert all(np.isfinite(defect.value) for defect in observation.defects)
        previous = state
    assert not np.array_equal(displacements[0], displacements[1])


def test_step_observations_are_bound_to_their_result_and_output_position() -> None:
    _, plan, _, result = solved()
    replayed = eqiora.Result.from_bytes(plan, result.to_bytes())
    for index, state in enumerate(result.trajectory.states):
        observation = result.spatial_step(index)
        copied = replayed.spatial_step(index)
        assert observation.result_identity == copied.result_identity
        assert observation.state_identity == state.digest == copied.state_identity
        assert observation.trajectory_identity == result.trajectory.digest
    # A State cannot be used as an occurrence lookup key in another Result.
    for wrong_type in (result.trajectory.states[0], object(), result.trajectory):
        with pytest.raises(TypeError):
            result.spatial_step(wrong_type)
    with pytest.raises(IndexError):
        result.spatial_step(len(result.trajectory.states))


def test_unrelated_common_result_rejects_spatial_step_observations() -> None:
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
            absolute_tolerances={(field, 0, 0, False): 1.0e-11},
        ),
    )
    unrelated = eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=0.1,
        output_times_s=(0.1,),
    )
    with pytest.raises(eqiora.CapabilityError):
        unrelated.spatial_step(0)


@pytest.mark.parametrize(
    "objective,faer_algorithm",
    [(None, None), (eqiora.solve.Robust, None), (eqiora.solve.Fast, None),
     (eqiora.solve.LowMemory, None),
     (None, eqiora.solve.LinearSolver.BiConjugateGradientStabilized),
     (None, eqiora.solve.LinearSolver.SparseLu)],
)
def test_manual_and_planned_runs_agree_without_sharing_observation_storage(
    objective: eqiora.solve.SolverPlanningObjective | None,
    faer_algorithm: eqiora.solve.LinearSolver | None,
) -> None:
    model, plan, _, first = solved()
    mesh = plan.mesh
    assert mesh is not None
    selected = (
        plan
        if objective is None and faer_algorithm is None
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
            ) if faer_algorithm is None else eqiora.solve.Linear(
                algorithm=faer_algorithm,
                provider=eqiora.solve.SolverProvider.faer(),
                preconditioner=eqiora.solve.Preconditioner.Identity,
                reduction=eqiora.solve.Reduction.Fast,
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
    for attribute in ("algorithm", "preconditioner", "reduction", "operator", "provider"):
        assert getattr(replayed.solve, attribute) == getattr(selected.solve, attribute)
    assert replayed.execution.provider == selected.execution.provider
    assert replayed.execution.placement == selected.execution.placement
    second = eqiora.run(
        replayed,
        state=initial(model, mesh, replayed),
        steps=2,
        output_steps=(1, 2),
    )
    for result in (first, second):
        for index in range(len(result.trajectory.states)):
            accepted = result.spatial_step(index)
            assert accepted.solve.true_residual_norm <= accepted.solve.residual_target
    for field, association in (
        (model.field("definition.fluid_velocity"), "vertex"),
        (model.field("definition.fluid_velocity"), "cell"),
        (model.field("definition.fluid_pressure"), "vertex"),
        (model.field("definition.solid_velocity"), "vertex"),
        (model.field("definition.solid_displacement"), "vertex"),
    ):
        for left_state, right_state in zip(
            first.trajectory.states, second.trajectory.states, strict=True
        ):
            left = left_state.field(field).values(association)
            right = right_state.field(field).values(association)
            np.testing.assert_allclose(left, right, rtol=0., atol=1e-9)
            assert not np.shares_memory(left, right)
    for index in range(2):
        left = first.spatial_step(index)
        right = second.spatial_step(index)
        assert left is not right
        left_actions = {(t.domain_id, t.field_id, t.entity, t.component): t.action
                        for t in left.boundary_work if t.kind == "interface"}
        right_actions = {(t.domain_id, t.field_id, t.entity, t.component): t.action
                         for t in right.boundary_work if t.kind == "interface"}
        assert left_actions.keys() == right_actions.keys()
        np.testing.assert_allclose(list(left_actions.values()), list(right_actions.values()),
                                   rtol=0., atol=1e-9)


def test_observation_arrays_survive_result_deletion() -> None:
    model, plan, _, result = solved()
    trajectory = result.trajectory
    state = trajectory.states[-1]
    observation = result.spatial_step(1)
    arrays = (
        trajectory.coordinates,
        trajectory.cells,
        state.field(model.field("definition.fluid_velocity")).values("vertex"),
        state.field(model.field("definition.fluid_pressure")).values("vertex"),
        state.field(model.field("definition.solid_displacement")).values("vertex"),
    )
    terms = observation.boundary_work
    del observation, state, trajectory, result, plan, model
    gc.collect()
    assert all(array.size > 0 and not array.flags.writeable for array in arrays)
    assert terms and all(np.isfinite(term.work) for term in terms)


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
fv = model.field("definition.fluid_velocity")
fp = model.field("definition.fluid_pressure")
sv = model.field("definition.solid_velocity")
sd = model.field("definition.solid_displacement")
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
observation = result.spatial_step(0)
assert observation.volume_terms
assert mesh.selection_entities("fluid")
assert "numpy" not in sys.modules
_ = result.trajectory.cells
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
