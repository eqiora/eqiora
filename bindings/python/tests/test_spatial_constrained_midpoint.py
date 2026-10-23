"""Physical constraints at collocation stages, endpoints and off-grid observations."""

import eqiora as q
import numpy as np
import pytest


def _problem(
    spatial,
    complex_field,
    mixed_rows,
    *,
    authored_initial=True,
    feedback=False,
    cells=2,
    step_s=0.1,
):
    module = q.Module("main")
    component = module.component("Mixed")
    body = component.volume("body", dimensions=1)
    component.boundary("left", parent=body)
    component.boundary("right", parent=body)
    value_type = q.ValueType.complex() if complex_field else q.ValueType.real()
    fields = [
        component.field(
            name,
            value_type=value_type,
            role=role,
            on=body,
            spatial_regularity=q.SpatialRegularity.Smooth,
        )
        for name, role in [("u", q.FieldRole.State), ("z", q.FieldRole.Variable)]
    ]
    u, z = fields
    initial_u = 1 + 2j if complex_field else 1.0
    # z=|u|² exercises real-linear complex derivatives as well as real constraints.
    if authored_initial:
        component.initial((u, initial_u), (z, abs(initial_u) ** 2))
    rate = q.lang.quantity(1, q.units.s) * q.lang.derivative(u)
    dynamic = rate + z if feedback else rate - 1
    constraint = z - q.lang.math.abs2(u)
    evolution = component.relation(
        "evolution",
        q.lang.equation(dynamic + constraint if mixed_rows else dynamic, 0),
        on=body,
    )
    auxiliary = component.relation(
        "constraint",
        q.lang.equation(dynamic - constraint if mixed_rows else constraint, 0),
        on=body,
    )
    if spatial is q.fem.Q1:
        component.weak_form(
            "mixed",
            tests={
                evolution: component.test("wu", for_=u),
                auxiliary: component.test("wz", for_=z),
            },
        )
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0.0, 1.0))
    geometry = graph.build(
        interval,
        named_topology={
            "body": interval.region,
            "left": interval.boundaries[0],
            "right": interval.boundaries[1],
        },
    )
    support = geometry.selection("body")
    model = q.compile(
        source=module,
        entry="Mixed",
        geometry=geometry,
        bindings={
            "body": support,
            "left": (geometry.selection("left"), support),
            "right": (geometry.selection("right"), support),
        },
    )
    mesh = q.meshing.generate(
        q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(cells,)))
    )
    fields = [model.field("definition." + name) for name in ("u", "z")]
    linear = q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(),
        relative_tolerance=0.0,
        absolute_tolerance=1e-14,
        maximum_iterations=100,
    )
    plan = q.resolve(
        model,
        mesh=mesh,
        spatial=spatial(),
        temporal=q.time.ImplicitMidpoint(
            step_s=step_s,
            relative_tolerance=1e-10,
            absolute_tolerances={
                (field, 0, 0, imaginary): 1e-12
                for field in fields
                for imaginary in ([False, True] if complex_field else [False])
            },
        ),
        solve=q.solve.Newton(
            linear=linear,
            relative_tolerance=0.0,
            absolute_tolerance=1e-12,
            maximum_iterations=16,
            maximum_line_search_steps=16,
        ),
    )
    return plan, fields, initial_u


@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize(
    "spatial,mixed_rows",
    [(q.fem.Q1, False), (q.fem.Q1, True), (q.fvm.CellCenteredTpfa, False)],
)
def test_constrained_midpoint_physical_outputs_and_replay(
    spatial, mixed_rows, complex_field
):
    plan, fields, initial_u = _problem(spatial, complex_field, mixed_rows)
    initial = q.State.initial(plan)
    before = initial.to_bytes()
    times = (0.05, 0.1, 0.15, 0.2, 0.3)
    dense = q.run(plan, state=initial, until_s=0.3, output_times_s=times)
    sparse = q.run(plan, state=initial, until_s=0.3, output_times_s=(0.3,))
    association = "vertex" if spatial is q.fem.Q1 else "cell"
    assert len(dense.trajectory.states) == len(times)
    for time, state in zip(times, dense.trajectory.states, strict=True):
        # Independent exact solution: u=u0+t, z=|u0+t|². In particular the
        # endpoint is not the extrapolated algebraic collocation coordinate.
        for field, expected in zip(
            fields, (initial_u + time, abs(initial_u + time) ** 2), strict=True
        ):
            actual = np.asarray(state.field(field).values(association))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert (
        dense.trajectory.states[-1].to_bytes()
        == sparse.trajectory.states[-1].to_bytes()
    )
    resumed = q.run(
        plan,
        state=dense.trajectory.states[1],
        until_s=0.3,
        output_times_s=(0.3,),
    )
    for field in fields:
        np.testing.assert_allclose(
            resumed.trajectory.states[-1].field(field).values(association),
            dense.trajectory.states[-1].field(field).values(association),
            rtol=0,
            atol=1e-10,
        )
    assert initial.to_bytes() == before
    assert q.Result.from_bytes(plan, dense.to_bytes()).to_bytes() == dense.to_bytes()
