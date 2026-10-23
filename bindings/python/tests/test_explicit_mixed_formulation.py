"""Explicit mixed rows share ordinary source/Python spatial execution."""

import eqiora as e
import pytest
import numpy as np


@pytest.mark.parametrize("authored", [False, True, "derive"])
@pytest.mark.parametrize("emitted", [False, True])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_mixed_constant_state_executes_and_replays(authored, emitted, nonlinear):
    q = e.lang
    module = e.Module("main")
    c = module.component("Mixed")
    body = c.volume("body", dimensions=1)
    left = c.boundary("left", parent=body)
    right = c.boundary("right", parent=body)
    u = c.field(
        "u",
        value_type=e.ValueType.real(),
        role=e.FieldRole.State,
        on=body,
        spatial_regularity=e.SpatialRegularity.Smooth,
    )
    v = c.field(
        "v",
        value_type=e.ValueType.real(),
        role=e.FieldRole.Variable,
        on=body,
        spatial_regularity=e.SpatialRegularity.Smooth,
    )
    D = q.quantity(1, e.units.m**2 / e.units.s)
    ell2 = q.quantity(1, e.units.m**2)
    auxiliary_value = u + u**3 if nonlinear else 0
    initial_v = 2 if nonlinear else 0
    c.initial((u, 1), (v, initial_v))
    evolution = c.relation(
        "evolution", q.equation(q.derivative(u) - D * q.div(q.grad(v)), 0), on=body
    )
    auxiliary = c.relation(
        "auxiliary",
        q.equation(v - auxiliary_value + ell2 * q.div(q.grad(u)), 0),
        on=body,
    )
    for name, boundary in [("left", left), ("right", right)]:
        for fieldname, field in [("u", u), ("v", v)]:
            c.relation(
                name + "_" + fieldname,
                q.equation(q.normal(q.grad(field)), 0),
                on=boundary,
            )
    if authored:
        wu = c.test("wu", for_=u)
        wv = c.test("wv", for_=v)
        if authored == "derive":
            c.weak_form("mixed", tests={evolution: wu, auxiliary: wv})
        else:
            c.weak_form(
                "mixed",
                equations={
                    evolution: (
                        q.integrate(
                            body,
                            wu * q.derivative(u) + D * q.dot(q.grad(wu), q.grad(v)),
                        ),
                        0,
                    ),
                    auxiliary: (
                        q.integrate(
                            body,
                            wv * (v - auxiliary_value)
                            - ell2 * q.dot(q.grad(wv), q.grad(u)),
                        ),
                        0,
                    ),
                },
            )
    graph = e.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0.0, 1.0))
    geometry = graph.build(
        interval,
        named_topology={
            "body": interval.region,
            "left": interval.boundaries[0],
            "right": interval.boundaries[1],
        },
    )
    b = geometry.selection("body")
    source = module.to_eqi() if emitted else module
    model = e.compile(
        source=source,
        entry="Mixed",
        geometry=geometry,
        bindings={
            "body": b,
            "left": (geometry.selection("left"), b),
            "right": (geometry.selection("right"), b),
        },
    )
    mesh = e.meshing.generate(
        e.meshing.resolve(geometry, e.meshing.CartesianMesher(cells=(2,)))
    )
    linear = e.solve.Linear(
        algorithm=e.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=e.solve.Preconditioner.Identity,
        reduction=e.solve.Reduction.Reproducible,
        provider=e.solve.SolverProvider.reference(),
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        maximum_iterations=100,
    )
    plan = e.resolve(
        model,
        mesh=mesh,
        spatial=e.fem.Q1(),
        temporal=e.time.BackwardEuler(0.1),
        solve=e.solve.Newton(
            linear=linear,
            relative_tolerance=0.0,
            absolute_tolerance=1e-11,
            maximum_iterations=16,
            maximum_line_search_steps=16,
        )
        if nonlinear
        else linear,
    )
    result = e.run(plan, state=e.State.initial(plan), steps=1, output_steps=(1,))
    expected = {
        model.field("definition.u").id: 1.0,
        model.field("definition.v").id: float(initial_v),
    }
    assert len(result.trajectory.states) == 1
    for snapshot in result.trajectory.states[0].fields:
        np.testing.assert_allclose(
            snapshot.values("vertex"), expected[snapshot.field.id], rtol=0, atol=1e-12
        )
    assert e.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
