"""Explicit row choices resolve a cyclic ambiguity through ordinary execution."""

import eqiora as e
import numpy as np
import pytest


def cyclic_plan(authored, emitted, permuted):
    q = e.lang
    module = e.Module("main")
    component = module.component("Cycle")
    body = component.volume("body", dimensions=1)
    sides = [component.boundary(name, parent=body) for name in ("left", "right")]
    names = ["alpha", "beta", "gamma"] if not permuted else ["zeta", "first", "middle"]
    order = [0, 1, 2] if not permuted else [2, 0, 1]
    fields = {
        i: component.field(
            names[i],
            value_type=e.ValueType.real(),
            role=e.FieldRole.Variable,
            on=body,
            spatial_regularity=e.SpatialRegularity.Smooth,
        )
        for i in order
    }
    scale = q.quantity(1, e.units.m**2)
    flux = {
        i: scale * (q.grad(fields[(i + 1) % 3]) + q.grad(fields[(i + 2) % 3]))
        for i in range(3)
    }
    relations = {
        i: component.relation(
            f"balance_{i}", q.equation(fields[i] - q.div(flux[i]), i + 1), on=body
        )
        for i in order
    }
    for side, boundary in enumerate(sides):
        for i in order:
            component.relation(
                f"flux_{side}_{i}", q.equation(q.normal(flux[i]), 0), on=boundary
            )
    if authored:
        tests = {
            i: component.test(f"test_{i}", for_=fields[i]) for i in reversed(order)
        }
        if authored == "derive":
            component.weak_form(
                "explicit_cycle", tests={relations[i]: tests[i] for i in order}
            )
        else:
            component.weak_form(
                "explicit_cycle",
                equations=dict(
                    zip(
                        [relations[i] for i in order],
                        [
                            (
                                q.integrate(
                                    body,
                                    tests[i] * fields[i]
                                    + q.dot(q.grad(tests[i]), flux[i]),
                                ),
                                q.integrate(body, tests[i] * (i + 1)),
                            )
                            for i in order
                        ],
                        strict=True,
                    )
                ),
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
    parent = geometry.selection("body")
    model = e.compile(
        source=module.to_eqi() if emitted else module,
        entry="Cycle",
        geometry=geometry,
        bindings={
            "body": parent,
            "left": (geometry.selection("left"), parent),
            "right": (geometry.selection("right"), parent),
        },
    )
    mesh = e.meshing.generate(
        e.meshing.resolve(geometry, e.meshing.CartesianMesher(cells=(2,)))
    )
    linear = e.solve.Linear(
        algorithm=e.solve.LinearSolver.SparseLu,
        preconditioner=e.solve.Preconditioner.Identity,
        reduction=e.solve.Reduction.Fast,
        provider=e.solve.SolverProvider.faer(),
        relative_tolerance=1e-12,
        absolute_tolerance=1e-14,
        maximum_iterations=100,
    )
    plan = e.resolve(model, mesh=mesh, spatial=e.fem.Q1(), solve=linear)
    expected = [(model.field(f"definition.{names[i]}"), float(i + 1)) for i in range(3)]
    return plan, expected


@pytest.mark.parametrize("control", [True, "derive"])
@pytest.mark.parametrize("emitted", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_explicit_cycle_executes_where_principal_inference_is_ambiguous(
    control, emitted, permuted
):
    with pytest.raises(e.ValidationError, match="principal"):
        cyclic_plan(False, emitted, permuted)
    plan, expected = cyclic_plan(control, emitted, permuted)
    result = e.run(plan)
    for field, value in expected:
        np.testing.assert_allclose(
            result.output(field).values("vertex").numpy(), value, rtol=0, atol=1e-11
        )
    assert e.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
