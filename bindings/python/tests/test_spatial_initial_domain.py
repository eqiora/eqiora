"""Explicit initial support uses the ordinary coordinate and State owners."""

import eqiora as q
import numpy as np
import pytest


@pytest.mark.parametrize("spatial", [q.fem.Q1, q.fvm.CellCenteredTpfa])
@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize("emitted", [False, True])
def test_supported_coordinate_initialization_executes_and_replays(
    spatial, complex_field, emitted
):
    module = q.Module("main")
    component = module.component("Profile")
    body = component.volume("body", dimensions=1)
    left = component.boundary("left", parent=body)
    right = component.boundary("right", parent=body)
    value_type = q.ValueType.complex() if complex_field else q.ValueType.real()
    u = component.field(
        "u",
        value_type=value_type,
        role=q.FieldRole.State,
        on=body,
        spatial_regularity=q.SpatialRegularity.Smooth,
    )
    multiplier = 1 + 2j if complex_field else 1.0
    x = q.lang.coordinate(0) / q.lang.quantity(1, q.units.m)
    component.initial((u, multiplier * (1 + x)), on=body)
    component.relation(
        "flow",
        q.lang.equation(
            q.lang.derivative(u)
            - q.lang.quantity(1, q.units.m**2 / q.units.s) * q.lang.div(q.lang.grad(u)),
            0,
        ),
        on=body,
    )
    component.relation(
        "left_value", q.lang.equation(q.lang.trace(u), multiplier), on=left
    )
    component.relation(
        "right_value", q.lang.equation(q.lang.trace(u), 2 * multiplier), on=right
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
    selection = geometry.selection("body")
    source = module.to_eqi() if emitted else module
    if emitted:
        assert "initial on body" in source
    model = q.compile(
        source=source,
        entry="Profile",
        geometry=geometry,
        bindings={
            "body": selection,
            "left": (geometry.selection("left"), selection),
            "right": (geometry.selection("right"), selection),
        },
    )
    model = q.Model.from_bytes(model.to_bytes())
    mesh = q.meshing.generate(
        q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2,)))
    )
    plan = q.resolve(
        model,
        mesh=mesh,
        spatial=spatial(),
        temporal=q.time.BackwardEuler(0.1),
        solve=q.solve.Linear(
            objective=q.solve.Robust,
            relative_tolerance=1e-12,
            absolute_tolerance=1e-14,
            maximum_iterations=100,
        ),
    )
    plan = q.Plan.from_bytes(plan.to_bytes())
    initial = q.State.initial(plan)
    before = initial.to_bytes()
    association = "vertex" if spatial is q.fem.Q1 else "cell"
    # Nodal values or centroid samples of c*(1+x); its Laplacian vanishes.
    expected = multiplier * np.array(
        [1.0, 1.5, 2.0] if spatial is q.fem.Q1 else [1.25, 1.75]
    )
    np.testing.assert_allclose(
        initial.fields[0].values(association), expected, rtol=0, atol=1e-13
    )
    result = q.run(plan, state=initial, steps=1, output_steps=(1,))
    np.testing.assert_allclose(
        result.trajectory.states[-1].fields[0].values(association),
        expected,
        rtol=0,
        atol=1e-11,
    )
    assert initial.to_bytes() == before
    assert q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()


@pytest.mark.parametrize(
    ("initial", "diagnostic"),
    [
        ("initial on missing {u=0;}", "unresolved initialization Domain"),
        ("initial on other {u=0;}", "residual support.*differs from Relation scope"),
        (
            "initial on body {u=coordinate(1)/1[m];}",
            "coordinate axis 1 is outside Domain dimension 1",
        ),
        (
            "initial {u=coordinate(0)/1[m];}",
            "coordinate operator requires a Cartesian Relation scope",
        ),
    ],
)
def test_initial_support_rejects_foreign_or_invalid_coordinates(initial, diagnostic):
    source = f"""model Invalid() {{
        domain body=box(0,1); domain other=box(0,1);
        state u:1 on body in smooth;
        {initial}
        relation flow on body {{derivative(u)=0[1/s];}}
    }}"""
    with pytest.raises(q.ValidationError, match=diagnostic):
        q.compile(source=source)


def test_initial_support_must_belong_to_the_component():
    module = q.Module("main")
    first = module.component("First")
    other = module.component("Other")
    foreign = other.volume("body", dimensions=1)
    with pytest.raises(
        q.lang.ModuleError, match="support must belong to this Component"
    ):
        first.initial((0, 0), on=foreign)
