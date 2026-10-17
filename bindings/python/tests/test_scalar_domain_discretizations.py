"""Ordinary real/complex Field execution across the Cartesian discretizations."""

import eqiora
import pytest


@pytest.mark.parametrize("channels", [False, True])
@pytest.mark.parametrize("complex_domain", [False, True])
@pytest.mark.parametrize("natural_right", [False, True])
@pytest.mark.parametrize("method, association", [(eqiora.fem.Q1, "vertex"), (eqiora.fvm.CellCenteredTpfa, "cell")])
def test_reaction_diffusion_preserves_field_values_and_replay(channels, complex_domain, natural_right, method, association):
    q = eqiora.lang
    module = eqiora.Module("main")
    component = module.component("ReactionDiffusion")
    body = component.volume("body", dimensions=1)
    left = component.boundary("left", parent=body)
    right = component.boundary("right", parent=body)
    value_type = eqiora.ValueType.complex if complex_domain else eqiora.ValueType.real
    scalar = value_type()

    def field_type(dimension=None):
        element = value_type(dimension)
        return eqiora.ValueType.array(element, 2) if channels else element

    def components(first, second):
        return (first, second) if channels else first

    def trace_value(first, second):
        return q.array((first, second)) if channels else first

    a = component.parameter("a", value_type=value_type(eqiora.Dimension(length=2)))
    reaction = component.parameter("q", value_type=scalar)
    load = component.parameter("f", value_type=field_type())
    slope = component.parameter("s", value_type=field_type(eqiora.Dimension(length=-1)))
    flux = component.parameter("g", value_type=field_type(eqiora.Dimension(length=1)))
    u = component.field("u", value_type=field_type(), role=eqiora.FieldRole.Variable,
                        spatial_regularity=eqiora.SpatialRegularity.Smooth, on=body)
    component.relation("balance", q.equation(-q.div(a*q.grad(u)) + reaction*u,
                                             load + slope*q.coordinate(0)), on=body)
    component.relation("left_value", q.equation(q.trace(u), trace_value(1+1j, -2+3j) if complex_domain else trace_value(1., -2.)), on=left)
    right_law = q.equation(q.normal(a*q.grad(u)), flux) if natural_right else q.equation(q.trace(u), trace_value(3+0j, -1+5j) if complex_domain else trace_value(3., -1.))
    component.relation("right_value", right_law, on=right)
    graph = eqiora.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={"body": interval.region,
                           "left": interval.boundaries[0], "right": interval.boundaries[1]})
    bindings = {
        "body": geometry.selection("body"),
        "left": (geometry.selection("left"), geometry.selection("body")),
        "right": (geometry.selection("right"), geometry.selection("body")),
        "a": 2+3j if complex_domain else 2.,
        "q": 3-2j if complex_domain else 3.,
        "f": components(5+1j, 13j) if complex_domain else components(3., -6.),
        "s": components(4-7j, 7+4j) if complex_domain else components(6., 3.),
        "g": components(7+4j, -4+7j) if complex_domain else components(4., 2.),
    }
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(8,))))
    solve = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=128,
    )
    identities = []
    for source in (module, module.to_eqi()):
        model = eqiora.compile(source=source, entry="ReactionDiffusion", geometry=geometry, bindings=bindings)
        plan = eqiora.resolve(model, mesh=mesh, spatial=method(), solve=solve)
        identities.append(plan.identity)
        restored = eqiora.Plan.from_bytes(plan.to_bytes())
        assert restored.to_bytes() == plan.to_bytes()
        field, = restored.fields
        result = eqiora.run(restored)
        output = result.output(field)
        assert output.field == field
        assert output.value_type == field_type()
        assert output.value_type.array_rank == int(channels)
        assert output.value_type.frame == "invariant"
        values = output.values(association)
        entities = 9 if association == "vertex" else 8
        width = 2 if channels else 1
        assert output.associations == (association,)
        assert output.coefficient_count(association) == entities
        assert output.logical_shape(association) == ((entities, 2) if channels else (entities,))
        assert values.shape == (entities * width,)
        assert values.dtype == ("complex128" if complex_domain else "float64")
        assert values.readonly
        # u0=1+2x+i(1-x): q*u0=5+4x+i(1-7x), a*u0'=7+4i.
        # u1=-2+x+i(3+2x): q*u1=7x+i(13+4x), a*u1'=-4+7i.
        # Real specializations use q=3, a=2 and omit the imaginary parts.
        for entity in range(entities):
            x = (entity + (0. if association == "vertex" else .5)) / 8
            expected = (1 + 2*x + (1j*(1-x) if complex_domain else 0.),
                        -2 + x + (1j*(3+2*x) if complex_domain else 0.))
            for channel in range(width):
                value = values[entity * width + channel]
                assert type(value) is (complex if complex_domain else float)
                assert abs(value - expected[channel]) < 1e-10
        replayed = eqiora.Result.from_bytes(restored, result.to_bytes())
        assert replayed.to_bytes() == result.to_bytes()
        assert replayed.output(field).value_type == output.value_type
        assert tuple(replayed.output(field).values(association)) == tuple(values)
    assert identities[0] == identities[1]
