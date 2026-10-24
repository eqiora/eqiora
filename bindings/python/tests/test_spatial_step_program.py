"""Ordinary Run-bound Programs retain a fixed physical predecessor."""
import eqiora as q
import numpy as np
import pytest

from test_nonlinear_spatial_differentiation import linear_policy, newton_policy


def step_plan(nonlinear, complex_field):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    scalar = "complex<1>" if complex_field else "1"
    phase = "math.complex(1,2)" if complex_field else "1"
    reaction = f"{phase}*(h/({phase}))^3" if nonlinear else "h"
    source = f"""public model Main(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        parameter p:1=2; state h:{scalar} on body in h1;
        initial {{h=0;}}
        relation evolution on body {{1[s]*derivative(h)+{reaction}=({phase})*p;}}
    }}"""
    model = q.compile(source=source, entry="Main", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2,))))
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), temporal=q.time.BackwardEuler(1.),
                     solve=newton_policy() if nonlinear else linear_policy())
    plan = q.Plan.from_bytes(plan.to_bytes())
    return plan, plan.model.parameter(model.parameter("p").id)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("complex_field", [False, True])
def test_spatial_step_program_products_and_collection(nonlinear, complex_field):
    plan, parameter = step_plan(nonlinear, complex_field)
    state = q.State.initial(plan, time_s=0.)
    before = state.to_bytes()
    options = dict(inputs=(parameter,), outputs=(plan.fields[0],), state=state)
    program = q.diff.compile(plan, **options, steps=1, output_steps=(1,))
    physical = q.diff.compile(plan, **options, until_s=1., output_times_s=(1.,))
    phase = 1+2j if complex_field else 1.
    p, derivative = (10., 1/13) if nonlinear else (4., .5)
    # h=phase*v, v_previous=0: v+v^3=p or 2v=p.
    point = program.evaluate((p,))
    np.testing.assert_allclose(point.primal().output[0], [2*phase]*3, rtol=0, atol=1e-10)
    np.testing.assert_allclose(physical.evaluate((p,)).primal().output[0], [2*phase]*3,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(point.jvp((1.,)).tangent[0], [phase*derivative]*3,
                               rtol=0, atol=1e-10)
    weight = 1+3j if complex_field else 1.
    pairing = 7. if complex_field else 1.
    np.testing.assert_allclose(point.vjp((np.full(3, weight),)).input_cotangent,
                               [3*pairing*derivative], rtol=0, atol=1e-10)
    collection = program.map((np.array([2., p, p]),), point_shape=(3,)).execute_recorded()
    encoded = collection.to_bytes()
    reopened = q.EvaluationCollection.from_bytes(encoded)
    assert reopened.to_bytes() == encoded
    for member, value in zip(reopened[0], [phase, 2*phase, 2*phase], strict=True):
        np.testing.assert_allclose(member.result.output[0], [value]*3, rtol=0, atol=1e-10)
    shared = program.map((), point_shape=(2,), shared_inputs=(parameter,), shared=(p,)).execute()
    np.testing.assert_allclose(shared.vjp((np.full((2, 3), weight),)).shared_cotangents[0],
                               6*pairing*derivative, rtol=0, atol=1e-10)
    assert state.to_bytes() == before


def test_spatial_step_program_requires_one_exact_run():
    plan, parameter = step_plan(False, False)
    options = dict(inputs=(parameter,), outputs=(plan.fields[0],))
    state = q.State.initial(plan, time_s=0.)
    with pytest.raises(TypeError):
        q.diff.compile(plan, **options, steps=1, output_steps=(1,))
    with pytest.raises(TypeError):
        q.diff.compile(plan, **options, state=state)
    with pytest.raises(TypeError):
        q.diff.compile(plan, **options, state=state, steps=1, output_steps=(1,),
                       until_s=1., output_times_s=(1.,))
    with pytest.raises(q.EqioraError):
        q.diff.compile(plan, **options, state=state, steps=2, output_steps=(2,))
