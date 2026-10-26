"""Ordinary Run-bound Programs retain a fixed physical predecessor."""
import eqiora as q
import numpy as np
import pytest

from test_nonlinear_spatial_differentiation import linear_policy, newton_policy


def step_plan(nonlinear, complex_field, *, observables=False, observable_output="total", midpoint=False):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    scalar = "complex<1>" if complex_field else "1"
    phase = "math.complex(1,2)" if complex_field else "1"
    reaction = f"{phase}*(h/({phase}))^3" if nonlinear else "h"
    observations = ""
    if observables:
        amount_type = "complex<m>" if complex_field else "m"
        amount = "math.real(amount)" if complex_field else "amount"
        energy = "math.abs2(h)" if complex_field else "h*h"
        observations = f"""
        observable amount:{amount_type}=integral(h*time()/1[s],measure(body));
        observable total:1={amount}/1[m]+p*time()/1[s];
        observable energy:m=integral({energy},measure(body));
        """
    source = f"""public model Main(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        parameter p:1=2; state h:{scalar} on body in h1;
        initial {{h=0;}}
        relation evolution on body {{1[s]*derivative(h)+{reaction}=({phase})*p;}}
        {observations}
    }}"""
    model = q.compile(source=source, entry="Main", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2,))))
    temporal = (q.time.ImplicitMidpoint(
        step_s=1., relative_tolerance=1e-10,
        absolute_tolerances={(model.field("h"), 0, 0, imaginary): 1e-12
                             for imaginary in ((False, True) if complex_field else (False,))},
    ) if midpoint else q.time.BackwardEuler(1.))
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), temporal=temporal,
                     solve=newton_policy() if nonlinear else linear_policy())
    plan = q.Plan.from_bytes(plan.to_bytes())
    if observables:
        return plan, plan.model.parameter(model.parameter("p").id), tuple(
            plan.model.observable(model.observable(name).id) for name in (observable_output, "energy"))
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
    np.testing.assert_allclose(point.primal().output[0], [[2*phase]*3], rtol=0, atol=1e-10)
    np.testing.assert_allclose(physical.evaluate((p,)).primal().output[0], [[2*phase]*3],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(point.jvp((1.,)).tangent[0], [[phase*derivative]*3],
                               rtol=0, atol=1e-10)
    weight = 1+3j if complex_field else 1.
    pairing = 7. if complex_field else 1.
    np.testing.assert_allclose(point.vjp((np.full((1, 3), weight),)).input_cotangent,
                               [3*pairing*derivative], rtol=0, atol=1e-10)
    collection = program.map((np.array([2., p, p]),), point_shape=(3,)).execute_recorded()
    encoded = collection.to_bytes()
    reopened = q.EvaluationCollection.from_bytes(encoded)
    assert reopened.to_bytes() == encoded
    for member, value in zip(reopened[0], [phase, 2*phase, 2*phase], strict=True):
        np.testing.assert_allclose(member.result.output[0], [[value]*3], rtol=0, atol=1e-10)
    shared = program.map((), point_shape=(2,), shared_inputs=(parameter,), shared=(p,)).execute()
    np.testing.assert_allclose(shared.vjp((np.full((2, 1, 3), weight),)).shared_cotangents[0],
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
    trajectory = q.diff.compile(plan, **options, state=state, steps=2, output_steps=(2,))
    assert trajectory.outputs[0].shape == (1, 3)
    assert trajectory.outputs[0].sample_times_s == (2.,)
    assert len(trajectory.primal().evidence.steps) == 2


def held_step_program(nonlinear, complex_field, advanced):
    """A framework fixture with an exact held zero or accepted unit predecessor."""
    plan, parameter = step_plan(nonlinear, complex_field)
    state = q.State.initial(plan, time_s=0.)
    if advanced:
        # Default p=2 gives v=1 from v_previous=0 in either equation.
        result = q.run(plan, state=state, steps=1, output_steps=(1,))
        state = q.State.from_bytes(plan, result.trajectory.states[0].to_bytes())
        phase = 1+2j if complex_field else 1.
        np.testing.assert_allclose(state.fields[0].values("vertex"), [phase]*3,
                                   rtol=0, atol=1e-10)
    program = q.diff.compile(plan, inputs=(parameter,), outputs=(plan.fields[0],),
                             state=state, steps=1, output_steps=(1,))
    return program, state

def trajectory_reference(parameter, nonlinear, complex_field, observable_output=None):
    """Closed cubic inverse and implicit derivative, independent of solver output.

    Uniform h=phase*v reduces the three BE steps to v+v^3=p+v_previous
    (or 2*v=p+v_previous). The cubic is strictly increasing on the real
    line. Its hyperbolic closed form selects its unique real root; no Newton
    iteration or Eqiora differentiation is used by this reference.
    """
    previous, derivative = 0., 0.
    values, derivatives = [], []
    for _ in range(3):
        rhs = parameter + previous
        previous = (2 / np.sqrt(3) * np.sinh(np.arcsinh(3*np.sqrt(3)*rhs/2)/3)
                    if nonlinear else rhs/2)
        derivative = (1 + derivative) / (1 + 3*previous**2 if nonlinear else 2)
        values.append(previous)
        derivatives.append(derivative)
    phase = 1+2j if complex_field else 1.
    field, tangent = tuple(np.repeat(np.asarray(v)[[0, 2], None], 3, axis=1)*phase
                           for v in (values, derivatives))
    if observable_output is None:
        return field, tangent
    times = np.array([1., 3.])
    if observable_output == "amount":
        return times*field[:, 0], times*tangent[:, 0]
    assert observable_output == "total"
    return times*(field[:, 0].real+parameter), times*(tangent[:, 0].real+1)


def trajectory_program(nonlinear, complex_field, observable_output):
    if observable_output is None:
        plan, parameter = step_plan(nonlinear, complex_field)
        output, quadrature = plan.fields[0], None
    else:
        plan, parameter, (output, _) = step_plan(
            nonlinear, complex_field, observables=True, observable_output=observable_output)
        quadrature = 2
    return q.diff.compile(plan, inputs=(parameter,), outputs=(output,),
        state=q.State.initial(plan, time_s=0.), steps=3, output_steps=(1, 3),
        quadrature_points=quadrature)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("complex_field", [False, True])
def test_trajectory_program_keeps_sample_axis_all_steps_and_shared_reverse(complex_field, nonlinear):
    plan, parameter = step_plan(nonlinear, complex_field)
    state = q.State.initial(plan, time_s=0.)
    program = q.diff.compile(plan, inputs=(parameter,), outputs=(plan.fields[0],),
                             state=state, steps=3, output_steps=(1, 3))
    assert program.outputs[0].shape == (2, 3)
    assert program.outputs[0].sample_times_s == (1., 3.)
    assert program.inputs[0].sample_times_s is None
    point = program.evaluate((4.,))
    expected, derivative = trajectory_reference(4., nonlinear, complex_field)
    np.testing.assert_allclose(point.primal().output[0], expected, rtol=0, atol=1e-10)
    forward = point.jvp((1.,))
    np.testing.assert_allclose(forward.tangent[0], derivative, rtol=0, atol=1e-10)
    assert len(forward.evidence.steps) == 3
    assert all(step.derivative_solve is not None for step in forward.evidence.steps)
    weight = 1+3j if complex_field else 1.
    pairing = 7. if complex_field else 1.
    np.testing.assert_allclose(point.vjp((np.full((2, 3), weight),)).input_cotangent,
                               [derivative.real.sum()*pairing], rtol=0, atol=1e-10)
    collection = program.map((np.array([4., 4.]),), point_shape=(2,)).execute_recorded()
    encoded = collection.to_bytes()
    reopened = q.EvaluationCollection.from_bytes(encoded)
    assert reopened.to_bytes() == encoded
    for member in reopened[0]:
        np.testing.assert_allclose(member.result.output[0], expected, rtol=0, atol=1e-10)
        assert len(member.result.evidence.steps) == 3
    shared = program.map((), point_shape=(2,), shared_inputs=(parameter,), shared=(4.,)).execute()
    np.testing.assert_allclose(shared.vjp((np.full((2, 2, 3), weight),)).shared_cotangents[0],
                               2*derivative.real.sum()*pairing, rtol=0, atol=1e-10)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize("observable_output", ["total", "amount"])
def test_trajectory_observables_keep_time_direct_terms_and_replay(nonlinear, complex_field, observable_output):
    plan, parameter, (total, energy) = step_plan(
        nonlinear, complex_field, observables=True, observable_output=observable_output)
    state = q.State.initial(plan, time_s=0.)
    before = state.to_bytes()
    outputs = (total, plan.fields[0], energy)
    options = dict(inputs=(parameter,), outputs=outputs, state=state,
                   steps=3, output_steps=(1, 3))
    with pytest.raises(q.ValidationError):
        q.diff.compile(plan, **options)
    program = q.diff.compile(plan, **options, quadrature_points=2)
    assert tuple(output.shape for output in program.outputs) == ((2,), (2, 3), (2,))
    assert all(output.sample_times_s == (1., 3.) for output in program.outputs)
    times = np.array([1., 3.])
    expected, derivative = trajectory_reference(4., nonlinear, complex_field)
    h, dh = expected[:, 0], derivative[:, 0]
    integral_output = observable_output == "amount"
    values = (times*(h if integral_output else h.real+4), expected, np.abs(h)**2)
    tangents = (times*(dh if integral_output else dh.real+1), derivative, 2*(h.conj()*dh).real)
    point = program.evaluate((4.,))
    scalar_program = q.diff.compile(plan, **(options | {"outputs": (total,)}), quadrature_points=2)
    scalar = scalar_program.evaluate((4.,))
    np.testing.assert_allclose(scalar.primal().output[0], values[0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(scalar.jvp((1.,)).tangent[0], tangents[0], rtol=0, atol=1e-9)
    # Ordinary Result observations and differentiated evaluation use the same
    # accepted endpoint meaning, including the time inside the spatial integral.
    result = q.run(plan, state=state, steps=3, output_steps=(1, 3))
    result = q.Result.from_bytes(plan, result.to_bytes())
    canonical = program.primal().output
    for index in range(2):
        np.testing.assert_allclose(result.observe(total, output_index=index, quadrature_points=2).value,
                                   canonical[0][index], rtol=0, atol=1e-9)
    for actual, wanted in zip(point.primal().output, values, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-9)
    for actual, wanted in zip(point.jvp((1.,)).tangent, tangents, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-9)
    observable_weights = (np.array([2.+1j, -1.-2j]) if integral_output and complex_field
                          else np.array([2., -1.]))
    weights = (observable_weights, np.full((2, 3), 1+3j if complex_field else 1.),
               np.array([-.5, 2.]))
    paired = sum(np.vdot(w, d).real for w, d in zip(weights, tangents, strict=True))
    np.testing.assert_allclose(point.vjp(weights).input_cotangent, [paired], rtol=0, atol=1e-9)
    collection = program.map((np.array([4., 4.]),), point_shape=(2,)).execute_recorded()
    encoded = collection.to_bytes()
    reopened = q.EvaluationCollection.from_bytes(encoded)
    assert reopened.to_bytes() == encoded
    for member in reopened[0]:
        for actual, wanted in zip(member.result.output, values, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-9)
    assert state.to_bytes() == before


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("complex_field", [False, True])
def test_midpoint_trajectory_products_follow_stage_and_endpoint_meaning(nonlinear, complex_field):
    plan, parameter, (total, energy) = step_plan(
        nonlinear, complex_field, observables=True, midpoint=True)
    state = q.State.initial(plan, time_s=0.)
    before = state.to_bytes()
    times = np.array([1., 3.])

    def reference(p):
        # Midpoint stage m satisfies m^3+2*m=2*v_previous+p (or
        # 3*m=2*v_previous+p). The closed inverse selects its unique real root.
        v, d = 0., 0.
        values, derivatives = [], []
        for _ in range(3):
            rhs = 2*v+p
            m = (2*np.sqrt(2/3)*np.sinh(np.arcsinh(3*np.sqrt(3)*rhs/(4*np.sqrt(2)))/3)
                 if nonlinear else rhs/3)
            slope = 3*m*m if nonlinear else 1.
            v, d = 2*m-v, (1+(1-.5*slope)*d)/(1+.5*slope)
            values.append(v)
            derivatives.append(d)
        phase = 1+2j if complex_field else 1.
        h, dh = (phase*np.asarray(x)[[0, 2]] for x in (values, derivatives))
        return ((np.repeat(h[:, None], 3, axis=1), times*(h.real+p), np.abs(h)**2),
                (np.repeat(dh[:, None], 3, axis=1), times*(dh.real+1), 2*(h.conj()*dh).real))

    ordinary = q.run(plan, state=state, until_s=3., output_times_s=(1., 3.))
    canonical, _ = reference(2.)
    for index in range(2):
        np.testing.assert_allclose(
            ordinary.observe(total, output_index=index, quadrature_points=2).value,
            canonical[1][index], rtol=0, atol=1e-9)
    program = q.diff.compile(plan, inputs=(parameter,), outputs=(plan.fields[0], total, energy),
        state=state, until_s=3., output_times_s=(1., 3.), quadrature_points=2)
    point = program.evaluate((4.,))
    values, tangents = reference(4.)
    for actual, expected in zip(point.primal().output, values, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    for actual, expected in zip(point.jvp((1.,)).tangent, tangents, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    weights = (np.full((2, 3), 1+3j if complex_field else 1.),
               np.array([2., -1.]), np.array([-.5, 2.]))
    paired = sum(np.vdot(w, d).real for w, d in zip(weights, tangents, strict=True))
    np.testing.assert_allclose(point.vjp(weights).input_cotangent, [paired], rtol=0, atol=1e-9)
    assert state.to_bytes() == before
