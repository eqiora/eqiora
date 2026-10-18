"""Local finite-map mathematics through installed structured/source authoring."""
import pytest
import numpy as np

import eqiora


def _policy():
    return eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=64,
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
    )


@pytest.mark.parametrize("exponent", [-400, -700, 400, 700])
@pytest.mark.parametrize("method", [eqiora.fem.Q1, eqiora.fvm.CellCenteredTpfa])
def test_affine_map_coefficients_retain_range_through_spatial_parameter_actions(exponent, method):
    scale = 2.0 ** exponent
    graph = eqiora.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 3.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    model = eqiora.compile(source=f"""
space S=orthonormal(x,y);
public model ScaledMap(support body:volume(ambient_dimension=1),
    support left:boundary(parent=body), support right:boundary(parent=body)) {{
    parameter a:map<m,S,S>=linear_map(S,S,[[{scale!r}[m],0[m]],[0[m],{scale!r}[m]]]);
    variable h:1 on body in smooth;
    relation prescribed on body {{h=(1+coordinate(0)/1[m])^2;}}
    variable u:1 on body in smooth;
    relation balance on body {{-div(grad(u)+grad(math.sqrt(h)*(matrix_trace(inverse(a))*{scale!r}[m])))=0;}}
    relation left_value on left {{trace(u)=matrix_trace(inverse(a))*{scale!r}[m];}}
    relation right_value on right {{trace(u)=4*(matrix_trace(inverse(a))*{scale!r}[m]);}}
}}
""", entry="ScaledMap", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    })
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(3,))))
    plan = eqiora.resolve(model, mesh=mesh, spatial=method(), solve=_policy())
    eqiora.run(plan)
    program = eqiora.diff.compile(plan, inputs=(plan.model.parameter(model.parameter("a").id),),
                                 outputs=(plan.fields[0],))
    # On [0,3], sqrt(h)=1+x, so the forcing has zero second derivative.
    # The boundary values give u=t(1+x), t=s*tr(a^-1). Both discretizations
    # reproduce this line at their vertices / cell centers. For a=k*s*I,
    # t=2/k and D_a t[H]=-tr(H)/(k^2*s).
    response = np.array([1., 2., 3., 4.] if method is eqiora.fem.Q1 else [1.5, 2.5, 3.5])
    weights = np.arange(1, len(response) + 1) / 4
    for factor in (1., 2.):
        a = np.eye(2) * scale * factor
        point = program.evaluate((a,))
        expected = 2 * response / factor
        np.testing.assert_allclose(point.primal().output[0], expected, rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.jvp((a,)).tangent[0], -expected, rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.vjp((weights,)).input_cotangent[0] * scale * factor**2,
                                   -np.dot(weights, response) * np.eye(2), rtol=0, atol=2e-10)


@pytest.mark.parametrize("exponent", [-400, -700, 400, 700])
def test_weighted_inverse_actions_keep_range_through_the_complete_parameter_chain(exponent):
    scale = 2.0 ** exponent
    model = eqiora.compile(source=f"""
space S=orthonormal(x,y);
model M() {{
    parameter p:1=0;
    parameter e:coordinates<1,S>=coordinates(S,[1,0]);
    parameter a:map<1,S,S>=linear_map(S,S,[[{scale!r},0],[0,{scale!r}]]);
    variable z:1;
    relation pin {{z=0;}}
    observable output:1=pair(adjoint(e),apply(inverse((1+p)*a),e));
}}
""")
    plan = eqiora.resolve(model, solve=_policy())
    result = eqiora.run(plan, state=eqiora.State.initial(plan))
    assert result.observe(model.observable("output")).value == 1.0 / scale
    program = eqiora.diff.compile(
        plan, inputs=(model.parameter("p"),), outputs=(model.observable("output"),)
    )
    # y=1/(s*(1+p)), y'=-1/(s*(1+p)^2); all expected powers are exact.
    for parameter in (0.0, 1.0):
        point = program.evaluate((parameter,))
        expected = (1.0 / scale) / (1.0 + parameter)
        derivative = -expected / (1.0 + parameter)
        assert point.primal().output[0].item() == expected
        assert point.jvp((1.0,)).tangent[0].item() == derivative
        assert point.vjp((1.0,)).input_cotangent[0].item() == derivative
    with pytest.raises(eqiora.ValidationError, match="singular"):
        program.evaluate((-1.0,))


def test_small_triangular_inverse_component_survives_physical_rescaling():
    s, t = 2.0 ** -500, 2.0 ** -1050
    model = eqiora.compile(source=f"""
space S=orthonormal(x,y,z);
model M() {{
    parameter a:map<1,S,S>=linear_map(S,S,[[{s!r},{t!r},0],[0,{s!r},{t!r}],[0,0,{s!r}]]);
    variable z:1;
    relation pin {{z=0;}}
    observable output:map<1,S,S>=inverse(a);
}}
""")
    plan = eqiora.resolve(model, solve=_policy())
    result = eqiora.run(plan, state=eqiora.State.initial(plan))
    # For N01=N12=1, N³=0: (sI+tN)^-1=I/s-tN/s²+t²N²/s³.
    assert result.observe(model.observable("output")).value[0][2] == 2.0 ** -600


@pytest.mark.parametrize("exponent", [-400, -700])
@pytest.mark.parametrize("inline_projection", [False, True])
@pytest.mark.parametrize("inline_direction", [False, True])
def test_inverse_projection_avoids_unrequested_unrepresentable_partials(
    exponent, inline_projection, inline_direction
):
    scale = 2.0 ** exponent
    vector = "coordinates(S,[1[1],0[1]])" if inline_projection else "e"
    direction = "linear_map(S,S,[[1,0],[0,0]])" if inline_direction else "h"
    model = eqiora.compile(source=f"""
space S=orthonormal(x,y);
model M() {{
    parameter p:1=0;
    parameter e:coordinates<1,S>=coordinates(S,[1,0]);
    parameter a:map<1,S,S>=linear_map(S,S,[[0,{scale!r}],[{scale!r},0]]);
    parameter h:map<1,S,S>=linear_map(S,S,[[1,0],[0,0]]);
    variable z:1;
    relation pin {{z=0;}}
    observable output:1=p+pair(adjoint({vector}),apply(inverse(a+p*{direction}),{vector}));
}}
""")
    plan = eqiora.resolve(model, solve=_policy())
    result = eqiora.run(plan, state=eqiora.State.initial(plan))
    assert result.observe(model.observable("output")).value == 0.0
    program = eqiora.diff.compile(
        plan, inputs=(model.parameter("p"),), outputs=(model.observable("output"),)
    )
    # A=[[p,s],[s,0]] has inverse [[0,1/s],[1/s,-p/s²]]. Therefore
    # y=p+e^T inverse(A)e=p and dy/dp=1, even if 1/s² does not fit f64.
    for parameter in (0.0, scale):
        point = program.evaluate((parameter,))
        assert point.primal().output[0].item() == parameter
        assert point.jvp((1.0,)).tangent[0].item() == 1.0
        assert point.vjp((1.0,)).input_cotangent[0].item() == 1.0


@pytest.mark.parametrize("n", [2, 6, 9])
def test_inverse_identity_and_invariants_share_the_ordinary_lifecycle(n):
    q = eqiora.lang
    source = eqiora.Module("finite_invariants")
    space = source.space("S", labels=tuple(f"q{i}" for i in range(n)))
    owner = source.model("M")
    scalar = eqiora.ValueType.real()
    vector_type = eqiora.ValueType.coordinates(scalar, space)
    map_type = eqiora.ValueType.linear_map(scalar, space, space)
    matrix = tuple(tuple(float(i + 1 + (i == j)) for j in range(n)) for i in range(n))
    # A=I+u 1^T, u_i=i+1: det(A)=1+sum(u), A^-1=I-u 1^T/det(A).
    determinant = 1 + n * (n + 1) // 2
    expected_state = tuple(float(i + 1) for i in range(n))
    rhs_values = tuple(sum(a * x for a, x in zip(row, expected_state)) for row in matrix)
    a = owner.parameter("a", value_type=map_type)
    owner.set_default(a, owner.linear_map(space, space, matrix))
    rhs = owner.parameter("rhs", value_type=vector_type)
    owner.set_default(rhs, owner.coordinates(space, rhs_values))
    state = owner.field("state", role=eqiora.FieldRole.Variable, value_type=vector_type)
    owner.relation("local", q.equation(state, q.apply(q.inverse(a), rhs)))
    owner.observable("state_value", state, value_type=vector_type)
    owner.observable("inverse_value", q.inverse(a), value_type=map_type)
    owner.observable("identity_value", owner.identity(space), value_type=map_type)
    owner.observable("composed", q.compose(a, owner.identity(space)), value_type=map_type)
    owner.observable("det_value", q.determinant(a), value_type=scalar)
    owner.observable("trace_value", q.matrix_trace(a), value_type=scalar)
    model = eqiora.compile(source=source, entry="M")
    assert model.structural_fingerprint == eqiora.compile(source=source.to_eqi(), entry="M").structural_fingerprint
    observed = {name: model.observable(name) for name in (
        "state_value", "inverse_value", "identity_value", "composed", "det_value", "trace_value")}
    model = eqiora.Model.from_bytes(model.to_bytes())
    plan = eqiora.resolve(model, solve=_policy())
    plan = eqiora.Plan.from_bytes(plan.to_bytes())
    state = eqiora.State.initial(plan)
    state = eqiora.State.from_bytes(plan, state.to_bytes())
    result = eqiora.run(plan, state=state)
    result = eqiora.Result.from_bytes(plan, result.to_bytes())
    assert result.observe(observed["state_value"]).value == pytest.approx(expected_state, abs=1e-9)
    assert result.observe(observed["det_value"]).value == pytest.approx(determinant, abs=1e-9)
    assert result.observe(observed["trace_value"]).value == n + determinant - 1
    inverse = result.observe(observed["inverse_value"]).value
    identity = result.observe(observed["identity_value"]).value
    composed = result.observe(observed["composed"]).value
    for i in range(n):
        assert inverse[i] == pytest.approx(tuple(float(i == j) - (i + 1) / determinant for j in range(n)), abs=1e-12)
        assert identity[i] == tuple(float(i == j) for j in range(n))
        assert composed[i] == matrix[i]
    foreign = eqiora.FiniteSpace("S", labels=tuple(f"q{i}" for i in range(n)))
    with pytest.raises(q.ModuleError):
        owner.identity(foreign)


@pytest.mark.parametrize("matrix,reason", [
    ("[[1,2],[2,4]]", "singular or numerically unresolved"),
    ("[[1,0],[0,1e-18]]", "reciprocal infinity-norm estimate"),
])
def test_inverse_failure_policy_reaches_python_plan_resolution(matrix, reason):
    model = eqiora.compile(source=f"""
space S=orthonormal(x,y);
model M() {{
    parameter a:map<1,S,S>=linear_map(S,S,{matrix});
    parameter rhs:coordinates<1,S>=coordinates(S,[1,1]);
    variable x:coordinates<1,S>;
    relation local {{x=apply(inverse(a),rhs);}}
}}
""")
    with pytest.raises(eqiora.ValidationError, match=reason):
        eqiora.resolve(model, solve=_policy())


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("tpfa", [False, True])
@pytest.mark.parametrize("varying", [False, True])
def test_unitful_map_coefficients_compose_with_guarded_spatial_derivatives(inverse, tpfa, varying):
    import numpy as np

    graph = eqiora.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 3.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    scale = "matrix_trace(inverse(a))*1[m]" if inverse else "determinant(a)/1[m^2]"
    coefficient = scale
    profile = ""
    if varying:
        # Both profiles give scale(b(x))=(1+x)*scale(a), through different
        # finite-map operations: row scaling for det, scalar division for inv.
        expression = "a/(1+coordinate(0)/1[m])" if inverse else (
            "compose(identity(S)+(coordinate(0)/1[m])*linear_map(S,S,[[1,0],[0,0]]),a)")
        profile = f"""
        variable b:map<m,S,S> on body in smooth;
        relation matrix_profile on body {{b={expression};}}
        """
        coefficient = scale.replace("(a)", "(b)")
    flux = f"grad(u)+grad(math.sqrt(h)*({coefficient}))"
    model = eqiora.compile(source=f"""
    space S=orthonormal(x,y);
    public model GuardedMap(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body)) {{
        parameter a:map<m,S,S>=linear_map(S,S,[[2[m],1[m]],[3[m],4[m]]]);
        {profile}
        variable h:1 on body in smooth;
        relation prescribed on body {{h=(1+coordinate(0)/1[m])^2;}}
        variable u:1 on body in smooth;
        relation balance on body {{-div({flux})=0;}}
        relation left_value on left {{trace(u)={scale};}}
        relation right_value on right {{trace(u)=4*({scale});}}
    }}
    """, entry="GuardedMap", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    })
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(3,))))
    plan = eqiora.resolve(model, mesh=mesh,
        spatial=eqiora.fvm.CellCenteredTpfa() if tpfa else eqiora.fem.Q1(), solve=_policy())
    plan = eqiora.Plan.from_bytes(plan.to_bytes())
    result = eqiora.run(plan)
    assert eqiora.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
    program = eqiora.diff.compile(plan, inputs=(plan.model.parameter(model.parameter("a").id),),
                                  outputs=(plan.fields[0],))
    # For constant map coefficients, h=(1+x)^2 gives grad(sqrt(h))=1;
    # u=t*(1+x) solves the balance with endpoint values t and 4*t, t=scale(a).
    # Prescribing u avoids assuming sqrt(h) has a boundary trace from smooth h alone.
    response = 1+np.arange(3 if tpfa else 4)+(0.5 if tpfa else 0.)
    if varying:
        # Now f=sqrt(h)*scale(b)=t*(1+x)^2, with flux u'+f'.
        # Q1 gives exact nodal values of u=t*(1+4*x-x^2).
        # TPFA: C=2*u0=u1-u0+4*t=u2-u1+6*t=16*t-2*u2,
        # hence C=6*t and cell values [3,5,5]*t. The half-cell
        # boundary closure has its ordinary quadratic discretization error.
        response = np.array([3., 5., 5.]) if tpfa else np.array([1., 4., 5., 4.])
    direction = np.array([[1., 2.], [-1., 3.]])
    weights = np.arange(1, len(response)+1)/4
    points = [np.array([[2., 1.], [3., 4.]]), np.array([[3., 1.], [2., 2.]])]
    if not inverse:
        points.append(np.array([[1., 2.], [2., 4.]]))
    for matrix in points:
        a, b, c, d = matrix.ravel()
        det = a*d-b*c
        cofactor = np.array([[d, -c], [-b, a]])
        value = (a+d)/det if inverse else det
        # Differentiate the explicit 2x2 polynomial/rational expression.
        derivative = np.eye(2)/det-(a+d)*cofactor/det**2 if inverse else cofactor
        point = program.evaluate((matrix,))
        np.testing.assert_allclose(point.primal().output[0], response*value, rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.jvp((direction,)).tangent[0],
                                   response*np.sum(derivative*direction), rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.vjp((weights,)).input_cotangent[0],
                                   np.dot(weights, response)*derivative, rtol=0, atol=2e-10)
    if inverse:
        with pytest.raises(eqiora.ValidationError, match="singular|unresolved"):
            program.evaluate((np.array([[1., 2.], [2., 4.]]),))
