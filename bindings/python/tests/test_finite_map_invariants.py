"""Local finite-map mathematics through installed structured/source authoring."""
import pytest

import eqiora


def _policy():
    return eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=64,
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
    )


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
