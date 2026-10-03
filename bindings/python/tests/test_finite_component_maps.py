"""Nominal finite maps share source authoring, exact replay, and the ordinary lifecycle."""
import pytest

import eqiora


def test_finite_types_keep_scalar_basis_duality_and_order():
    spin = eqiora.FiniteSpace("Spin", labels=("up", "down"))
    control = eqiora.FiniteSpace("Control", labels=("q", "v"))
    scalar = eqiora.ValueType.complex()
    ket = eqiora.ValueType.coordinates(scalar, spin)
    bra = eqiora.ValueType.coordinates(scalar, spin, dual=True)
    assert ket != bra
    assert ket != eqiora.ValueType.coordinates(scalar, control)
    assert ket != eqiora.ValueType.array(scalar, 2)
    matrix = eqiora.ValueType.linear_map(scalar, spin, control)
    assert matrix != eqiora.ValueType.linear_map(scalar, control, spin)
    assert matrix.shape == [2, 2] and matrix.array_rank == 0
    parameter = eqiora.Parameter("matrix", value_type=matrix, value=((1, 1j), (2, -1)))
    field = eqiora.Field("state", role=eqiora.FieldRole.Variable, value_type=ket)
    rhs = eqiora.Parameter("rhs", value_type=ket, value=(1+1j, 2-1j))
    model = eqiora.compile(source=eqiora.Module("M", spin, control, parameter, rhs, field,
        eqiora.Relation("r", equations=((field, rhs),))))
    reference = model.parameter("matrix")
    replay = eqiora.Model.from_bytes(model.to_bytes())
    assert replay.parameter(reference.id).value_type == matrix
    assert replay.parameter(reference.id).value == ((1+0j, 1j), (2+0j, -1+0j))
    for invalid in (eqiora.ValueType.array(scalar, 1), eqiora.ValueType.vector(scalar, 2)):
        with pytest.raises(ValueError):
            eqiora.ValueType.coordinates(invalid, spin)
    with pytest.raises(ValueError):
        eqiora.ValueType.linear_map(eqiora.ValueType.integer(), spin, control)
    with pytest.raises(ValueError):
        eqiora.ValueType.coordinates(eqiora.ValueType.integer(), spin, dual=True)


def test_python_quantum_map_authoring_and_plan_result_replay():
    q = eqiora.lang
    source = eqiora.Module("main")
    spin = source.space("Spin", labels=("up", "down"))
    owner = source.model("M")
    scalar = eqiora.ValueType.complex()
    ket_type = eqiora.ValueType.coordinates(scalar, spin)
    map_type = eqiora.ValueType.linear_map(scalar, spin, spin)
    h = owner.parameter("h", value_type=map_type)
    owner.set_default(h, owner.linear_map(spin, spin, ((0, -1j), (1j, 0))))
    rhs = owner.parameter("rhs", value_type=ket_type)
    owner.set_default(rhs, owner.coordinates(spin, (-1-2j, -1+1j)))
    state = owner.field("state", role=eqiora.FieldRole.Variable, value_type=ket_type)
    owner.relation("r", q.equation(q.apply(h, state), rhs))
    owner.observable("output", state, value_type=ket_type)
    owner.observable("norm", q.pair(q.adjoint(state), state), value_type=scalar)
    owner.observable("squared", q.compose(h, h), value_type=map_type)
    owner.observable("transposed", q.transpose(h), value_type=eqiora.ValueType.linear_map(scalar, spin, spin, source_dual=True, target_dual=True))
    model = eqiora.compile(source=source, entry="M")
    text_model = eqiora.compile(source=source.to_eqi(), entry="M")
    assert model.structural_fingerprint == text_model.structural_fingerprint
    observables = {name: model.observable(name) for name in ("output", "norm", "squared", "transposed")}
    model = eqiora.Model.from_bytes(model.to_bytes())
    policy = eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(),
    )
    plan = eqiora.resolve(model, solve=policy)
    plan = eqiora.Plan.from_bytes(plan.to_bytes())
    initial = eqiora.State.initial(plan)
    initial = eqiora.State.from_bytes(plan, initial.to_bytes())
    result = eqiora.run(plan, state=initial)
    result = eqiora.Result.from_bytes(plan, result.to_bytes())
    # Pauli Y*[1+i,2-i]=[-1-2i,-1+i], with squared norm 7 and Y²=I.
    assert result.observe(observables["output"]).value == pytest.approx((1+1j, 2-1j), abs=1e-10)
    assert result.observe(observables["norm"]).value == pytest.approx(7+0j, abs=1e-10)
    assert result.observe(observables["squared"]).value == ((1+0j, 0j), (0j, 1+0j))
    assert result.observe(observables["transposed"]).value == ((0j, 1j), (-1j, 0j))
    foreign = eqiora.FiniteSpace("Spin", labels=("up", "down"))
    with pytest.raises(q.ModuleError):
        owner.linear_map(foreign, spin, ((1, 0), (0, 1)))
