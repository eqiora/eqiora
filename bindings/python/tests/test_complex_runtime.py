"""Typed complex values use ordinary Model/Plan/State/Run/Result owners."""
import json

import pytest
import eqiora


def linear():
    return eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer(),
    )


def run(model):
    plan = eqiora.resolve(model, solve=linear())
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    state = eqiora.State.initial(restored)
    state = eqiora.State.from_bytes(restored, state.to_bytes())
    result = eqiora.run(restored, state=state)
    replay = eqiora.Result.from_bytes(restored, result.to_bytes())
    assert replay.to_bytes() == result.to_bytes()
    return restored, replay


def test_native_complex_equation_matches_source_graph_and_retains_both_parts():
    kind = eqiora.ValueType.complex()
    coefficient = eqiora.Parameter("coefficient", value_type=kind, value=1-2j)
    z = eqiora.Field("z", role=eqiora.FieldRole.Variable, value_type=kind)
    native = eqiora.compile(source=eqiora.Module("M", coefficient, z,
        eqiora.Relation("r", equations=[(coefficient*z, 11-2j)]),
        eqiora.Observable("output", value_type=kind, expression=z)))
    source = eqiora.compile(source="""model M(){
        parameter coefficient:complex<1>=math.complex(1,-2);
        variable z:complex<1>; relation r{coefficient*z=math.complex(11,-2);}
        observable output:complex<1>=z;
    }""")
    assert native.structural_fingerprint == source.structural_fingerprint
    for model in (native, source):
        plan, result = run(model)
        assert len(plan.fields) == 1
        assert plan.capability.unknown_count == 2
        assert plan.enforcement is None
        # (1-2i)(3+4i)=11-2i, independently of the implementation's block ordering.
        value = result.observe(model.observable("output")).value
        assert type(value) is complex
        assert value == pytest.approx(3+4j, abs=1e-10)
        payload = json.loads(result.to_bytes())
        payload["content"]["payload"]["values"][1] = 0.0
        with pytest.raises(eqiora.ValidationError, match="original Model equality residual"):
            eqiora.Result.from_bytes(plan, json.dumps(payload).encode())


def test_complex_channel_module_source_and_replay_agree_at_real_projection():
    q = eqiora.lang
    source = eqiora.Module("main")
    owner = source.model("M")
    kind = eqiora.ValueType.array(eqiora.ValueType.complex(), 2)
    z = owner.field("z", role=eqiora.FieldRole.Variable, value_type=kind)
    owner.relation("r", q.equation(q.math.complex(1, -2)*z,
                                   q.array([11-2j, 5j])))
    owner.observable("output", q.math.real(z[0]) + q.math.imag(z[1]) + q.math.abs2(z[1]),
                     value_type=eqiora.ValueType.real())
    model = eqiora.compile(source=source, entry="M")
    emitted = eqiora.compile(source=source.to_eqi(), entry="M")
    assert model.structural_fingerprint == emitted.structural_fingerprint
    for candidate, output in ((model, model.observable("output")),
                              (emitted, emitted.observable("output")),
                              (eqiora.Model.from_bytes(model.to_bytes()), model.observable("output"))):
        plan, result = run(candidate)
        assert len(plan.fields) == 1
        assert plan.capability.unknown_count == 4
        # z=[3+4i,-2+i]; Re(z[0])+Im(z[1])+|z[1]|²=3+1+5=9.
        value = result.observe(output).value
        assert type(value) is float
        assert value == pytest.approx(9, abs=1e-10)


@pytest.mark.parametrize("expression", [
    "math.arg(z-z)", "math.log(math.real(z)-3)",
    "math.sqrt(math.real(z)-4)",
])
def test_observable_domain_errors_do_not_silently_project_or_promote(expression):
    model = eqiora.compile(source=f"""model M(){{
        variable z:complex<1>; relation r{{z=math.complex(3,4);}}
        observable output:1={expression};
    }}""")
    _, result = run(model)
    with pytest.raises(eqiora.ExecutionError, match="EQ0505"):
        result.observe(model.observable("output"))
