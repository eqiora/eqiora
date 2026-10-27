"""Typed complex values use ordinary Model/Plan/State/Run/Result owners."""
import json

import pytest
import eqiora


def linear(*, preconditioner=eqiora.solve.Preconditioner.Identity):
    return eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=preconditioner,
        reduction=eqiora.solve.Reduction.Reproducible, provider=eqiora.solve.SolverProvider.reference(),
    )


def run(model, *, solve=None):
    plan = eqiora.resolve(model, solve=linear() if solve is None else solve)
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    state = eqiora.State.initial(restored)
    state = eqiora.State.from_bytes(restored, state.to_bytes())
    result = eqiora.run(restored, state=state)
    replay = eqiora.Result.from_bytes(restored, result.to_bytes())
    assert replay.to_bytes() == result.to_bytes()
    return restored, replay


@pytest.mark.parametrize("emitted_source", [False, True])
@pytest.mark.parametrize("provider,algorithm,reduction", [
    ("reference", "BiConjugateGradientStabilized", "Reproducible"),
    ("faer", "BiConjugateGradientStabilized", "Fast"),
    ("faer", "SparseLu", "Fast"),
])
def test_real_finite_values_keep_the_admitted_solver_policy(
    emitted_source, provider, algorithm, reduction,
):
    q = eqiora.lang
    source = eqiora.Module("main")
    owner = source.model("M")
    kind = eqiora.ValueType.real()
    x = owner.field("x", role=eqiora.FieldRole.Variable, value_type=kind)
    y = owner.field("y", role=eqiora.FieldRole.Variable, value_type=kind)
    owner.relation("first", q.equation(4*x+y, 6))
    owner.relation("second", q.equation(2*x+3*y, 8))
    owner.observable("output", x+10*y, value_type=kind)
    model = eqiora.compile(
        source=source.to_eqi() if emitted_source else source, entry="M",
    )
    solve = eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=getattr(eqiora.solve.LinearSolver, algorithm),
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=getattr(eqiora.solve.Reduction, reduction),
        provider=getattr(eqiora.solve.SolverProvider, provider)(),
    )
    _, result = run(model, solve=solve)
    # The independent solution is x=1,y=2 for both authoring routes.
    assert result.observe(model.observable("output")).value == pytest.approx(21, abs=1e-10)


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
        observed = result.observe(model.observable("output"))
        assert observed.project_component("magnitude")[0] == pytest.approx(5, abs=1e-10)
        assert observed.project_component("squared_magnitude")[0] == pytest.approx(25, abs=1e-10)
        value = observed.value
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
    with pytest.raises(eqiora.ExecutionError, match="EQ0507"):
        result.observe(model.observable("output"))


@pytest.mark.parametrize(("solve", "provider", "preconditioner"), [
    (linear(), eqiora.solve.SolverProvider.reference(), "identity"),
    (linear(preconditioner=eqiora.solve.Preconditioner.Jacobi), eqiora.solve.SolverProvider.reference(), "jacobi"),
    (eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer(),
    ), eqiora.solve.SolverProvider.faer(), "identity"),
    (eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Jacobi,
        reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer(),
    ), eqiora.solve.SolverProvider.faer(), "jacobi"),
    (eqiora.solve.Linear(
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=8,
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast, provider=eqiora.solve.SolverProvider.faer(),
    ), eqiora.solve.SolverProvider.faer(), "identity"),
])
def test_two_by_two_complex_system_retains_selected_provider_through_plan_replay(solve, provider, preconditioner):
    model = eqiora.compile(source="""model M(){
        variable z:array<complex<1>,2>;
        relation r{
            math.complex(1,1)*z[0]+2*z[1]=math.complex(-5,5);
            math.complex(0,3)*z[0]+math.complex(4,-1)*z[1]=math.complex(-13,9);
        }
        observable first:complex<1>=z[0];
        observable second:complex<1>=z[1];
    }""")
    plan, result = run(model, solve=solve)
    assert plan.solve.backend == provider.id
    assert plan.solve.preconditioner == preconditioner
    assert plan.solve.maximum_iterations == 8
    assert plan.solve.relative_tolerance == 1e-13
    assert plan.solve.absolute_tolerance == 1e-15
    assert result.observe(model.observable("first")).value == pytest.approx(1+2j, abs=1e-10)
    assert result.observe(model.observable("second")).value == pytest.approx(-2+1j, abs=1e-10)


def test_pure_imaginary_coefficient_solves_without_real_block_breakdown():
    model = eqiora.compile(source="""model M(){
        variable z:complex<1>;
        relation r{math.complex(0,1)*z=math.complex(-2,1);}
        observable output:complex<1>=z;
    }""")
    _, result = run(model)
    assert result.observe(model.observable("output")).value == pytest.approx(1+2j, abs=1e-10)


def test_native_component_projections_preserve_dimension_lineage_and_undefined_phase():
    import math
    model = eqiora.compile(source="""model M(){
        variable z:complex<V>; relation r{math.complex(1,-2)*z=math.complex(11[V],-2[V]);}
        observable response:complex<V>=z;
        observable zero:complex<V>=math.complex(0[V],0[V]);
        observable squared:V^2=math.abs2(z);
        observable peak_power:W=math.real(z*math.conj(math.complex(1[A],2[A])))/2;
    }""")
    _, result = run(model)
    output = result.observe(model.observable("response"))
    identity = output.result_identity
    for name, expected in [("real",3), ("imaginary",4), ("magnitude",5), ("squared_magnitude",25)]:
        value, dimension = output.project_component(name)
        assert value == pytest.approx(expected, abs=1e-10)
        comparison = result.observe(model.observable("squared")) if name == "squared_magnitude" else output
        assert dimension == comparison.value_type.dimension
    phase, dimension = output.project_component("phase")
    assert math.sin(phase) == pytest.approx(4/5, abs=1e-12)
    assert math.cos(phase) == pytest.approx(3/5, abs=1e-12)
    assert dimension == eqiora.ValueType.real().dimension
    assert result.observe(model.observable("zero")).project_component("phase") is None
    assert result.observe(model.observable("peak_power")).value == pytest.approx(5.5,abs=1e-10)
    assert output.result_identity == identity == json.loads(result.to_bytes())["identity"]
    with pytest.raises(ValueError, match="projection must"):
        output.project_component("power")
    with pytest.raises(ValueError, match="component count"):
        output.project_component("real", 1)


def test_explicit_finite_wavefunction_probability_uses_native_component_projections():
    model = eqiora.compile(source="""model State(){
        variable psi:array<complex<1>,2>;
        relation fixed{psi=[math.complex(0.6,0),math.complex(0,0.8)];}
        observable amplitude:array<complex<1>,2>=psi;
        observable probability:1=math.abs2(psi[0])+math.abs2(psi[1]);
    }""")
    _, result = run(model)
    amplitude = result.observe(model.observable("amplitude"))
    for index, expected in enumerate((0.36,0.64)):
        assert amplitude.project_component("squared_magnitude",index)[0] == pytest.approx(expected, abs=1e-12)
    assert result.observe(model.observable("probability")).value == pytest.approx(1,abs=1e-12)


@pytest.mark.parametrize("objective,candidate,provider,algorithm,reduction", [
    (eqiora.solve.Robust, "eqiora.reference.bicgstab-general-identity-reproducible-complex-f64", "eqiora.reference", "bicgstab", "reproducible"),
    (eqiora.solve.Fast, "eqiora.faer.sparse-lu-general-identity-fast-complex-f64", "eqiora.faer", "sparse-lu", "fast"),
    (eqiora.solve.LowMemory, "eqiora.faer.bicgstab-general-identity-fast-complex-f64", "eqiora.faer", "bicgstab", "fast"),
])
def test_complex_automatic_objectives_keep_exact_selection_through_replay(
    objective, candidate, provider, algorithm, reduction,
):
    model = eqiora.compile(source="""model M(){
        variable z:array<complex<1>,2>;
        relation r{
            math.complex(1,1)*z[0]+2*z[1]=math.complex(-5,5);
            math.complex(0,3)*z[0]+math.complex(4,-1)*z[1]=math.complex(-13,9);
        }
        observable first:complex<1>=z[0];
        observable second:complex<1>=z[1];
    }""")
    plan, result = run(model, solve=eqiora.solve.Linear(
        relative_tolerance=1e-12, absolute_tolerance=1e-14,
        maximum_iterations=100, objective=objective,
    ))
    assert plan.solve.objective is objective
    assert plan.solve.planning_policy_id == "eqiora.host-serial-solver-planning/v6"
    assert plan.solve.selected_candidate_id == candidate
    assert plan.solve.backend == provider
    assert plan.solve.algorithm == algorithm
    assert plan.solve.reduction == reduction
    # Independently: A*[1+2i,-2+i]=[-5+5i,-13+9i].
    assert result.observe(model.observable("first")).value == pytest.approx(1+2j, abs=1e-10)
    assert result.observe(model.observable("second")).value == pytest.approx(-2+1j, abs=1e-10)
