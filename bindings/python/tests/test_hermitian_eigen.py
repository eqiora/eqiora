"""Source-to-Result Hermitian execution in the installed Python package."""

import eqiora
import json


def test_python_hermitian_plan_run_result_and_replay() -> None:
    model = eqiora.compile(source="""
    space Spin=orthonormal(up,down);
    model Quantum() {
     parameter h:map<complex<J>,Spin,Spin>=linear_map(Spin,Spin,[[2[J],math.complex(0[J],-1[J])],[math.complex(0[J],1[J]),2[J]]]);
     variable u:coordinates<complex<1>,Spin>;
     variable lambda:J;
     relation states {apply(h,u)=lambda*u;}
    }
    """)
    controls = dict(count=2, provider=eqiora.solve.SolverProvider.faer(), residual_tolerance=1e-12, normalization_tolerance=1e-12)
    plan = eqiora.resolve(model, solve=eqiora.solve.HermitianEigen(**controls))
    assert plan.formulation.effective == eqiora.FormulationKind.FiniteHermitianPencil
    assert plan.capability.excluded_space is None
    assert plan.capability.mode_field == model.field("u")
    assert plan.capability.eigenvalue_field == model.field("lambda")
    assert plan.fields == (model.field("u"), model.field("lambda"))
    assert plan.solve.count == plan.requested_solve.count == 2
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    assert restored.to_bytes() == plan.to_bytes()
    assert restored.solve.provider == plan.solve.provider
    result = eqiora.run(restored)
    assert result.eigen_convergence == "converged"
    assert result.eigenpair_count == 2
    assert result.eigen_candidate_counts == (2, 0)
    for i, expected in enumerate((1., 3.)):
        pair = result.eigenpair(i)
        assert isinstance(pair, eqiora.Eigenpair)
        assert "result_identity=" in repr(pair)
        assert "object at" not in repr(pair)
        assert abs(pair.eigenvalue - expected) < 1e-12
        assert pair.relative_residual < 1e-12 and pair.normalization_defect < 1e-12
        assert pair.mode_field == model.field("u")
        assert pair.eigenvalue_field == model.field("lambda")
        assert any(abs(complex(v).imag) > 0 for v in pair.mode)
    projector, projector_type = result.eigenprojector([0, 1])
    for row in range(2):
        for col in range(2):
            assert abs(projector[row][col] - (1 if row == col else 0)) < 1e-12
    replayed = eqiora.Result.from_bytes(restored, result.to_bytes())
    assert replayed.to_bytes() == result.to_bytes()
    assert replayed.eigen_convergence == "converged"
    dimension = result.eigenpair(0).eigenvalue_type.dimension
    partial_plan = eqiora.resolve(model, solve=eqiora.solve.HermitianEigen(**controls, interval=((0., dimension), (2., dimension))))
    assert partial_plan.solve.interval == ((0., dimension), (2., dimension))
    assert eqiora.run(partial_plan).eigen_convergence == "partial"
    nearest = dict(controls, count=1)
    target_plan = eqiora.resolve(model, solve=eqiora.solve.HermitianEigen(**nearest, target=(2.75, dimension)))
    assert abs(eqiora.run(target_plan).eigenpair(0).eigenvalue - 3.) < 1e-12
    for action in (
        lambda: eqiora.resolve(model, solve=eqiora.solve.HermitianEigen(**dict(controls, provider=eqiora.solve.SolverProvider.reference()))),
        lambda: eqiora.run(plan, until_s=1.),
        lambda: eqiora.State.initial(plan),
        lambda: eqiora.Result.from_bytes(partial_plan, result.to_bytes()),
        lambda: result.eigenprojector([0, 0]),
    ):
        try:
            action()
        except (eqiora.EqioraError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("unsupported or inconsistent spectral request must reject")


def test_python_source_coordinate_embedding_and_original_residual() -> None:
    source = """
    space Full=orthonormal(first,second);
    space Reduced=orthonormal(relative);
    model Floating() {
     parameter a:map<1,Full,Full>=linear_map(Full,Full,[[1,-1],[-1,1]]);
     parameter b:map<1,Full,Full>=linear_map(Full,Full,[[1,-1],[-1,1]]);
     parameter p:map<1,Reduced,Full>=linear_map(Reduced,Full,[[-2],[2]]);
     parameter r:map<1,Full,Full>=linear_map(Full,Full,[[0,2],[3,1]]);
     variable u:coordinates<1,Full>;
     variable q:coordinates<1,Reduced>;
     variable lambda:1;
     relation pencil {apply(a,u)=lambda*apply(b,u);}
     relation coordinates {apply(r,u)=apply(p,q);}
    }
    """
    model = eqiora.compile(source=source)
    solve = eqiora.solve.HermitianEigen(count=1, provider=eqiora.solve.SolverProvider.faer(), residual_tolerance=1e-12, normalization_tolerance=1e-12)
    plan = eqiora.resolve(model, solve=solve)
    assert plan.fields == (model.field("u"), model.field("lambda"), model.field("q"))
    excluded = plan.capability.excluded_space
    assert isinstance(excluded, eqiora.solve.EigenExclusion)
    assert excluded.dimension == 1
    assert excluded.is_operator_null and excluded.is_metric_null
    assert excluded.operator_defect <= excluded.tolerance
    assert excluded.metric_defect <= excluded.tolerance
    for row in excluded.projector:
        for entry in row:
            assert abs(entry - 0.5) < 1e-12
    assert "dimension=1" in repr(excluded)
    (embedding,) = plan.capability.coordinate_embeddings
    assert isinstance(embedding, eqiora.solve.EigenCoordinateMap)
    assert embedding.target_field == model.field("u")
    assert embedding.coordinate_field == model.field("q")
    assert embedding.mapping == ((1.,), (-1.,))
    assert embedding.relation_id and "relation_id=" in repr(embedding)
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    assert restored.fields == plan.fields
    restored_exclusion = restored.capability.excluded_space
    assert restored_exclusion.projector == excluded.projector
    assert restored_exclusion.projector_type == excluded.projector_type
    assert restored_exclusion.is_operator_null and restored_exclusion.is_metric_null
    (restored_embedding,) = restored.capability.coordinate_embeddings
    assert restored_embedding.relation_id == embedding.relation_id
    assert restored_embedding.mapping == embedding.mapping
    assert restored_embedding.mapping_type == embedding.mapping_type
    result = eqiora.run(restored)
    assert result.eigen_convergence == "converged"
    pair = result.eigenpair(0)
    assert abs(pair.eigenvalue - 1.) < 1e-12
    assert len(pair.mode) == 2
    assert abs(abs(pair.mode[0]) - .5) < 1e-12
    assert abs(pair.mode[0] + pair.mode[1]) < 1e-12
    assert pair.mode_field == model.field("u")
    q, q_type = pair.field(model.field("q"))
    assert len(q) == 1 and abs(q[0] - pair.mode[0]) < 1e-12
    assert pair.field(model.field("u")) == (pair.mode, pair.mode_type)
    assert pair.field(model.field("lambda")) == (pair.eigenvalue, pair.eigenvalue_type)
    foreign = eqiora.compile(source=source.replace("model Floating", "model Other"))
    try:
        pair.field(foreign.field("q"))
    except ValueError:
        pass
    else:
        raise AssertionError("coordinate values require exact Model ownership")
    replay = eqiora.Result.from_bytes(restored, result.to_bytes())
    assert replay.to_bytes() == result.to_bytes()
    assert replay.eigenpair(0).mode == pair.mode
    assert replay.eigenpair(0).field(model.field("q")) == (q, q_type)
    wire = json.loads(result.to_bytes())
    assert wire["schema"] == "eqiora.common-result/v12"
    candidate = wire["content"]["payload"]["spectral"]["candidates"][0]
    assert "mode" not in candidate
    assert len(candidate["coordinates"]) == 1
    # A forged original-space vector cannot masquerade as admitted coordinates.
    candidate["coordinates"] = [[.5, 0.], [-.5, 0.]]
    try:
        eqiora.Result.from_bytes(restored, json.dumps(wire).encode())
    except (eqiora.EqioraError, ValueError):
        pass
    else:
        raise AssertionError("candidate must retain its exact admitted coordinate shape")


def test_python_authored_finite_weak_form_execution_and_replay() -> None:
    source = """
    space Spin=orthonormal(up,down);
    public component Wave() {
     parameter h:map<complex<1>,Spin,Spin>=linear_map(Spin,Spin,[[2,math.complex(0,-1)],[math.complex(0,1),2]]);
     variable u:coordinates<complex<1>,Spin>;
     variable lambda:1;
     relation states {apply(h,u)=lambda*u;}
     form weak for states {test eta:1 for u; inner(eta,apply(h,u))=inner(eta,lambda*u);}
    }
    """
    model = eqiora.compile(source=source, entry="Wave")
    solve = eqiora.solve.HermitianEigen(count=2, provider=eqiora.solve.SolverProvider.faer(), residual_tolerance=1e-12, normalization_tolerance=1e-12)
    plan = eqiora.resolve(model, solve=solve)
    assert plan.formulation.requested == eqiora.FormulationSelectionMode.Authored
    assert plan.formulation.requested_source_identity == model.authored_formulations[0].source_identity
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    assert restored.to_bytes() == plan.to_bytes()
    result = eqiora.run(restored)
    for i, expected in enumerate((1., 3.)):
        assert abs(result.eigenpair(i).eigenvalue - expected) < 1e-12
    assert eqiora.Result.from_bytes(restored, result.to_bytes()).to_bytes() == result.to_bytes()
    wrong = eqiora.compile(source=source.replace("inner(eta,apply(h,u))", "inner(eta,2*apply(h,u))"), entry="Wave")
    try:
        eqiora.resolve(wrong, solve=solve)
    except eqiora.EqioraError:
        pass
    else:
        raise AssertionError("finite weak correspondence ignored a changed coefficient")
