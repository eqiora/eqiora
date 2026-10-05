//! Integral constraints retain Model identity through numerical execution and replay.
use super::*;
use eqiora_schema::kernel::{ExprNode, KernelNode, SymbolRef};

#[test]
fn integral_model_coupling_solves_the_distribution_amplitude() {
    // Independently, integral_phase f = (45/2)*A, so totals 45 and 90
    // determine A=2 and A=4. At x=1 the reduced density is (45/4)*A.
    for (total, amplitude) in [(45, 2.0), (90, 4.0)] {
        let source = SOURCE.replace(
            "relation amplitude_value { amplitude=3[s/m^2]; }",
            &format!("relation amplitude_value {{ mass={total}; }}"),
        );
        let (model, symbols, result) = solve(&source, [-2.0, 4.0]);
        let rules = std::collections::HashMap::from([(
            symbols.get("velocity").unwrap().downcast().unwrap(),
            QuadratureRule::gauss_legendre(3).unwrap(),
        )]);
        let value = result
            .observe_at(
                &model,
                symbols.get("density").unwrap().downcast().unwrap(),
                &[DynQuantity::new(
                    1.0,
                    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                )],
                &rules,
            )
            .unwrap();
        assert!(
            (value.value().real_scalar_value().unwrap().value() - 45.0 * amplitude / 4.0).abs()
                <= 1e-11
        );
    }
}

#[test]
fn nested_integral_constraint_replays_and_retains_its_original_dependency() {
    use eqiora_numerics::ResolvedCommonPlan;
    let source = SOURCE.replace(
        "relation amplitude_value { amplitude=3[s/m^2]; }",
        "relation amplitude_value { composed=45; }",
    );
    let (original, symbols, result) = solve(&source, [-2.0, 4.0]);
    let model =
        ModelEnvelope::from_json(&original.canonical_json().unwrap(), Default::default()).unwrap();
    let kernel = model.to_program().unwrap();
    let relation = symbols.get("amplitude_value").unwrap();
    let observable = symbols.get("composed").unwrap();
    assert!(kernel.edges().iter().any(|edge| edge.from() == relation
        && edge.to() == observable
        && edge.kind() == eqiora_graph::EdgeKind::DependsOn));
    let Some(KernelNode::Relation(definition)) = kernel.node(relation) else {
        panic!("Relation")
    };
    assert!(definition.expression().nodes().iter().any(|node|
        matches!(node, ExprNode::Symbol(SymbolRef::Observable(id)) if id.erase() == observable)));
    let plan = ResolvedCommonPlan::from_bytes(
        &result.plan().to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora::time::TimeBackendCapabilities::new(
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[eqiora::ScalarDomain::Real, eqiora::ScalarDomain::Complex],
            &[eqiora::ScalarType::F64],
        ),
    )
    .unwrap();
    let replay = CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap();
    assert_eq!(replay, result);
    let algebraic = plan.as_algebraic().unwrap();
    let rerun = algebraic
        .run_result(&algebraic.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    for result in [replay, rerun] {
        let rules = std::collections::HashMap::from([(
            symbols.get("phase").unwrap().downcast().unwrap(),
            QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap(),
        )]);
        let mass = result
            .observe(
                &model,
                symbols.get("mass").unwrap().downcast().unwrap(),
                &rules,
            )
            .unwrap();
        // Nested and full integrals both give (45/2)A, independently by Fubini
        // for this polynomial on a rectangular product, hence A=2 and mass=45.
        assert!((mass.value().real_scalar_value().unwrap().value() - 45.0).abs() <= 1e-11);
    }
}

#[test]
fn spherical_average_constraint_uses_the_volume_jacobian() {
    let source =
        "model Distribution(support position:interval(m), support velocity:interval(m/s)) {
        coordinate r:m on position from position[0];
        variable a:1;
        relation value { average=46[1/m^3]/5; }
        let density:1/m^3=a*(2[1/m^3]+3[1/m^5]*r^2);
        observable total:1=integral(density,spherical_measure(position));
        observable volume:m^3=integral(1,spherical_measure(position));
        observable average:1/m^3=total/volume;
        observable amplitude:1=a;
    }";
    let (model, symbols, result) = solve(source, [-2.0, 4.0]);
    let rules = std::collections::HashMap::from([(
        symbols.get("position").unwrap().downcast().unwrap(),
        QuadratureRule::gauss_legendre(3).unwrap(),
    )]);
    // On [0,2], spherical average of 2+3r² is 2+9R²/5=46/5.
    // The target therefore fixes a=1; line averaging would instead give 6a.
    let empty = std::collections::HashMap::new();
    for (name, expected) in [
        ("amplitude", 1.0),
        ("total", 1472.0 * std::f64::consts::PI / 15.0),
    ] {
        let value = result
            .observe(
                &model,
                symbols.get(name).unwrap().downcast().unwrap(),
                if name == "amplitude" { &empty } else { &rules },
            )
            .unwrap();
        assert!((value.value().real_scalar_value().unwrap().value() - expected).abs() <= 1e-11);
    }
}

#[test]
fn original_operand_candidates_require_exact_observable_identity_type_and_support() {
    use eqiora_core::{Id, ValueLiteral};
    let source = SOURCE.replace(
        "relation amplitude_value { amplitude=3[s/m^2]; }",
        "relation amplitude_value { mass=45; }",
    );
    let (model, symbols) = model(&source, [-2.0, 4.0]);
    let kernel = model.to_program().unwrap();
    let relation = symbols.get("amplitude_value").unwrap().downcast().unwrap();
    let mass = symbols.get("mass").unwrap().downcast().unwrap();
    let literal = |name, value| {
        let Some(KernelNode::Observable(definition)) = kernel.node(symbols.get(name).unwrap())
        else {
            panic!("Observable")
        };
        ValueLiteral::from_real(definition.value_type().clone(), value).unwrap()
    };
    let evaluate = |values: &[(Id<_>, ValueLiteral)]| {
        kernel.evaluate_relation_operands(relation, &[], &[], values)
    };
    let candidate = (mass, literal("mass", 45.0));
    let roots = evaluate(std::slice::from_ref(&candidate)).unwrap();
    assert_eq!(roots[0], roots[1]);
    for (values, message) in [
        (
            vec![(Id::new(), literal("mass", 45.0))],
            "outside this Model",
        ),
        (
            vec![candidate.clone(), candidate],
            "repeat one exact Observable",
        ),
        (
            vec![(mass, literal("density", 45.0))],
            "exact Observable type",
        ),
        (
            vec![(
                symbols.get("density").unwrap().downcast().unwrap(),
                literal("density", 45.0),
            )],
            "output support",
        ),
    ] {
        let error = evaluate(&values).unwrap_err();
        assert!(error.message().contains(message), "{error:?}");
    }
    assert!(evaluate(&[]).is_err());
}

#[test]
fn integral_coupling_rejects_nonpolynomial_and_excessive_coordinate_work() {
    let coupled = SOURCE.replace(
        "relation amplitude_value { amplitude=3[s/m^2]; }",
        "relation amplitude_value { mass=45; }",
    );
    for (density, message) in [
        ("amplitude*math.exp(-(v/4[m/s])^2)", "polynomial"),
        (
            "amplitude/(1+(v/4[m/s])^2)",
            "coordinate-independent denominators",
        ),
        ("amplitude*(v/4[m/s])^14", "Gauss"),
    ] {
        let source = coupled.replace("amplitude*(1+x/2[m])*(1+(v/4[m/s])^2)", density);
        // The Model is meaningful and retains the requested integral. This failure
        // belongs to the finite numerical profile, not source or Model admission.
        let (model, _) = model(&source, [-2.0, 4.0]);
        let error = resolve(&model).unwrap_err();
        assert!(error.message().contains(message), "{error:?}");
    }
}

#[test]
fn integral_dependency_expansion_has_a_preallocation_work_bound() {
    let declarations = (1..=20)
        .map(|i| format!("observable s{i}:1=s{}+s{};", i - 1, i - 1))
        .collect::<String>();
    let source = SOURCE.replace(
        "relation amplitude_value { amplitude=3[s/m^2]; }",
        &format!("relation amplitude_value {{ s20=45; }} observable s0:1=mass; {declarations}"),
    );
    let (model, _) = model(&source, [-2.0, 4.0]);
    let error = resolve(&model).unwrap_err();
    assert!(
        error.message().contains("65536 expression operations"),
        "{error:?}"
    );
}

#[test]
fn nonlinear_integral_constraint_accepts_the_positive_root_and_replays() {
    use eqiora_artifact::CanonicalModelArtifact;
    use eqiora_numerics::{
        CommonInitialField,
        finite_constraints::{ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement},
    };
    let source = SOURCE
        .replace(
            "relation amplitude_value { amplitude=3[s/m^2]; }",
            "relation amplitude_value { mass=90; inequality(amplitude>=0[s/m^2]); }",
        )
        .replace(
            "let f:s/m^2 on phase=amplitude*",
            "let f:s/m^2 on phase=(amplitude^2/1[s/m^2])*",
        );
    let (model, symbols) = model(&source, [-2.0, 4.0]);
    let relation = symbols.get("amplitude_value").unwrap().downcast().unwrap();
    let unit = DimExponents::from_integers([0, -2, 1, 0, 0, 0, 0]).unwrap();
    let enforcement = FiniteConstraintEnforcement::strict_interior(vec![
        ConstraintTolerance::inequality(
            ConstraintRef::new(relation, 1),
            DynQuantity::new(1e-8, unit),
        )
        .unwrap(),
    ])
    .unwrap();
    let linear = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let policy = CommonSolvePolicy::Newton {
        nonlinear: eqiora_realization::NonlinearSolvePlan::new(
            0.0,
            1e-12,
            NonZeroUsize::new(32).unwrap(),
            16,
        )
        .unwrap(),
        linear: CommonLinearRequest::exact(linear, FaerLinearSolver.provider()).unwrap(),
    };
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        policy,
        Some(enforcement),
        &[],
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    let seed = CommonInitialField::finite(
        model.artifact_reference().unwrap().artifact().clone(),
        symbols.get("amplitude").unwrap().downcast().unwrap(),
        eqiora::ValueShape::scalar(),
        vec![(1.0, 0.)],
    )
    .unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[seed]).unwrap(), &FaerLinearSolver)
        .unwrap();
    // R=(45/2)A²-90; the positive root is A=2 and R'(2)=90.
    // Residual <=1e-12 implies root error below 1e-12 near this root.
    assert!((result.finite_values().unwrap()[0] - 2.0).abs() <= 1e-12);
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan()).unwrap(),
        result
    );
    let rules = std::collections::HashMap::from([(
        symbols.get("phase").unwrap().downcast().unwrap(),
        QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap(),
    )]);
    let value = result
        .observe(
            &model,
            symbols.get("mass").unwrap().downcast().unwrap(),
            &rules,
        )
        .unwrap();
    assert!((value.value().real_scalar_value().unwrap().value() - 90.0).abs() <= 1e-11);
}

#[test]
fn finite_coupling_rejects_complex_observables_during_plan_admission() {
    let source =
        "model Distribution(support position:interval(m), support velocity:interval(m/s)) {
        variable z:complex<1>; observable output:complex<1>=z;
        relation value {output=math.complex(1,2);}
    }";
    let (model, _) = model(source, [-2.0, 4.0]);
    let error = resolve(&model).unwrap_err();
    assert!(error.message().contains("real scalar outputs"), "{error:?}");
}

#[test]
fn observable_references_remain_unavailable_to_initial_and_discrete_relations() {
    for source in [
        "model Probe() { state x:1; initial {x=2;} observable y:1=2; }",
        "model Probe() { clock tick=periodic(1[s]); variable x:1 at tick; relation value at tick {x=2;} observable y:1=2; }",
    ] {
        eqiora_compiler::compile("premise.eqi", source).unwrap();
        let errors =
            eqiora_compiler::compile("unsupported.eqi", &source.replace("x=2", "x=y")).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("Field")
                    || error.message().contains("Observable")),
            "{errors:?}"
        );
    }
}
