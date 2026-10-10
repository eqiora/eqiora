//! Complex nonlinear equations retain original typed Fields through exact replay.
use eqiora::ValueShape;
use eqiora::api::ModelDocument;
use eqiora::artifact::{CanonicalModelArtifact, ModelEnvelope};
use eqiora::solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonAlgebraicState, CommonInitialField, CommonLinearRequest,
    CommonResult, CommonSolvePolicy, ResolvedCommonPlan,
};
use eqiora_realization::NonlinearSolvePlan;
use std::num::NonZeroUsize;

fn plan(source: &str) -> (ModelDocument, CommonAlgebraicPlan) {
    let document = ModelDocument::compile("complex_newton.eqi", source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let linear = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        NonZeroUsize::new(32).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let solve = CommonSolvePolicy::Newton {
        nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(32).unwrap(), 16).unwrap(),
        linear: CommonLinearRequest::exact(linear, FaerLinearSolver.provider()).unwrap(),
    };
    let plan =
        CommonAlgebraicPlan::resolve(&model, solve, None, &[], None, &FaerLinearSolver).unwrap();
    (document, plan)
}

fn seed(
    document: &ModelDocument,
    plan: &CommonAlgebraicPlan,
    name: &str,
    shape: ValueShape,
    values: Vec<(f64, f64)>,
) -> CommonInitialField {
    CommonInitialField::finite(
        plan.model_artifact()
            .artifact_reference()
            .unwrap()
            .artifact()
            .clone(),
        document.aliases()[name].downcast().unwrap(),
        shape,
        values,
    )
    .unwrap()
}

#[test]
fn complex_cubic_uses_common_plan_state_run_and_result_replay() {
    let (document, plan) = plan(
        "model M(){parameter alpha:1=0.25;parameter b:complex<1>=math.complex(2.25,4.5);variable z:complex<1>;relation r{z+alpha*math.abs2(z)*z=b;}observable output:1=math.abs2(z);}",
    );
    assert_eq!(plan.symbols().len(), 1);
    assert_eq!(plan.coordinate_count(), 2);
    let state = plan
        .initial_state(&[seed(
            &document,
            &plan,
            "z",
            ValueShape::scalar(),
            vec![(0., 0.)],
        )])
        .unwrap();
    assert_eq!(
        CommonAlgebraicState::from_bytes(&state.to_bytes().unwrap(), &plan).unwrap(),
        state
    );
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
    let replayed = ResolvedCommonPlan::from_bytes(
        &resolved.to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora::time::TimeBackendCapabilities::new(
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[eqiora::ScalarDomain::Real, eqiora::ScalarDomain::Complex],
            &[eqiora::ScalarType::F64],
        ),
    )
    .unwrap();
    assert_eq!(replayed, resolved);
    let result = replayed
        .as_algebraic()
        .unwrap()
        .run_result(&state, &FaerLinearSolver)
        .unwrap();
    // F has real Jacobian >= I for alpha=1/4, so ||F||<=1e-12
    // bounds distance to the independent root 1+2i by 1e-12.
    for (actual, exact) in result.finite_values().unwrap().iter().zip([1., 2.]) {
        assert!((actual - exact).abs() <= 1e-12);
    }
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
        )
        .unwrap();
    assert!((observed.value().real_scalar_value().unwrap().value() - 5.).abs() < 1e-11);
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &replayed).unwrap(),
        result
    );
    // Equal component counts do not authorize reshaping scalar to [1].
    let wrong = seed(
        &document,
        &plan,
        "z",
        ValueShape::new([1]).unwrap(),
        vec![(0., 0.)],
    );
    assert!(plan.initial_state(&[wrong]).is_err());
}

#[test]
fn mixed_real_and_six_component_complex_seed_retains_each_field_shape() {
    let equalities = (0..6)
        .map(|i| format!("z[{i}]+0.25*math.abs2(z[{i}])*z[{i}]=math.complex(2.25,4.5);"))
        .collect::<String>();
    let norm = (0..6)
        .map(|i| format!("math.abs2(z[{i}])"))
        .collect::<Vec<_>>()
        .join("+");
    let source = format!(
        "model M(){{variable z:array<complex<1>,6>;variable w:1;relation r{{{equalities}w+0.25*w*w*w=4;}}observable output:1={norm}+w*w;}}"
    );
    let (document, plan) = plan(&source);
    assert_eq!(plan.symbols().len(), 2);
    assert_eq!(plan.coordinate_count(), 13);
    let complex = seed(
        &document,
        &plan,
        "z",
        ValueShape::new([6]).unwrap(),
        vec![(0., 0.); 6],
    );
    let real = seed(&document, &plan, "w", ValueShape::scalar(), vec![(0., 0.)]);
    let state = plan
        .initial_state(&[real.clone(), complex.clone()])
        .unwrap();
    let result = plan.run_result(&state, &FaerLinearSolver).unwrap();
    let output = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
        )
        .unwrap();
    // Six independent roots 1+2i and real root 2 give sum |z|²+w²=34.
    assert!((output.value().real_scalar_value().unwrap().value() - 34.).abs() < 1e-10);
    assert_eq!(
        plan.initial_state(&[complex.clone(), real.clone()])
            .unwrap(),
        state
    );
    let wrong_shape = seed(
        &document,
        &plan,
        "z",
        ValueShape::new([2, 3]).unwrap(),
        vec![(0., 0.); 6],
    );
    assert!(plan.initial_state(&[wrong_shape, real]).is_err());
    let imaginary_real = seed(&document, &plan, "w", ValueShape::scalar(), vec![(0., 1.)]);
    assert!(plan.initial_state(&[complex, imaginary_real]).is_err());
}

#[test]
fn different_physical_units_have_exact_plan_bound_residual_scales() {
    use eqiora::{DimExponents, DynQuantity};
    use eqiora_numerics::finite_constraints::ConstraintRef;
    use eqiora_realization::PositivePhysicalScale;
    let source = "model M(){parameter a:1/m^2=0.25;parameter b:complex<m>=math.complex(2.25,4.5);parameter c:1/s^2=0.25;parameter d:s=4;variable z:complex<m>;variable w:s;relation r{z+a*math.abs2(z)*z=b;w+c*w*w*w=d;}observable output:m^2=math.abs2(z);}";
    let (document, base) = plan(source);
    let relation = document.aliases()["r"].downcast().unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let physical =
        |value, dimension| PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap();
    let scales = [
        (ConstraintRef::new(relation, 0), physical(4., length)),
        (ConstraintRef::new(relation, 1), physical(2., time)),
    ];
    let resolve = |scales: &[_]| {
        CommonAlgebraicPlan::resolve(
            base.model_artifact(),
            CommonSolvePolicy::Newton {
                linear: CommonLinearRequest::exact(base.linear(), base.solver_provider()).unwrap(),
                nonlinear: base.nonlinear().unwrap(),
            },
            None,
            scales,
            None,
            &FaerLinearSolver,
        )
    };
    let scaled = resolve(&scales).unwrap();
    assert_ne!(scaled.identity(), base.identity());
    assert_eq!(scaled, resolve(&[scales[1], scales[0]]).unwrap());
    assert!(resolve(&[(scales[0].0, physical(4., time))]).is_err());
    assert!(resolve(&[scales[0], scales[0]]).is_err());
    assert_eq!(base, resolve(&base.residual_scales()).unwrap());
    let initial = scaled
        .initial_state(&[
            seed(
                &document,
                &scaled,
                "z",
                ValueShape::scalar(),
                vec![(0., 0.)],
            ),
            seed(
                &document,
                &scaled,
                "w",
                ValueShape::scalar(),
                vec![(0., 0.)],
            ),
        ])
        .unwrap();
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(scaled.clone()));
    let bytes = resolved.to_bytes().unwrap();
    let replay = |bytes: &[u8]| {
        ResolvedCommonPlan::from_bytes(
            bytes,
            &FaerLinearSolver,
            eqiora::time::TimeBackendCapabilities::new(
                eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
                &[eqiora::ScalarDomain::Real, eqiora::ScalarDomain::Complex],
                &[eqiora::ScalarType::F64],
            ),
        )
    };
    assert_eq!(replay(&bytes).unwrap(), resolved);
    let result = scaled.run_result(&initial, &FaerLinearSolver).unwrap();
    let output = result
        .observe(
            scaled.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
        )
        .unwrap();
    // Scaling changes numerical acceptance, not roots or physical output units.
    assert_eq!(
        output.value().value_type().dimension(),
        DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap()
    );
    assert!((output.value().real_scalar_value().unwrap().value() - 5.).abs() < 1e-10);
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap(),
        result
    );
    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["schema"], "eqiora.resolved-common-plan/v15");
    wire["residual_scales"][0]["value"] = serde_json::json!(0.);
    assert!(replay(&serde_json::to_vec(&wire).unwrap()).is_err());
}

#[test]
fn nonlinear_complex_parameter_sensitivities_preserve_real_pairing_under_scaling() {
    use eqiora::api::DifferentiableProgram;
    use eqiora::{DimExponents, DynQuantity};
    use eqiora_numerics::finite_constraints::ConstraintRef;
    use eqiora_realization::PositivePhysicalScale;
    let (document, base) = plan(
        "model M(){parameter alpha:1=0.25;parameter b:complex<1>=math.complex(2.25,4.5);variable z:complex<1>;relation r{z+alpha*math.abs2(z)*z=b;}observable output:1=math.abs2(z);}",
    );
    // At |z|²=5 and alpha=1/4, the radial Jacobian eigenvalue is 19/4.
    // Thus d|z|²/dalpha=-2*25/(19/4)=-200/19 and
    // grad_b |z|²=2*(x,y)/(19/4)=(8x,8y)/19 under the real pairing.
    for scale in [1., 4.] {
        let scaled = CommonAlgebraicPlan::resolve(
            base.model_artifact(),
            CommonSolvePolicy::Newton {
                linear: CommonLinearRequest::exact(base.linear(), base.solver_provider()).unwrap(),
                nonlinear: base.nonlinear().unwrap(),
            },
            None,
            &[(
                ConstraintRef::new(document.aliases()["r"].downcast().unwrap(), 0),
                PositivePhysicalScale::new(DynQuantity::new(scale, DimExponents::DIMENSIONLESS))
                    .unwrap(),
            )],
            None,
            &FaerLinearSolver,
        )
        .unwrap();
        let initial = scaled
            .initial_state(&[seed(
                &document,
                &scaled,
                "z",
                ValueShape::scalar(),
                vec![(0., 0.)],
            )])
            .unwrap();
        let program = DifferentiableProgram::compile(
            ResolvedCommonPlan::Algebraic(Box::new(scaled)),
            &[
                document.parameter_ref("alpha").unwrap(),
                document.parameter_ref("b").unwrap(),
            ],
            &document.observable_ref("output").unwrap(),
            Some(initial),
            &FaerLinearSolver,
        )
        .unwrap();
        assert_eq!(program.identity().input_dimension(), 3);
        for (x, y) in [(1., 2.), (2., -1.)] {
            let point = program.evaluate(&[0.25, 2.25 * x, 2.25 * y]).unwrap();
            assert!((point.primal().output()[0] - 5.).abs() < 1e-10);
            let gradient = [-200. / 19., 8. * x / 19., 8. * y / 19.];
            let reverse = point.vjp(&[1.]).unwrap();
            for (actual, expected) in reverse.input_cotangent().iter().zip(gradient) {
                assert!((actual - expected).abs() < 1e-10);
            }
            let direction = [1., 2., -1.];
            let expected: f64 = gradient.iter().zip(direction).map(|(a, b)| a * b).sum();
            assert!((point.jvp(&direction).unwrap().tangent()[0] - expected).abs() < 1e-10);
            let partial = point.residual_jvp(&[0., 0.], &[1., 0., 0.]).unwrap();
            for (actual, expected) in partial.iter().zip([5. * x / scale, 5. * y / scale]) {
                assert!((actual - expected).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn complex_newton_rejects_singular_zero_residual_and_exhausted_updates() {
    for equation in ["z+math.conj(z)=0", "z*z=math.complex(-1,0)"] {
        let (document, plan) = plan(&format!(
            "model M(){{variable z:complex<1>;relation r{{{equation};}}}}"
        ));
        let initial = plan
            .initial_state(&[seed(
                &document,
                &plan,
                "z",
                ValueShape::scalar(),
                vec![(0., 0.)],
            )])
            .unwrap();
        assert!(plan.run_result(&initial, &FaerLinearSolver).is_err());
    }
    // At zero, J=1e-160 I and the finite Newton update is 1e160.
    // Every one of the 16 allowed halvings still makes z² overflow. The
    // original-operand trial check must reject all trials, never commit one.
    let (document, overflow) =
        plan("model M(){parameter a:1=1e-160;variable z:complex<1>;relation r{a*z+z*z=1;}}");
    let initial = overflow
        .initial_state(&[seed(
            &document,
            &overflow,
            "z",
            ValueShape::scalar(),
            vec![(0., 0.)],
        )])
        .unwrap();
    let error = overflow
        .run_result(&initial, &FaerLinearSolver)
        .unwrap_err();
    assert!(error.message().contains("line search"), "{error:?}");
    let (document, base) = plan(
        "model M(){variable z:complex<1>;relation r{z+0.25*math.abs2(z)*z=math.complex(2.25,4.5);}}",
    );
    let solve = CommonSolvePolicy::Newton {
        linear: CommonLinearRequest::exact(base.linear(), base.solver_provider()).unwrap(),
        nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(1).unwrap(), 0).unwrap(),
    };
    let limited = CommonAlgebraicPlan::resolve(
        base.model_artifact(),
        solve,
        None,
        &[],
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    let initial = limited
        .initial_state(&[seed(
            &document,
            &limited,
            "z",
            ValueShape::scalar(),
            vec![(0., 0.)],
        )])
        .unwrap();
    assert!(limited.run_result(&initial, &FaerLinearSolver).is_err());
    let unsupported = ModelDocument::compile(
        "branch.eqi",
        "model M(){variable z:complex<1>;relation r{math.log(z)=0;}}",
    )
    .unwrap();
    let model = ModelEnvelope::from_program(unsupported.program()).unwrap();
    assert!(
        CommonAlgebraicPlan::resolve(&model, solve, None, &[], None, &FaerLinearSolver).is_err()
    );
}

#[test]
fn complex_only_linear_provider_cannot_replace_the_real_newton_differential() {
    use eqiora::solver::{
        BackendId, LinearOperatorProperties, LinearProblem, LinearSolution, PreconditionerPolicy,
        ReplicatedLinearExecution, SolverCapabilities, SolverProvider,
    };
    use eqiora_core::ScalarType;
    #[derive(Debug)]
    struct ComplexOnly;
    impl LinearSolverBackend for ComplexOnly {
        fn provider(&self) -> SolverProvider {
            SolverProvider::new(BackendId::new("eqiora.test.complex-only"), "1", &[])
        }
        fn capabilities(&self) -> SolverCapabilities {
            SolverCapabilities::new(
                [LinearSolver::SparseLu],
                [LinearOperatorProperties::General],
                [PreconditionerPolicy::Identity],
                [ReductionPolicy::Fast],
                eqiora::ScalarDomain::Complex,
                [ScalarType::F64],
            )
            .unwrap()
        }
        fn solve_with_execution(
            &self,
            _: &LinearProblem<'_>,
            _: SolverPlan,
            _: &dyn ReplicatedLinearExecution,
        ) -> Result<LinearSolution, eqiora::Diagnostic> {
            panic!("incompatible provider must reject before execution")
        }
    }
    let (_, base) = plan(
        "model M(){variable z:complex<1>;relation r{z+0.25*math.abs2(z)*z=math.complex(2.25,4.5);}}",
    );
    let backend = ComplexOnly;
    let solve = CommonSolvePolicy::Newton {
        linear: CommonLinearRequest::exact(base.linear(), backend.provider()).unwrap(),
        nonlinear: base.nonlinear().unwrap(),
    };
    let error =
        CommonAlgebraicPlan::resolve(base.model_artifact(), solve, None, &[], None, &backend)
            .unwrap_err();
    assert!(error.message().contains("Real"), "{error:?}");
}
