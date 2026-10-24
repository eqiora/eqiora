//! One mathematical Field retains both parts through the ordinary finite lifecycle.
#[path = "complex_algebraic_lifecycle/solver_policy.rs"]
mod solver_policy;
mod support;
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::solver::{
    LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan,
};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonAlgebraicState, CommonLinearRequest, CommonResult,
    CommonSolvePolicy, ResolvedCommonPlan,
};
use std::num::NonZeroUsize;
use support::derivative_products::FixtureCoordinates;

fn solve(source: &str) -> (ModelDocument, CommonAlgebraicPlan, CommonResult) {
    solve_with(source, true)
}

fn solve_with(source: &str, reference: bool) -> (ModelDocument, CommonAlgebraicPlan, CommonResult) {
    let backend: &dyn LinearSolverBackend = if reference {
        &REFERENCE_LINEAR_SOLVER
    } else {
        &FaerLinearSolver
    };
    let document = ModelDocument::compile("complex.eqi", source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        if reference {
            LinearSolver::BiConjugateGradientStabilized
        } else {
            LinearSolver::SparseLu
        },
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(if reference {
        ReductionPolicy::Reproducible
    } else {
        ReductionPolicy::Fast
    });
    let request =
        CommonSolvePolicy::Linear(CommonLinearRequest::exact(solver, backend.provider()).unwrap());
    let plan = CommonAlgebraicPlan::resolve(&model, request, None, &[], None, backend).unwrap();
    let initial = plan.initial_state(&[]).unwrap();
    assert_eq!(
        CommonAlgebraicState::from_bytes(&initial.to_bytes().unwrap(), &plan).unwrap(),
        initial
    );
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
    let replayed = ResolvedCommonPlan::from_bytes(
        &resolved.to_bytes().unwrap(),
        &FaerLinearSolver,
        &[eqiora::time::TimeBackendCapabilities::new(
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[eqiora::ScalarDomain::Real, eqiora::ScalarDomain::Complex],
            &[eqiora::ScalarType::F64],
        )],
    )
    .unwrap();
    assert_eq!(replayed, resolved);
    let result = replayed
        .as_algebraic()
        .unwrap()
        .run_result(&initial, backend)
        .unwrap();
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &replayed).unwrap(),
        result
    );
    (document, plan, result)
}

#[test]
fn complex_equation_and_real_equation_share_plan_run_and_real_observables() {
    for (domain, equation, observable, expected, coordinates) in [
        (
            "complex<1>",
            "math.complex(1,-2)*z=math.complex(11,-2)",
            "math.abs2(z)",
            25.,
            2,
        ),
        ("1", "2*z=6", "z*z", 9., 1),
    ] {
        let source = format!(
            "model M(){{variable z:{domain};relation r{{{equation};}}observable output:1={observable};}}"
        );
        let (document, plan, result) = solve_with(&source, domain == "complex<1>");
        assert_eq!(plan.symbols().len(), 1);
        assert_eq!(plan.coordinate_count(), coordinates);
        let observed = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        // (1-2i)(3+4i)=11-2i, hence |z|²=25. Binary64 solver
        // tolerance and this small well-conditioned system bound output error below 1e-10.
        assert!((observed.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-10);
    }
}

#[test]
fn shaped_complex_field_preserves_channel_order_and_explicit_real_output() {
    let source = "model M(){variable z: array<complex<1>,2>;relation r{math.complex(1,-2)*z=[math.complex(11,-2),math.complex(0,5)];}observable output:1=math.real(z[0])+math.imag(z[1])+math.abs2(z[1]);}";
    let (document, plan, result) = solve(source);
    assert_eq!(plan.symbols().len(), 1);
    assert_eq!(plan.coordinate_count(), 4);
    // z=[3+4i,-2+i], so Re(z[0])+Im(z[1])+|z[1]|²=3+1+5=9.
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    assert!((observed.value().real_scalar_value().unwrap().value() - 9.).abs() < 1e-10);
}

#[test]
fn replay_rejects_a_discarded_imaginary_part_before_digest_validation() {
    let (_, plan, result) = solve(
        "model M(){variable z:complex<1>;relation r{z=math.complex(3,4);}observable output:1=math.abs2(z);}",
    );
    let mut payload: serde_json::Value =
        serde_json::from_slice(&result.to_bytes().unwrap()).unwrap();
    payload["content"]["payload"]["values"][1] = 0.into();
    let error = CommonResult::from_bytes(
        &serde_json::to_vec(&payload).unwrap(),
        &ResolvedCommonPlan::Algebraic(Box::new(plan)),
    )
    .unwrap_err();
    assert!(
        error.message().contains("original Model equality residual"),
        "{error:?}"
    );
}

#[test]
fn contextual_zero_accepts_a_complete_complex_array_residual() {
    let source = "model M(){variable z:array<complex<1>,2>;relation r{z-[math.complex(3,4),math.complex(-2,1)]=0;}observable output:1=math.abs2(z[1]);}";
    let (document, plan, result) = solve(source);
    let observed = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    assert!((observed.value().real_scalar_value().unwrap().value() - 5.).abs() < 1e-10);
}

#[test]
fn finite_admission_counts_both_parts_before_expanding_large_fields() {
    let source = "model M(){variable z:array<complex<1>,129>;relation r{z=0;}}";
    let document = ModelDocument::compile("bounded.eqi", source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let request = CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(solver, FaerLinearSolver.provider()).unwrap(),
    );
    let error = CommonAlgebraicPlan::resolve(&model, request, None, &[], None, &FaerLinearSolver)
        .unwrap_err();
    assert!(
        error.message().contains("256 real coordinates"),
        "{error:?}"
    );
}

#[test]
fn general_complex_two_by_two_system_uses_the_typed_reference_path_through_plan_replay() {
    let source = "model M(){
      variable z:array<complex<1>,2>;
      relation r{
        math.complex(1,1)*z[0]+2*z[1]=math.complex(-5,5);
        math.complex(0,3)*z[0]+math.complex(4,-1)*z[1]=math.complex(-13,9);
      }
      observable first:complex<1>=z[0];
      observable second:complex<1>=z[1];
    }";
    let (document, plan, result) = solve(source);
    assert_eq!(plan.solver_provider(), REFERENCE_LINEAR_SOLVER.provider());
    for (name, (re, im)) in [("first", (1., 2.)), ("second", (-2., 1.))] {
        let observation = result
            .observe(
                plan.model_artifact(),
                document.aliases()[name].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        let actual = observation.value().component(0).unwrap();
        assert!((actual.0 - re).hypot(actual.1 - im) < 1e-10);
    }
}

#[test]
fn conjugate_dependence_retains_its_real_linear_profile() {
    let (document, plan, result) = solve_with(
        "model M(){variable z:complex<1>;relation r{math.conj(z)=math.complex(3,-4);}observable output:complex<1>=z;}",
        false,
    );
    assert_eq!(plan.solver_provider(), FaerLinearSolver.provider());
    let observation = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    assert_eq!(observation.value().component(0).unwrap(), (3., 4.));
}

#[test]
fn finite_quantum_and_control_maps_share_plan_run_result_and_preserve_bases() {
    // Pauli Y has Y²=I. Y*[1+i,2-i]=[-1-2i,-1+i], and the state norm is 7.
    // The independent real control map [[2,1],[1,3]] maps [1,2] to [4,7].
    for (source, complex, components, norm) in [
        (
            r#"space Spin=orthonormal(up,down); model M(){
          parameter h:map<complex<1>,Spin,Spin>=linear_map(Spin,Spin,[[0,math.complex(0,-1)],[math.complex(0,1),0]]);
          parameter rhs:coordinates<complex<1>,Spin>=coordinates(Spin,[math.complex(-1,-2),math.complex(-1,1)]);
          variable state:coordinates<complex<1>,Spin>;
          relation r{apply(h,state)=rhs;}
          observable output:coordinates<complex<1>,Spin>=state;
          observable norm:complex<1>=pair(adjoint(state),state);
        }"#,
            true,
            [(1., 1.), (2., -1.)],
            7.,
        ),
        (
            r#"space Control=orthonormal(position,velocity); model M(){
          parameter a:map<1,Control,Control>=linear_map(Control,Control,[[2,1],[1,3]]);
          parameter rhs:coordinates<1,Control>=coordinates(Control,[4,7]);
          variable state:coordinates<1,Control>;
          relation r{apply(a,state)=rhs;}
          observable output:coordinates<1,Control>=state;
          observable norm:1=pair(transpose(state),state);
        }"#,
            false,
            [(1., 0.), (2., 0.)],
            5.,
        ),
    ] {
        let (document, plan, result) = solve_with(source, complex);
        let observation = result
            .observe(
                plan.model_artifact(),
                document.aliases()["output"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        assert!(
            observation
                .value()
                .value_type()
                .coordinate_basis()
                .is_some()
        );
        for (index, expected) in components.into_iter().enumerate() {
            let actual = observation.value().component(index).unwrap();
            assert!((actual.0 - expected.0).hypot(actual.1 - expected.1) < 1e-10);
        }
        let observation = result
            .observe(
                plan.model_artifact(),
                document.aliases()["norm"].downcast().unwrap(),
                &Default::default(),
                None,
            )
            .unwrap();
        let actual = observation.value().component(0).unwrap();
        assert!((actual.0 - norm).hypot(actual.1) < 1e-10);
    }
}

#[test]
fn common_complex_differentiation_retains_real_pairing_and_parameter_points() {
    use eqiora::api::DifferentiableProgram;
    // Both pencils have z=p-ip. The conjugate pencil has real-linear, not
    // holomorphic, action. In either case J=2p²+p and dJ/dp=4p+1.
    for (equation, complex) in [
        ("a*z+math.conj(z)=math.complex(4,2)*p", false),
        ("a*z=math.complex(3,1)*p", true),
    ] {
        let source = format!(
            "model M(){{parameter p:1=3;parameter a:complex<1>=math.complex(1,2);variable z:complex<1>;relation r{{{equation};}}observable output:1=math.abs2(z)+p;}}"
        );
        let (document, plan, _) = solve_with(&source, complex);
        let backend: &'static dyn LinearSolverBackend = if complex {
            &REFERENCE_LINEAR_SOLVER
        } else {
            &FaerLinearSolver
        };
        let program = DifferentiableProgram::compile(
            ResolvedCommonPlan::Algebraic(Box::new(plan)),
            &[document.parameter_ref("p").unwrap()],
            &[document.observable_ref("output").unwrap().erase()],
            Default::default(),
            None,
            backend,
        )
        .unwrap();
        for p in [3., 5.] {
            let point = program
                .evaluate(&support::real_point(&program, &[p]).unwrap())
                .unwrap();
            assert!((point.accepted_unknowns()[0] - p).abs() < 1e-10);
            assert!((point.accepted_unknowns()[1] + p).abs() < 1e-10);
            let primal = point.primal();
            assert!((primal.output().fixture_coordinates()[0] - (2. * p * p + p)).abs() < 1e-9);
            assert!(primal.evidence().primal_solve().is_some());
            assert!(primal.evidence().nonlinear_iterations().is_none());
            assert!(
                (point
                    .jvp(&support::derivative_products::tangent(point.identity(), &[2.]).unwrap())
                    .unwrap()
                    .derivative()
                    .fixture_coordinates()
                    .as_slice()[0]
                    - 2. * (4. * p + 1.))
                    .abs()
                    < 1e-9
            );
            assert!(
                (point
                    .vjp(&support::derivative_products::cotangent(point.identity(), &[1.]).unwrap())
                    .unwrap()
                    .derivative()
                    .fixture_coordinates()
                    .as_slice()[0]
                    - (4. * p + 1.))
                    .abs()
                    < 1e-9
            );
        }
        assert!(
            (program
                .vjp(&support::derivative_products::cotangent(program.identity(), &[1.]).unwrap())
                .unwrap()
                .derivative()
                .fixture_coordinates()
                .as_slice()[0]
                - 13.)
                .abs()
                < 1e-9
        );
    }
}

#[test]
fn one_finite_point_pairs_complex_field_and_real_loss_outputs() {
    use eqiora::differentiation::{
        AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
    };
    use eqiora::solver::{LinearOperatorProperties, LinearSolveRequest};

    // 2x-2y=4p+Re(q), 2x=2p+Im(q). Thus x=p+Im(q)/2,
    // y=-p+(Im(q)-Re(q))/2, J=x²+y²+p. The full real Jacobian
    // at p=3,q=2-4i has rows (1,0,1/2), (-1,-1/2,1/2), (15,6,-5).
    let (document, plan, _) = solve_with(
        "model M(){parameter p:1=3;parameter q:complex<1>=math.complex(2,-4);variable z:complex<1>;relation r{math.complex(1,2)*z+math.conj(z)=math.complex(4,2)*p+q;}observable loss:1=math.abs2(z)+p;}",
        false,
    );
    let inputs = [
        document.parameter_ref("p").unwrap().id(),
        document.parameter_ref("q").unwrap().id(),
    ];
    let outputs = [
        document.field_ref("z").unwrap().id().erase(),
        document.observable_ref("loss").unwrap().id().erase(),
    ];
    let initial = plan.initial_state(&[]).unwrap();
    let point = plan
        .differentiate(&initial, &inputs, None, &outputs, &FaerLinearSolver)
        .unwrap();
    let close = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            // This nonsingular 2x2 system with bounded integer coefficients
            // uses SparseLU and binary64; 1e-10 covers roundoff in the quadratic loss.
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
    };
    close(point.linearization().output_values(), &[1., -6., 40.]);
    assert_eq!(
        point.linearization().state_scalar_domain(),
        eqiora::ScalarDomain::Real
    );
    let validated = plan
        .validate_differentiated_values(&initial, &inputs, None, &outputs, &[1., -6.])
        .unwrap();
    close(validated.output_values(), &[1., -6., 40.]);
    assert_eq!(validated.initial_residual_norm(), None);
    assert_eq!(validated, *point.linearization());
    assert!(
        plan.validate_differentiated_values(&initial, &inputs, None, &outputs, &[1., 6.],)
            .is_err()
    );
    close(
        point.linearization().relation().design_values(),
        &[3., 2., -4.],
    );
    let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
        point.linearization().relation(),
        point.linearization(),
        point.linearization().relation().state_jacobian(),
        point.linearization().state_scalar_domain(),
        point.linearization().residual_target(),
    )
    .unwrap();
    let solve = LinearSolveRequest::new(&FaerLinearSolver, plan.linear());
    let forward = forward_output_sensitivity(
        &accepted,
        &[1., 2., -1.],
        LinearOperatorProperties::General,
        solve,
    )
    .unwrap();
    close(forward.output_tangent(), &[0.5, -2.5, 32.]);
    let reverse = adjoint_output_gradient(
        &accepted,
        &[2., -3., 0.5],
        LinearOperatorProperties::General,
        solve,
    )
    .unwrap();
    close(reverse.gradient(), &[12.5, 4.5, -3.]);
    let reordered = plan
        .differentiate(
            &initial,
            &inputs,
            None,
            &[outputs[1], outputs[0]],
            &FaerLinearSolver,
        )
        .unwrap();
    close(reordered.linearization().output_values(), &[40., 1., -6.]);
    let selected_inputs = [
        document.parameter_ref("p").unwrap(),
        document.parameter_ref("q").unwrap(),
    ];
    let selected_outputs = [
        document.field_ref("z").unwrap().erase(),
        document.observable_ref("loss").unwrap().erase(),
    ];
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
    let program = eqiora::api::DifferentiableProgram::compile(
        resolved.clone(),
        &selected_inputs,
        &selected_outputs,
        Default::default(),
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    close(
        program.primal().output().fixture_coordinates().as_slice(),
        &[1., -6., 40.],
    );
    let primal = program.primal();
    assert_eq!(primal.output().signatures(), program.identity().outputs());
    for (selection, expected) in selected_outputs.iter().zip([(1., -6.), (40., 0.)]) {
        let leaf = primal.output().get(selection.id()).unwrap();
        assert_eq!(leaf.signature().model(), selection.model());
        assert_eq!(leaf.components().collect::<Vec<_>>(), [expected]);
        assert_eq!(leaf.coefficient(0).unwrap().component(0).unwrap(), expected);
    }
    assert!(program.evaluate(primal.output()).is_err());
    assert_eq!(
        program
            .jvp(
                &support::derivative_products::tangent(program.identity(), &[1., 2., -1.]).unwrap()
            )
            .unwrap()
            .output(),
        primal.output()
    );
    assert_eq!(
        program
            .vjp(
                &support::derivative_products::cotangent(program.identity(), &[2., -3., 0.5])
                    .unwrap()
            )
            .unwrap()
            .output(),
        primal.output()
    );
    assert_complex_mapped_products(&program);
    assert_typed_complex_values(&program);
    let direction = eqiora::api::DifferentiableTangent::from_components(
        program.identity().inputs(),
        &[&[(1., 0.)], &[(2., -1.)]],
    )
    .unwrap();
    let seed = eqiora::api::DifferentiableCotangent::from_components(
        program.identity().outputs(),
        &[&[(2., -3.)], &[(0.5, 0.)]],
    )
    .unwrap();
    let forward = program.jvp(&direction).unwrap();
    let reverse = program.vjp(&seed).unwrap();
    assert_eq!(
        forward.derivative().base_signatures(),
        program.identity().outputs()
    );
    assert_eq!(
        reverse.derivative().base_signatures(),
        program.identity().inputs()
    );
    let (re, im) = forward.derivative().components(0).unwrap().next().unwrap();
    close(&[re, im], &[0.5, -2.5]);
    let (re, im) = reverse.derivative().components(1).unwrap().next().unwrap();
    close(&[re, im], &[4.5, -3.]);

    close(
        program
            .jvp(
                &support::derivative_products::tangent(program.identity(), &[1., 2., -1.]).unwrap(),
            )
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice(),
        &[0.5, -2.5, 32.],
    );
    close(
        program
            .vjp(
                &support::derivative_products::cotangent(program.identity(), &[2., -3., 0.5])
                    .unwrap(),
            )
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice(),
        &[12.5, 4.5, -3.],
    );
    let parameters = eqiora::api::DifferentiableValues::from_components(
        program.identity().inputs(),
        &[&[(1., 0.)], &[(2., -4.)]],
    )
    .unwrap();
    let next = program.evaluate(&parameters).unwrap();
    close(
        next.primal().output().fixture_coordinates().as_slice(),
        &[-1., -4., 18.],
    );
    close(
        next.jvp(&support::derivative_products::tangent(next.identity(), &[1., 2., -1.]).unwrap())
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice(),
        &[0.5, -2.5, 20.],
    );
    let reverse_program = eqiora::api::DifferentiableProgram::compile(
        resolved.clone(),
        &selected_inputs,
        &[selected_outputs[1].clone(), selected_outputs[0].clone()],
        Default::default(),
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    close(
        reverse_program
            .primal()
            .output()
            .fixture_coordinates()
            .as_slice(),
        &[40., 1., -6.],
    );
    assert_ne!(program.identity(), reverse_program.identity());
    assert_eq!(
        program.identity().outputs()[0],
        reverse_program.identity().outputs()[1],
        "storage offsets do not change semantic value identity"
    );

    let foreign = ModelDocument::compile(
        "foreign.eqi",
        "model Foreign(){variable z:complex<1>;relation r{z=math.complex(1,2);}}",
    )
    .unwrap();
    for invalid_outputs in [
        vec![],
        vec![selected_outputs[0].clone(), selected_outputs[0].clone()],
        vec![selected_inputs[0].erase()],
        vec![foreign.field_ref("z").unwrap().erase()],
    ] {
        assert!(
            eqiora::api::DifferentiableProgram::compile(
                resolved.clone(),
                &selected_inputs,
                &invalid_outputs,
                Default::default(),
                None,
                &FaerLinearSolver
            )
            .is_err()
        );
    }

    assert!(
        plan.differentiate(&initial, &inputs, None, &[], &FaerLinearSolver)
            .is_err()
    );
    assert!(
        plan.differentiate(
            &initial,
            &inputs,
            None,
            &[outputs[0], outputs[0]],
            &FaerLinearSolver
        )
        .is_err()
    );
    assert!(
        plan.differentiate(
            &initial,
            &inputs,
            None,
            &[inputs[0].erase()],
            &FaerLinearSolver
        )
        .is_err()
    );
}

#[test]
fn shaped_complex_observable_keeps_channel_order_and_conjugate_partials() {
    use eqiora::api::DifferentiableProgram;
    // O=conj(p*c), with real p=3 and c=(1+2i,3-4i).
    let (document, plan, _) = solve(
        "model M(){parameter p:1=3;parameter c:array<complex<1>,2>=[math.complex(1,2),math.complex(3,-4)];variable z:array<complex<1>,2>;relation r{z=p*c;}observable output:array<complex<1>,2>=[math.conj(z[0]),math.conj(z[1])];}",
    );
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[
            document.parameter_ref("p").unwrap(),
            document.parameter_ref("c").unwrap(),
        ],
        &[document.observable_ref("output").unwrap().erase()],
        Default::default(),
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let close = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            // Identity state operator: 1e-10 bounds binary64 product roundoff
            // for this bounded analytic fixture without a conditioning factor.
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
    };
    let signatures = program.identity().inputs();
    assert!(signatures[0].shape().is_empty());
    assert_eq!(signatures[1].shape(), &[2]);
    assert_eq!(
        signatures[1].entity(),
        document.parameter_ref("c").unwrap().id().erase()
    );
    assert_eq!(
        signatures[1].mathematical_type().value_type.scalar_domain(),
        eqiora::ScalarDomain::Complex
    );
    assert_eq!(program.identity().outputs()[0].shape(), &[2]);
    assert_eq!(
        program.identity().outputs()[0]
            .mathematical_type()
            .value_type
            .scalar_domain(),
        eqiora::ScalarDomain::Complex
    );
    assert!(
        program.identity().outputs()[0]
            .coefficient_entities()
            .is_empty()
    );
    assert!(program.identity().outputs()[0].space().is_none());
    assert_eq!(program.identity().input_dimension(), 5);
    assert_eq!(program.identity().output_dimension(), 4);
    close(
        program.primal().output().fixture_coordinates().as_slice(),
        &[3., -6., 9., 12.],
    );
    close(
        program
            .jvp(
                &support::derivative_products::tangent(program.identity(), &[0.5, 1., -1., 2., 3.])
                    .unwrap(),
            )
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice(),
        &[3.5, 2., 7.5, -7.],
    );
    close(
        program
            .vjp(
                &support::derivative_products::cotangent(program.identity(), &[2., -3., 5., 7.])
                    .unwrap(),
            )
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice(),
        &[51., 6., 9., 15., -21.],
    );
}

#[test]
fn finite_output_selection_keeps_equal_shapes_with_different_units_distinct() {
    use eqiora::ir::LinearizedOutput;
    let (document, plan, _) = solve_with(
        "model M(){parameter p:m=3[m];parameter q:m=2[m];variable u:m;relation r{u=p+2*q;}observable loss:m^2=u*u+p*q;}",
        false,
    );
    let inputs = [
        document.parameter_ref("p").unwrap().id(),
        document.parameter_ref("q").unwrap().id(),
    ];
    let field = document.field_ref("u").unwrap().id().erase();
    let loss = document.observable_ref("loss").unwrap().id().erase();
    let initial = plan.initial_state(&[]).unwrap();
    let point = plan
        .differentiate(&initial, &inputs, None, &[field, loss], &FaerLinearSolver)
        .unwrap();
    // u=7[m], J=55[m²]. At p=3,q=2, du=(1,2), dJ=(16,31).
    assert_eq!(point.linearization().output_values(), [7., 55.]);
    // Partial output action: du_state=-3/4, dp=1/4, dq=-1/2.
    let mut tangent = [0.; 2];
    point
        .linearization()
        .jvp(&[-0.75], &[0.25, -0.5], &mut tangent)
        .unwrap();
    assert_eq!(tangent, [-0.75, -11.5]);
    let reversed = plan
        .differentiate(&initial, &inputs, None, &[loss, field], &FaerLinearSolver)
        .unwrap();
    assert_eq!(reversed.linearization().output_values(), [55., 7.]);
    let parameters = [
        document.parameter_ref("p").unwrap(),
        document.parameter_ref("q").unwrap(),
    ];
    let outputs = [
        document.field_ref("u").unwrap().erase(),
        document.observable_ref("loss").unwrap().erase(),
    ];
    let program = eqiora::api::DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &parameters,
        &outputs,
        Default::default(),
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    let signatures = program.identity().outputs();
    assert_eq!(signatures.len(), 2);
    assert_eq!(signatures[0].shape(), signatures[1].shape());
    assert_ne!(signatures[0].entity(), signatures[1].entity());
    for (signature, power) in signatures.iter().zip([1, 2]) {
        assert_eq!(
            signature.mathematical_type().value_type.dimension(),
            eqiora::DimExponents::from_integers([0, power, 0, 0, 0, 0, 0]).unwrap()
        );
    }
    let literals = signatures
        .iter()
        .zip([7., 55.])
        .map(|(signature, value)| {
            eqiora::ValueLiteral::from_real(signature.mathematical_type().value_type.clone(), value)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let typed = eqiora::api::DifferentiableValues::from_literals(signatures, &literals).unwrap();
    assert_eq!(program.primal().output(), &typed);
    assert_eq!(
        typed.get(field).unwrap().coefficient(0).unwrap(),
        literals[0]
    );
    assert_eq!(
        typed.get(loss).unwrap().coefficient(0).unwrap(),
        literals[1]
    );
    assert!(
        eqiora::api::DifferentiableValues::from_literals(
            signatures,
            &[literals[1].clone(), literals[0].clone()]
        )
        .is_err()
    );
    assert_eq!(program.primal().output().fixture_coordinates(), [7., 55.]);
    assert_eq!(
        program
            .jvp(&support::derivative_products::tangent(program.identity(), &[0.25, -0.5]).unwrap())
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice(),
        [-0.75, -11.5]
    );
    assert_eq!(
        program
            .vjp(&support::derivative_products::cotangent(program.identity(), &[2., -3.]).unwrap())
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice(),
        [-46., -89.]
    );
    let next = program
        .evaluate(&support::real_point(&program, &[4., 1.]).unwrap())
        .unwrap();
    assert_eq!(next.primal().output().fixture_coordinates(), [6., 40.]);
    assert_eq!(
        next.jvp(&support::derivative_products::tangent(next.identity(), &[0.25, -0.5]).unwrap())
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice(),
        [-0.75, -10.75]
    );
}

#[test]
fn complex_zero_residual_does_not_prove_differentiable_regularity() {
    use eqiora::api::DifferentiableProgram;
    // At p=0 every z satisfies p*z=0. A successful zero-residual primal
    // cannot establish a locally unique differentiable solution map.
    let (document, plan, _) = solve(
        "model M(){parameter p:1=0;variable z:complex<1>;relation r{p*z=0;}observable output:1=math.abs2(z);}",
    );
    let error = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[document.parameter_ref("p").unwrap()],
        &[document.observable_ref("output").unwrap().erase()],
        Default::default(),
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap_err();
    assert!(
        error
            .iter()
            .any(|error| error.message().contains("Jacobian is singular")),
        "{error:?}"
    );
}

#[test]
fn common_sensitivity_keeps_mixed_real_and_shaped_complex_coordinates() {
    use eqiora::api::DifferentiableProgram;
    // z_k=k*p*(1-i), k=1..6, and w=2p. Thus
    // J=sum|z_k|²+w²+p=(2*91+4)p²+p=186p²+p.
    let source = "model M(){parameter p:1=3;parameter a:complex<1>=math.complex(1,2);variable z:array<complex<1>,6>;variable w:1;relation r{a*z[0]+math.conj(z[0])=math.complex(4,2)*p;a*z[1]+math.conj(z[1])=math.complex(8,4)*p;a*z[2]+math.conj(z[2])=math.complex(12,6)*p;a*z[3]+math.conj(z[3])=math.complex(16,8)*p;a*z[4]+math.conj(z[4])=math.complex(20,10)*p;a*z[5]+math.conj(z[5])=math.complex(24,12)*p;w=2*p;}observable output:1=math.abs2(z[0])+math.abs2(z[1])+math.abs2(z[2])+math.abs2(z[3])+math.abs2(z[4])+math.abs2(z[5])+w*w+p;}";
    let (document, plan, _) = solve_with(source, false);
    assert_eq!(plan.symbols().len(), 2);
    assert_eq!(plan.coordinate_count(), 13);
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[document.parameter_ref("p").unwrap()],
        &[document.observable_ref("output").unwrap().erase()],
        Default::default(),
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    for p in [3., 5.] {
        let point = program
            .evaluate(&support::real_point(&program, &[p]).unwrap())
            .unwrap();
        assert_eq!(point.accepted_unknowns().len(), 13);
        assert!(
            (point.primal().output().fixture_coordinates()[0] - (186. * p * p + p)).abs() < 1e-8
        );
        assert!(
            (point
                .jvp(&support::derivative_products::tangent(point.identity(), &[2.]).unwrap())
                .unwrap()
                .derivative()
                .fixture_coordinates()
                .as_slice()[0]
                - 2. * (372. * p + 1.))
                .abs()
                < 1e-8
        );
        assert!(
            (point
                .vjp(&support::derivative_products::cotangent(point.identity(), &[1.]).unwrap())
                .unwrap()
                .derivative()
                .fixture_coordinates()
                .as_slice()[0]
                - (372. * p + 1.))
                .abs()
                < 1e-8
        );
    }
}

#[test]
fn complex_parameter_coordinates_preserve_nonholomorphic_partials_and_real_pairing() {
    use eqiora::api::DifferentiableProgram;
    // c=u+iv gives z=(u+2v)+i(-u+3v), J=2u²-2uv+13v².
    // At c=3+2i: J=58 and grad J=(8,46), so direction (1,-2) gives -84.
    let (document, plan, _) = solve_with(
        "model M(){parameter c:complex<1>=math.complex(3,2);variable z:complex<1>;relation r{math.complex(1,2)*z+math.conj(z)=math.complex(4,2)*c;}observable output:1=math.abs2(z);}",
        false,
    );
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[document.parameter_ref("c").unwrap()],
        &[document.observable_ref("output").unwrap().erase()],
        Default::default(),
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    assert_eq!(program.identity().input_dimension(), 2);
    let parameters = eqiora::api::DifferentiableValues::from_components(
        program.identity().inputs(),
        &[&[(3., 2.)]],
    )
    .unwrap();
    let point = program.evaluate(&parameters).unwrap();
    assert!((point.primal().output().fixture_coordinates()[0] - 58.).abs() < 1e-10);
    let residual = point.residual_jvp(&[0., 0.], &[1., -2.]).unwrap();
    assert_eq!(residual, [-8., 6.]);
    assert!(
        (point
            .jvp(&support::derivative_products::tangent(point.identity(), &[1., -2.]).unwrap())
            .unwrap()
            .derivative()
            .fixture_coordinates()
            .as_slice()[0]
            + 84.)
            .abs()
            < 1e-10
    );
    for (actual, expected) in point
        .vjp(&support::derivative_products::cotangent(point.identity(), &[1.]).unwrap())
        .unwrap()
        .derivative()
        .fixture_coordinates()
        .as_slice()
        .iter()
        .zip([8., 46.])
    {
        assert!((actual - expected).abs() < 1e-10);
    }
    assert!(
        eqiora::api::DifferentiableValues::from_components(program.identity().inputs(), &[&[]],)
            .is_err()
    );
}

#[test]
fn shaped_parameters_keep_channel_parts_selection_order_and_point_ownership() {
    use eqiora::api::DifferentiableProgram;
    // z=p*c, J=p² sum|c|²+p. These are exact independent values at two points.
    for (domain, initializer, points) in [
        (
            "complex<1>",
            "[math.complex(1,2),math.complex(3,4)]",
            vec![
                (
                    vec![2., 1., 2., 3., 4.],
                    122.,
                    vec![121., 8., 16., 24., 32.],
                ),
                (
                    vec![3., 2., -1., -2., 1.],
                    93.,
                    vec![61., 36., -18., -36., 18.],
                ),
            ],
        ),
        (
            "1",
            "[1,3]",
            vec![
                (vec![2., 1., 3.], 42., vec![41., 8., 24.]),
                (vec![3., 2., -2.], 75., vec![49., 36., -36.]),
            ],
        ),
    ] {
        let source = format!(
            "model M(){{parameter p:1=2;parameter c:array<{domain},2>={initializer};variable z:array<complex<1>,2>;relation r{{z=p*c;}}observable output:1=math.abs2(z[0])+math.abs2(z[1])+p;}}"
        );
        let (document, plan, _) = solve(&source);
        let mut original_point = None;
        for reversed in [false, true] {
            let mut inputs = vec![
                document.parameter_ref("p").unwrap(),
                document.parameter_ref("c").unwrap(),
            ];
            if reversed {
                inputs.reverse();
            }
            let program = DifferentiableProgram::compile(
                ResolvedCommonPlan::Algebraic(Box::new(plan.clone())),
                &inputs,
                &[document.observable_ref("output").unwrap().erase()],
                Default::default(),
                None,
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
            let point = program.default_point();
            if let Some(original) = &original_point {
                assert!(
                    program.evaluate(original).is_err(),
                    "input order is semantic"
                );
            } else {
                original_point = Some(point.clone());
            }
            let c = point
                .get(document.parameter_ref("c").unwrap().id().erase())
                .unwrap();
            let expected_components = if domain == "complex<1>" {
                vec![(1., 2.), (3., 4.)]
            } else {
                vec![(1., 0.), (3., 0.)]
            };
            assert_eq!(c.signature().shape(), [2]);
            assert_eq!(c.components().collect::<Vec<_>>(), expected_components);
            let literals = (0..2)
                .map(|index| point.leaf(index).unwrap().coefficient(0).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                eqiora::api::DifferentiableValues::from_literals(point.signatures(), &literals)
                    .unwrap(),
                *point
            );
            for (values, expected, gradient) in &points {
                let mut leaves = [
                    vec![(values[0], 0.)],
                    if domain == "complex<1>" {
                        vec![(values[1], values[2]), (values[3], values[4])]
                    } else {
                        vec![(values[1], 0.), (values[2], 0.)]
                    },
                ];
                let mut gradient = gradient.clone();
                if reversed {
                    leaves.reverse();
                    gradient.rotate_left(1);
                }
                let parameters = eqiora::api::DifferentiableValues::from_components(
                    program.identity().inputs(),
                    &leaves.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                )
                .unwrap();
                let point = program.evaluate(&parameters).unwrap();
                assert_eq!(program.identity().input_dimension(), values.len());
                assert!((point.primal().output().fixture_coordinates()[0] - expected).abs() < 1e-9);
                let direction = vec![1.; values.len()];
                assert!(
                    (point
                        .jvp(
                            &support::derivative_products::tangent(point.identity(), &direction)
                                .unwrap()
                        )
                        .unwrap()
                        .derivative()
                        .fixture_coordinates()
                        .as_slice()[0]
                        - gradient.iter().sum::<f64>())
                    .abs()
                        < 1e-9
                );
                for (actual, expected) in point
                    .vjp(&support::derivative_products::cotangent(point.identity(), &[1.]).unwrap())
                    .unwrap()
                    .derivative()
                    .fixture_coordinates()
                    .as_slice()
                    .iter()
                    .zip(&gradient)
                {
                    assert!((actual - expected).abs() < 1e-9);
                }
            }
            assert!(
                (program.primal().output().fixture_coordinates()[0] - points[0].1).abs() < 1e-9
            );
        }
    }
}

#[test]
fn nominal_complex_map_parameter_keeps_both_matrix_axes_and_conjugate_pairing() {
    use eqiora::api::DifferentiableProgram;
    // At A=I, x=(1+i,2-i), J=7. dJ=-2 Re(x^H dA x), giving
    // row-major real/imaginary derivatives [-4,0,-2,-6,-2,6,-10,0].
    let source = "space V=orthonormal(first,second);model M(){parameter a:map<complex<1>,V,V>=linear_map(V,V,[[1,0],[0,1]]);parameter b:coordinates<complex<1>,V>=coordinates(V,[math.complex(1,1),math.complex(2,-1)]);variable z:coordinates<complex<1>,V>;relation r{apply(a,z)=b;}observable output:1=math.real(pair(adjoint(z),z));}";
    let (document, plan, _) = solve(source);
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[document.parameter_ref("a").unwrap()],
        &[document.observable_ref("output").unwrap().erase()],
        Default::default(),
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    assert_eq!(program.identity().input_dimension(), 8);
    for scale in [1., 2.] {
        let parameters = eqiora::api::DifferentiableValues::from_components(
            program.identity().inputs(),
            &[&[(scale, 0.), (0., 0.), (0., 0.), (scale, 0.)]],
        )
        .unwrap();
        let point = program.evaluate(&parameters).unwrap();
        assert!(
            (point.primal().output().fixture_coordinates()[0] - 7. / (scale * scale)).abs() < 1e-10
        );
        let reverse = point
            .vjp(&support::derivative_products::cotangent(point.identity(), &[1.]).unwrap())
            .unwrap();
        for (actual, expected) in reverse
            .derivative()
            .fixture_coordinates()
            .as_slice()
            .iter()
            .zip([-4., 0., -2., -6., -2., 6., -10., 0.])
        {
            assert!((actual - expected / (scale * scale * scale)).abs() < 1e-10);
        }
        assert!(
            (point
                .jvp(
                    &support::derivative_products::tangent(
                        point.identity(),
                        &[0., 0., 0., 1., 0., 0., 0., 0.]
                    )
                    .unwrap()
                )
                .unwrap()
                .derivative()
                .fixture_coordinates()
                .as_slice()[0]
                + 6. / (scale * scale * scale))
                .abs()
                < 1e-10
        );
    }
}

fn assert_complex_mapped_products(program: &eqiora::api::DifferentiableProgram) {
    use eqiora::api::{
        EvaluationMapExecutionPolicy, EvaluationMapPlan, EvaluationMapProducts,
        EvaluationMapRetention,
    };
    use std::sync::Arc;
    // z(p,q)=(p+Im(q)/2) + i*(-p+(Im(q)-Re(q))/2), J=abs2(z)+p.
    // At q=2-4i and p=[1,3], sum(J)=58. Its p gradients are [7,15],
    // and the shared real-coordinate q gradient is (4,-5)+(6,-5)=(10,-10).
    for retention in [
        EvaluationMapRetention::Retain,
        EvaluationMapRetention::Recompute,
    ] {
        let policy = EvaluationMapExecutionPolicy::new(
            2,
            eqiora_realization::Target::HostCpu {
                threads: NonZeroUsize::new(2).unwrap(),
            },
            retention,
            1 << 30,
        )
        .unwrap();
        let map = EvaluationMapPlan::from_partition(
            Arc::new(program.clone()),
            &eqiora::api::DifferentiableValues::from_components(
                &program.identity().inputs()[1..],
                &[&[(2., -4.)]],
            )
            .unwrap(),
            &[1., 3.].map(|value| {
                eqiora::api::DifferentiableValues::from_components(
                    &program.identity().inputs()[..1],
                    &[&[(value, 0.)]],
                )
                .unwrap()
            }),
            policy,
        )
        .unwrap()
        .execute()
        .unwrap();
        for (index, expected) in [[-1., -4., 18.], [1., -6., 40.]].iter().enumerate() {
            let member = match retention {
                EvaluationMapRetention::Retain => map.evaluation(index).unwrap().clone(),
                EvaluationMapRetention::Recompute => map.recompute(index).unwrap(),
            };
            assert_eq!(
                member.primal().output().signatures(),
                program.identity().outputs()
            );
            assert_eq!(
                map.evidence(index).unwrap().implementation(),
                eqiora::api::DerivativeImplementation::OperatorIr
            );
            assert_eq!(*map.evidence(index).unwrap(), *member.primal().evidence());
            for (actual, expected) in member
                .primal()
                .output()
                .fixture_coordinates()
                .iter()
                .zip(expected)
            {
                assert!((actual - expected).abs() < 1e-10);
            }
        }
        let products = EvaluationMapProducts::new(&map, &[2], &[], &[0], 1 << 30).unwrap();
        let forward =
            support::derivative_products::mapped_jvp(&products, &[2., -1.], &[1., 1.], 1).unwrap();
        for (actual, expected) in forward
            .products()
            .iter()
            .flat_map(|p| p.derivative().fixture_coordinates())
            .zip([0.5, -2.5, 20., 0.5, -2.5, 32.])
        {
            assert!((actual - expected).abs() < 1e-10);
        }
        let reverse =
            support::derivative_products::mapped_vjp(&products, &[0., 0., 1., 0., 0., 1.]).unwrap();
        assert_eq!(
            reverse.shared_signatures(),
            &program.identity().inputs()[1..]
        );
        assert_eq!(
            reverse.mapped_signatures(),
            &program.identity().inputs()[..1]
        );
        assert_eq!(
            reverse.shared_cotangents()[0].base_signatures(),
            reverse.shared_signatures()
        );
        assert_eq!(
            reverse.shared_cotangents()[0].components(0).unwrap().len(),
            1
        );
        assert!(
            reverse
                .mapped_cotangents()
                .iter()
                .all(|value| value.base_signatures() == reverse.mapped_signatures())
        );
        for (actual, expected) in reverse.shared_cotangents()[0]
            .fixture_coordinates()
            .iter()
            .zip([10., -10.])
        {
            assert!((actual - expected).abs() < 1e-10);
        }
        for (actual, expected) in reverse
            .mapped_cotangents()
            .iter()
            .flat_map(FixtureCoordinates::fixture_coordinates)
            .zip([7., 15.])
        {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}

fn assert_typed_complex_values(program: &eqiora::api::DifferentiableProgram) {
    use eqiora::api::DifferentiableValues;
    let signatures = program.identity().inputs();
    let point = program.default_point();
    assert_eq!(point.signatures(), signatures);
    assert_eq!(
        point.leaf(0).unwrap().components().collect::<Vec<_>>(),
        [(3., 0.)]
    );
    assert_eq!(
        point.leaf(1).unwrap().components().collect::<Vec<_>>(),
        [(2., -4.)]
    );
    for index in 0..2 {
        assert_eq!(
            point.leaf(index).unwrap().signature().model(),
            program.identity().model()
        );
        assert!(point.leaf(index).unwrap().coefficient(1).is_err());
    }
    let literals = [
        point.leaf(0).unwrap().coefficient(0).unwrap(),
        point.leaf(1).unwrap().coefficient(0).unwrap(),
    ];
    let rebound = DifferentiableValues::from_literals(signatures, &literals).unwrap();
    assert_eq!(&rebound, point);
    assert_eq!(
        program.evaluate(&rebound).unwrap().primal(),
        program.primal()
    );
    let parts: [&[(f64, f64)]; 2] = [&[(1., 0.)], &[(2., -4.)]];
    let changed = DifferentiableValues::from_components(signatures, &parts).unwrap();
    assert_eq!(
        changed.leaf(1).unwrap().components().collect::<Vec<_>>(),
        parts[1]
    );
    for (actual, expected) in program
        .evaluate(&changed)
        .unwrap()
        .primal()
        .output()
        .fixture_coordinates()
        .iter()
        .zip([-1., -4., 18.])
    {
        assert!((actual - expected).abs() < 1e-10);
    }
    assert!(changed.leaf(2).is_none());
    for invalid in [
        [&[(1., 2.)][..], &[(2., -4.)][..]],
        [&[(f64::NAN, 0.)][..], &[(2., -4.)][..]],
        [&[(1., 0.)][..], &[][..]],
    ] {
        assert!(DifferentiableValues::from_components(signatures, &invalid).is_err());
    }
    let duplicate = [signatures[0].clone(), signatures[0].clone()];
    assert!(
        DifferentiableValues::from_components(&duplicate, &[&[(1., 0.)], &[(2., 0.)]]).is_err()
    );
    let (foreign_document, foreign_plan, _) = solve(
        "model M(){parameter other:1=3;parameter q:complex<1>=math.complex(2,-4);variable z:complex<1>;relation r{z=other+q;}}",
    );
    let foreign = eqiora::api::DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(foreign_plan)),
        &[
            foreign_document.parameter_ref("other").unwrap(),
            foreign_document.parameter_ref("q").unwrap(),
        ],
        &[foreign_document.field_ref("z").unwrap().erase()],
        Default::default(),
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    assert_eq!(
        foreign.default_point().fixture_coordinates(),
        point.fixture_coordinates()
    );
    assert!(
        program.evaluate(foreign.default_point()).is_err(),
        "equal default coordinates do not authenticate a foreign Model"
    );
    let mixed = [
        signatures[0].clone(),
        foreign.identity().inputs()[0].clone(),
    ];
    assert_ne!(mixed[0].entity(), mixed[1].entity());
    assert_ne!(mixed[0].model(), mixed[1].model());
    assert!(DifferentiableValues::from_components(&mixed, &[&[(1., 0.)], &[(2., 0.)]]).is_err());
}

#[test]
fn parameter_points_rederive_the_actual_domain_without_changing_solver_controls() {
    use eqiora::differentiation::{
        AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
    };
    use eqiora::solver::{LinearOperatorProperties, LinearSolveRequest};
    // Independently, (1+p)x=1 and (1-p)y=2 for z=x+iy.
    // Start once from each domain, including the Reference iterative path.
    for (canonical, reference) in [(0., true), (3., false)] {
        let source = format!(
            "model Profile(){{parameter p:1={canonical};variable z:complex<1>;relation r{{z+p*math.conj(z)=math.complex(1,2);}}}}"
        );
        let (document, plan, _) = solve_with(&source, reference);
        let backend: &dyn LinearSolverBackend = if reference {
            &REFERENCE_LINEAR_SOLVER
        } else {
            &FaerLinearSolver
        };
        let initial = plan.initial_state(&[]).unwrap();
        let inputs = [document.parameter_ref("p").unwrap().id()];
        let outputs = [document.field_ref("z").unwrap().id().erase()];
        for p in [0_f64, 3., 0.] {
            let expected = [1. / (1. + p), 2. / (1. - p)];
            let derivative = [-1. / (1. + p).powi(2), 2. / (1. - p).powi(2)];
            let point = plan
                .differentiate(&initial, &inputs, Some(&[p]), &outputs, backend)
                .unwrap();
            let linear = point.linearization();
            let domain = if p == 0. {
                eqiora::ScalarDomain::Complex
            } else {
                eqiora::ScalarDomain::Real
            };
            assert_eq!(linear.state_scalar_domain(), domain);
            for (actual, expected) in linear.output_values().iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
            let replay = plan
                .validate_differentiated_values(&initial, &inputs, Some(&[p]), &outputs, &expected)
                .unwrap();
            assert_eq!(replay.state_scalar_domain(), domain);
            let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
                linear.relation(),
                linear,
                linear.relation().state_jacobian(),
                domain,
                linear.residual_target(),
            )
            .unwrap();
            let solve = LinearSolveRequest::new(backend, plan.linear());
            let forward = forward_output_sensitivity(
                &accepted,
                &[1.],
                LinearOperatorProperties::General,
                solve,
            )
            .unwrap();
            for (actual, expected) in forward.output_tangent().iter().zip(derivative) {
                assert!((actual - expected).abs() < 1e-12);
            }
            let reverse = adjoint_output_gradient(
                &accepted,
                &[3., -2.],
                LinearOperatorProperties::General,
                solve,
            )
            .unwrap();
            assert!(
                (reverse.gradient()[0] - (3. * derivative[0] - 2. * derivative[1])).abs() < 1e-12
            );
            assert_eq!(
                reverse.adjoint().report().solver_provider(),
                backend.provider()
            );
            assert!(
                plan.validate_differentiated_values(
                    &initial,
                    &inputs,
                    Some(&[p]),
                    &outputs,
                    &[expected[0], -expected[1]]
                )
                .is_err()
            );
        }
        // The crossing permits a different representation, not a singular state.
        assert!(
            plan.differentiate(&initial, &inputs, Some(&[1.]), &outputs, backend)
                .is_err()
        );
        assert!(
            plan.validate_differentiated_values(
                &initial,
                &inputs,
                Some(&[1.]),
                &outputs,
                &[0.5, 0.]
            )
            .is_err()
        );
    }
}
