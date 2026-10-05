//! One mathematical Field retains both parts through the ordinary finite lifecycle.
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

fn solve(source: &str) -> (ModelDocument, CommonAlgebraicPlan, CommonResult) {
    solve_with(source, true)
}

fn solve_with(source: &str, complex: bool) -> (ModelDocument, CommonAlgebraicPlan, CommonResult) {
    let backend: &dyn LinearSolverBackend = if complex {
        &REFERENCE_LINEAR_SOLVER
    } else {
        &FaerLinearSolver
    };
    let document = ModelDocument::compile("complex.eqi", source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        if complex {
            LinearSolver::BiConjugateGradientStabilized
        } else {
            LinearSolver::SparseLu
        },
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(if complex {
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
        eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
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
            &document.observable_ref("output").unwrap(),
            None,
            backend,
        )
        .unwrap();
        for p in [3., 5.] {
            let point = program.evaluate(&[p]).unwrap();
            assert!((point.accepted_unknowns()[0] - p).abs() < 1e-10);
            assert!((point.accepted_unknowns()[1] + p).abs() < 1e-10);
            let primal = point.primal();
            assert!((primal.output()[0] - (2. * p * p + p)).abs() < 1e-9);
            assert!(primal.evidence().primal_solve().is_some());
            assert!(primal.evidence().nonlinear_iterations().is_none());
            assert!((point.jvp(&[2.]).unwrap().tangent()[0] - 2. * (4. * p + 1.)).abs() < 1e-9);
            assert!((point.vjp(&[1.]).unwrap().input_cotangent()[0] - (4. * p + 1.)).abs() < 1e-9);
        }
        assert!((program.vjp(&[1.]).unwrap().input_cotangent()[0] - 13.).abs() < 1e-9);
    }
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
        &document.observable_ref("output").unwrap(),
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
        &document.observable_ref("output").unwrap(),
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    for p in [3., 5.] {
        let point = program.evaluate(&[p]).unwrap();
        assert_eq!(point.accepted_unknowns().len(), 13);
        assert!((point.primal().output()[0] - (186. * p * p + p)).abs() < 1e-8);
        assert!((point.jvp(&[2.]).unwrap().tangent()[0] - 2. * (372. * p + 1.)).abs() < 1e-8);
        assert!((point.vjp(&[1.]).unwrap().input_cotangent()[0] - (372. * p + 1.)).abs() < 1e-8);
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
        &document.observable_ref("output").unwrap(),
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    assert_eq!(program.identity().input_dimension(), 2);
    let point = program.evaluate(&[3., 2.]).unwrap();
    assert!((point.primal().output()[0] - 58.).abs() < 1e-10);
    let residual = point.residual_jvp(&[0., 0.], &[1., -2.]).unwrap();
    assert_eq!(residual, [-8., 6.]);
    assert!((point.jvp(&[1., -2.]).unwrap().tangent()[0] + 84.).abs() < 1e-10);
    for (actual, expected) in point
        .vjp(&[1.])
        .unwrap()
        .input_cotangent()
        .iter()
        .zip([8., 46.])
    {
        assert!((actual - expected).abs() < 1e-10);
    }
    assert!(program.evaluate(&[3.]).is_err());
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
                &document.observable_ref("output").unwrap(),
                None,
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
            for (values, expected, gradient) in &points {
                let (mut values, mut gradient) = (values.clone(), gradient.clone());
                if reversed {
                    values.rotate_left(1);
                    gradient.rotate_left(1);
                }
                let point = program.evaluate(&values).unwrap();
                assert_eq!(program.identity().input_dimension(), values.len());
                assert!((point.primal().output()[0] - expected).abs() < 1e-9);
                let direction = vec![1.; values.len()];
                assert!(
                    (point.jvp(&direction).unwrap().tangent()[0] - gradient.iter().sum::<f64>())
                        .abs()
                        < 1e-9
                );
                for (actual, expected) in point
                    .vjp(&[1.])
                    .unwrap()
                    .input_cotangent()
                    .iter()
                    .zip(&gradient)
                {
                    assert!((actual - expected).abs() < 1e-9);
                }
            }
            assert!((program.primal().output()[0] - points[0].1).abs() < 1e-9);
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
        &document.observable_ref("output").unwrap(),
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    assert_eq!(program.identity().input_dimension(), 8);
    for scale in [1., 2.] {
        let point = program
            .evaluate(&[scale, 0., 0., 0., 0., 0., scale, 0.])
            .unwrap();
        assert!((point.primal().output()[0] - 7. / (scale * scale)).abs() < 1e-10);
        let reverse = point.vjp(&[1.]).unwrap();
        for (actual, expected) in reverse
            .input_cotangent()
            .iter()
            .zip([-4., 0., -2., -6., -2., 6., -10., 0.])
        {
            assert!((actual - expected / (scale * scale * scale)).abs() < 1e-10);
        }
        assert!(
            (point
                .jvp(&[0., 0., 0., 1., 0., 0., 0., 0.])
                .unwrap()
                .tangent()[0]
                + 6. / (scale * scale * scale))
                .abs()
                < 1e-10
        );
    }
}
