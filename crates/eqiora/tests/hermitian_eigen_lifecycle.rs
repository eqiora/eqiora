use eqiora::api::ModelDocument;
use eqiora_artifact::ModelEnvelope;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_core::{DimExponents, DynQuantity, ScalarDomain};
use eqiora_numerics::{CommonEigenPlan, CommonEigenRequest, ResolvedCommonPlan};
use eqiora_solver::{HermitianEigenproblem, LinearSolverBackend};
use std::num::NonZeroUsize;

fn request() -> CommonEigenRequest {
    CommonEigenRequest::dense(NonZeroUsize::new(2).unwrap(), 1e-12, 1e-12).unwrap()
}

const STRUCTURAL: &str = r#"
space Modes=orthonormal(first,second);
model Structure() {
 parameter k:map<kg/s^2,Modes,Modes>=linear_map(Modes,Modes,[[4[kg/s^2],4[kg/s^2]],[4[kg/s^2],16[kg/s^2]]]);
 parameter m:map<kg,Modes,Modes>=linear_map(Modes,Modes,[[2[kg],0],[0,8[kg]]]);
 variable u:coordinates<kg^(-1/2),Modes>;
 variable lambda:1/s^2;
 relation modes {apply(k,u)=lambda*apply(m,u);}
}
"#;

const QUANTUM: &str = r#"
space Spin=orthonormal(up,down);
model Quantum() {
 parameter h:map<complex<J>,Spin,Spin>=linear_map(Spin,Spin,[[2[J],math.complex(0[J],-1[J])],[math.complex(0[J],1[J]),2[J]]]);
 variable u:coordinates<complex<1>,Spin>;
 variable lambda:J;
 relation states {apply(h,u)=lambda*u;}
}
"#;

#[test]
fn structural_and_quantum_source_plans_retain_roles_units_and_provider_identity() {
    for (source, dimension, domain) in [
        (
            STRUCTURAL,
            DimExponents::from_integers([0, 0, -2, 0, 0, 0, 0]).unwrap(),
            ScalarDomain::Real,
        ),
        (
            QUANTUM,
            DimExponents::from_integers([1, 2, -2, 0, 0, 0, 0]).unwrap(),
            ScalarDomain::Complex,
        ),
    ] {
        let document = ModelDocument::compile("spectral.eqi", source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let plan = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
        assert_eq!(plan.mode_field().erase(), document.aliases()["u"]);
        assert_eq!(
            plan.eigenvalue_field().erase(),
            document.aliases()["lambda"]
        );
        assert_eq!(plan.operator().value_type().scalar_domain(), domain);
        assert_eq!(
            plan.operator()
                .value_type()
                .dimension()
                .div(plan.metric().value_type().dimension()),
            Some(dimension)
        );
        assert_eq!(
            plan,
            CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap()
        );
        let targeted = request()
            .with_target(DynQuantity::new(2., dimension))
            .unwrap();
        let another = CommonEigenPlan::resolve(&model, targeted, &FaerLinearSolver).unwrap();
        assert_eq!(another.model_digest(), plan.model_digest());
        assert_ne!(another.identity(), plan.identity());
        let wrong = request()
            .with_target(DynQuantity::new(2., DimExponents::DIMENSIONLESS))
            .unwrap();
        assert!(CommonEigenPlan::resolve(&model, wrong, &FaerLinearSolver).is_err());
        assert!(
            CommonEigenPlan::resolve(&model, request(), &eqiora_solver::REFERENCE_LINEAR_SOLVER)
                .is_err()
        );
        let resolved = ResolvedCommonPlan::Eigen(Box::new(another));
        let formulation = resolved.formulation().unwrap();
        assert_eq!(
            formulation.effective(),
            eqiora_numerics::FormulationKind::FiniteHermitianPencil
        );
        assert_eq!(formulation.source_relation(), Some(plan.relation()));
        assert!(formulation.state_coordinates().is_empty());
        assert!(resolved.effective_solver().is_none());
        assert!(resolved.mesh_digest().is_none());
        let bytes = resolved.to_bytes().unwrap();
        let replay = |bytes: &[u8]| {
            ResolvedCommonPlan::from_bytes(
                bytes,
                &FaerLinearSolver,
                eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            )
        };
        let restored = replay(&bytes).unwrap();
        assert_eq!(restored, resolved);
        assert_eq!(restored.as_eigen().unwrap().mode_field(), plan.mode_field());
        let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(wire["schema"], "eqiora.resolved-common-plan/v8");
        for (key, value) in [
            ("count", serde_json::json!(1)),
            ("algorithm", serde_json::json!("unsupported")),
        ] {
            let mut forged = wire.clone();
            forged["spectral"][key] = value;
            assert!(replay(&serde_json::to_vec(&forged).unwrap()).is_err());
        }
        let mut crossed = wire;
        crossed["family"] = serde_json::json!("algebraic");
        assert!(replay(&serde_json::to_vec(&crossed).unwrap()).is_err());
    }
}

#[test]
fn source_plan_rejects_unhandled_constraints_nonlinearity_and_wrong_normalization_units() {
    for source in [
        STRUCTURAL.replace("kg^(-1/2),Modes", "1,Modes"),
        STRUCTURAL.replace("lambda*apply(m,u)", "lambda*lambda*1[s^2]*apply(m,u)"),
        STRUCTURAL.replace("relation modes", "variable extra:1; relation modes"),
        STRUCTURAL.replace(
            "lambda*apply(m,u);",
            "lambda*apply(m,u); inequality(lambda>=0[1/s^2]);",
        ),
        QUANTUM.replace("lambda*u", "lambda*transpose(adjoint(u))"),
    ] {
        // The nonlinear and wrong-unit examples remain well-typed source;
        // admission must not silently omit their mathematical requirements.
        let document = ModelDocument::compile("invalid-spectral.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        assert!(CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).is_err());
    }
}

#[test]
fn source_clients_execute_with_physical_metric_normalization() {
    // Structural: M^-1/2 K M^-1/2 = [[2,1],[1,2]].
    // Quantum: det(H-lambda I) = (2-lambda)^2-1.
    // Thus both independently have eigenvalues 1 and 3 in their own units.
    for source in [STRUCTURAL, QUANTUM] {
        let document = ModelDocument::compile("spectral.eqi", source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let plan = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
        let pencil = HermitianEigenproblem::new(plan.operator(), plan.metric()).unwrap();
        let pairs = FaerLinearSolver.hermitian_eigenpairs(&pencil).unwrap();
        assert_eq!(pairs.len(), 2);
        for ((value, mode), expected) in pairs.iter().zip([1., 3.]) {
            assert!((value.component(0).unwrap().0 - expected).abs() < 1e-12);
            let (residual, normalization) = pencil.eigenpair_defects(value, mode).unwrap();
            assert!(residual < 1e-12);
            assert!(normalization < 1e-12);
        }
        let modes = pairs
            .iter()
            .map(|(_, mode)| mode.clone())
            .collect::<Vec<_>>();
        let projector = pencil.metric_projector(&modes, 1e-12).unwrap();
        // A complete B-orthonormal basis projects to identity even when B != I.
        for (index, expected) in [1., 0., 0., 1.].into_iter().enumerate() {
            let (real, imaginary) = projector.component(index).unwrap();
            assert!((real - expected).abs() < 1e-12);
            assert!(imaginary.abs() < 1e-12);
        }
    }
}

#[test]
fn common_results_replay_selection_evidence_and_partial_convergence() {
    for source in [STRUCTURAL, QUANTUM] {
        let document = ModelDocument::compile("spectral.eqi", source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let plan = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
        let result = plan.run_result(&FaerLinearSolver).unwrap();
        assert_eq!(result.family_name(), "eigen");
        assert_eq!(result.eigen_convergence(), Some("converged"));
        assert_eq!(result.eigen_candidate_counts(), Some((2, 0)));
        assert_eq!(result.eigenpair_count(), 2);
        for (index, expected) in [1., 3.].into_iter().enumerate() {
            let (lambda, _, residual, normalization) = result.eigenpair(index).unwrap();
            assert!((lambda.component(0).unwrap().0 - expected).abs() < 1e-12);
            assert!(residual < 1e-12 && normalization < 1e-12);
        }
        let bytes = result.to_bytes().unwrap();
        let replay = eqiora_numerics::CommonResult::from_bytes(&bytes, result.plan()).unwrap();
        assert_eq!(replay, result);
        assert_eq!(replay.to_bytes().unwrap(), bytes);
        let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(wire["schema"], "eqiora.common-result/v10");
        // Semantic rejection occurs before content-digest and canonical-byte checks.
        let mut forged = wire.clone();
        forged["content"]["payload"]["spectral"]["candidates"][0]["residual"] = 0.5.into();
        let error = eqiora_numerics::CommonResult::from_bytes(
            &serde_json::to_vec(&forged).unwrap(),
            result.plan(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("original-pencil evidence"));
        let mut crossed = wire;
        crossed["content"]["family"] = "algebraic".into();
        assert!(
            eqiora_numerics::CommonResult::from_bytes(
                &serde_json::to_vec(&crossed).unwrap(),
                result.plan()
            )
            .is_err()
        );
        assert!(
            plan.run_result(&eqiora_solver::REFERENCE_LINEAR_SOLVER)
                .is_err()
        );
        let dimension = result.eigenpair(0).unwrap().0.value_type().dimension();
        for (lo, hi, expected_count, status) in
            [(0., 2., 1, "partial"), (4., 5., 0, "not-converged")]
        {
            let controls = request()
                .within_interval([
                    DynQuantity::new(lo, dimension),
                    DynQuantity::new(hi, dimension),
                ])
                .unwrap();
            let partial_plan =
                CommonEigenPlan::resolve(&model, controls, &FaerLinearSolver).unwrap();
            let partial = partial_plan.run_result(&FaerLinearSolver).unwrap();
            assert_eq!(partial.eigenpair_count(), expected_count);
            assert_eq!(partial.eigen_convergence(), Some(status));
            assert_eq!(
                eqiora_numerics::CommonResult::from_bytes(
                    &partial.to_bytes().unwrap(),
                    partial.plan()
                )
                .unwrap(),
                partial
            );
            assert!(eqiora_numerics::CommonResult::from_bytes(&bytes, partial.plan()).is_err());
        }
        let target = CommonEigenRequest::dense(NonZeroUsize::new(1).unwrap(), 1e-12, 1e-12)
            .unwrap()
            .with_target(DynQuantity::new(2.75, dimension))
            .unwrap();
        let targeted = CommonEigenPlan::resolve(&model, target, &FaerLinearSolver)
            .unwrap()
            .run_result(&FaerLinearSolver)
            .unwrap();
        assert_eq!(targeted.eigenpair_count(), 1);
        assert!((targeted.eigenpair(0).unwrap().0.component(0).unwrap().0 - 3.).abs() < 1e-12);
    }
}

#[derive(Debug)]
struct CandidateProbe(&'static str);
impl LinearSolverBackend for CandidateProbe {
    fn provider(&self) -> eqiora_solver::SolverProvider {
        eqiora_solver::SolverProvider::new(
            eqiora_solver::BackendId::new("eqiora.test.spectral-candidates"),
            "1",
            &[],
        )
    }
    fn capabilities(&self) -> eqiora_solver::SolverCapabilities {
        eqiora_solver::SolverCapabilities::reference()
    }
    fn solve_with_execution(
        &self,
        _: &eqiora_solver::LinearProblem<'_>,
        _: eqiora_solver::SolverPlan,
        _: &dyn eqiora_solver::ReplicatedLinearExecution,
    ) -> Result<eqiora_solver::LinearSolution, eqiora_core::Diagnostic> {
        panic!("an eigenproblem must not be executed as an RHS solve")
    }
    fn require_hermitian_eigenproblem(
        &self,
        problem: &HermitianEigenproblem<'_>,
    ) -> Result<(), eqiora_core::Diagnostic> {
        FaerLinearSolver.require_hermitian_eigenproblem(problem)
    }
    fn hermitian_eigenpairs(
        &self,
        problem: &HermitianEigenproblem<'_>,
    ) -> Result<Vec<(eqiora_core::ValueLiteral, eqiora_core::ValueLiteral)>, eqiora_core::Diagnostic>
    {
        use eqiora_core::ValueLiteral;
        let mut pairs = FaerLinearSolver.hermitian_eigenpairs(problem)?;
        match self.0 {
            "unchanged" => {}
            "rotate" => {
                let first = pairs[0].1.clone();
                let second = pairs[1].1.clone();
                for (j, (a, b)) in [(0.6, 0.8), (-0.8, 0.6)].into_iter().enumerate() {
                    let entries: Vec<_> = (0..problem.dimension())
                        .map(|i| {
                            let (xr, xi) = first.component(i).unwrap();
                            let (yr, yi) = second.component(i).unwrap();
                            (a * xr + b * yr, a * xi + b * yi)
                        })
                        .collect();
                    pairs[j].1 = ValueLiteral::new(problem.mode_type().clone(), entries).unwrap();
                }
            }
            "missing" => {
                pairs.pop();
            }
            "empty" => pairs.clear(),
            "wrong-eigenvalue" => {
                pairs[0].0 =
                    ValueLiteral::new(problem.eigenvalue_type().clone(), vec![(0., 0.)]).unwrap()
            }
            "duplicate" => pairs[1] = pairs[0].clone(),
            "zero" => {
                pairs[0].1 = ValueLiteral::new(
                    problem.mode_type().clone(),
                    vec![(0., 0.); problem.dimension()],
                )
                .unwrap()
            }
            "phase-permutation" => {
                for (_, mode) in &mut pairs {
                    let entries: Vec<_> = (0..problem.dimension())
                        .map(|i| {
                            let (re, im) = mode.component(i).unwrap();
                            (-im, re)
                        })
                        .collect();
                    *mode = ValueLiteral::new(problem.mode_type().clone(), entries).unwrap();
                }
                pairs.reverse();
            }
            _ => panic!("unknown probe"),
        }
        Ok(pairs)
    }
}

#[test]
fn result_acceptance_distinguishes_failed_candidates_from_nonunique_modes() {
    let document = ModelDocument::compile("quantum.eqi", QUANTUM).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    for (probe, count, status, failed) in [
        ("missing", 1, "partial", 0),
        ("empty", 0, "not-converged", 0),
        ("wrong-eigenvalue", 1, "partial", 1),
    ] {
        let provider = CandidateProbe(probe);
        let plan = CommonEigenPlan::resolve(&model, request(), &provider).unwrap();
        let result = plan.run_result(&provider).unwrap();
        assert_eq!(result.eigenpair_count(), count);
        assert_eq!(result.eigen_convergence(), Some(status));
        assert_eq!(result.eigen_candidate_counts().unwrap().1, failed);
        assert_eq!(
            eqiora_numerics::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan())
                .unwrap(),
            result
        );
    }
    for probe in ["duplicate", "zero"] {
        let provider = CandidateProbe(probe);
        let plan = CommonEigenPlan::resolve(&model, request(), &provider).unwrap();
        assert!(plan.run_result(&provider).is_err());
    }
    let provider = CandidateProbe("phase-permutation");
    let plan = CommonEigenPlan::resolve(&model, request(), &provider).unwrap();
    let changed = plan.run_result(&provider).unwrap();
    let reference = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver)
        .unwrap()
        .run_result(&FaerLinearSolver)
        .unwrap();
    assert_eq!(changed.eigen_convergence(), Some("converged"));
    for indices in [&[0][..], &[0, 1][..]] {
        let left = changed.eigenprojector(indices).unwrap();
        let right = reference.eigenprojector(indices).unwrap();
        for i in 0..4 {
            let (lr, li) = left.component(i).unwrap();
            let (rr, ri) = right.component(i).unwrap();
            assert!((lr - rr).hypot(li - ri) < 1e-12);
        }
    }
    assert!(changed.eigenprojector(&[0, 0]).is_err());
    assert!(changed.eigenprojector(&[2]).is_err());
}

#[test]
fn repeated_eigenspace_rotations_keep_projectors_but_not_exact_result_identity() {
    let source = QUANTUM
        .replace("math.complex(0[J],-1[J])", "0[J]")
        .replace("math.complex(0[J],1[J])", "0[J]");
    let document = ModelDocument::compile("degenerate.eqi", &source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = CommonEigenPlan::resolve(&model, request(), &CandidateProbe("unchanged")).unwrap();
    let original = plan
        .run_result(&CandidateProbe("unchanged"))
        .unwrap()
        .with_elapsed_seconds(0.)
        .unwrap();
    let rotated = plan
        .run_result(&CandidateProbe("rotate"))
        .unwrap()
        .with_elapsed_seconds(0.)
        .unwrap();
    assert_eq!(original.plan(), rotated.plan());
    assert_ne!(original.identity(), rotated.identity());
    for result in [&original, &rotated] {
        assert_eq!(result.eigen_convergence(), Some("converged"));
        for i in 0..2 {
            assert!((result.eigenpair(i).unwrap().0.component(0).unwrap().0 - 2.).abs() < 1e-12);
        }
        let projector = result.eigenprojector(&[1, 0]).unwrap();
        for (i, expected) in [1., 0., 0., 1.].into_iter().enumerate() {
            let (re, im) = projector.component(i).unwrap();
            assert!((re - expected).hypot(im) < 1e-12);
        }
        assert_eq!(
            eqiora_numerics::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan())
                .unwrap(),
            *result
        );
    }
    let document = ModelDocument::compile("quantum.eqi", QUANTUM).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let one = CommonEigenRequest::dense(NonZeroUsize::new(1).unwrap(), 1e-12, 1e-12).unwrap();
    let provider = CandidateProbe("wrong-eigenvalue");
    let result = CommonEigenPlan::resolve(&model, one, &provider)
        .unwrap()
        .run_result(&provider)
        .unwrap();
    // The first requested candidate failed. The valid second candidate cannot replace it.
    assert_eq!(result.eigenpair_count(), 0);
    assert_eq!(result.eigen_convergence(), Some("not-converged"));
}

#[test]
fn equality_orientation_preserves_the_positive_metric_pencil() {
    let document = ModelDocument::compile("canonical.eqi", STRUCTURAL).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let reference = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
    for equation in [
        "lambda*apply(m,u)=apply(k,u)",
        "-apply(k,u)=-lambda*apply(m,u)",
        "lambda*apply(m,u)-apply(k,u)=0",
    ] {
        let source = STRUCTURAL.replace("apply(k,u)=lambda*apply(m,u)", equation);
        let document = ModelDocument::compile("equivalent.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let plan = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
        // Separately authored Models retain distinct nominal space identities.
        // Compare their declared matching coordinate order and physical units.
        for (actual, expected) in [
            (plan.operator(), reference.operator()),
            (plan.metric(), reference.metric()),
        ] {
            assert_eq!(
                actual.value_type().dimension(),
                expected.value_type().dimension()
            );
            for i in 0..4 {
                assert_eq!(actual.component(i), expected.component(i));
            }
        }
        let result = plan.run_result(&FaerLinearSolver).unwrap();
        assert_eq!(result.eigen_convergence(), Some("converged"));
        for (i, expected) in [1., 3.].into_iter().enumerate() {
            assert!(
                (result.eigenpair(i).unwrap().0.component(0).unwrap().0 - expected).abs() < 1e-12
            );
        }
    }
}

#[test]
fn equality_orientation_does_not_regularize_singular_or_indefinite_metrics() {
    for metric in [
        "[[2[kg],0],[0,0[kg]]]",
        "[[2[kg],0],[0,-8[kg]]]",
        "[[-2[kg],0],[0,8[kg]]]",
    ] {
        let source = STRUCTURAL.replace("[[2[kg],0],[0,8[kg]]]", metric);
        let document = ModelDocument::compile("invalid-metric.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let error = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap_err();
        assert!(error.to_string().contains("positive definite"));
    }
}
