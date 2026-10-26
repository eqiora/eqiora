//! Spatial products retain the exact ordinary Plan's solver selection.
use super::*;
use eqiora::api::RecordedDifferentiableEvaluation;
use eqiora::solver::ReductionPolicy;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{CommonLinearRequest, CommonSolvePolicy, ResolvedCommonPlan};

#[test]
fn sparse_lu_spatial_products_keep_the_provider_and_exact_boundary_response() {
    let source = COMPONENT.replace(
        "source_scale * math.sin(wave_number * coordinate(0))\n        * math.sin(wave_number * coordinate(1))",
        "0 / 1[m^2]",
    );
    assert_ne!(source, COMPONENT);
    for method in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let (document, _) = document_and_plan_with_source(method, &source);
        let solver = SolverPlan::new(
            LinearSolver::SparseLu,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        let plan = ResolvedCommonPlan::Spatial(Box::new(
            support::common_scalar_plan::plan_for_document_with_solver(
                &document,
                [4, 3],
                method,
                CommonSolvePolicy::Linear(
                    CommonLinearRequest::exact(solver, FaerLinearSolver.provider()).unwrap(),
                ),
                &FaerLinearSolver,
            ),
        ));
        let inputs = [document.parameter_ref("boundary_offset").unwrap()];
        let output = document.field_ref("definition.potential").unwrap().erase();
        assert!(
            DifferentiableProgram::compile(
                plan.clone(),
                &inputs,
                std::slice::from_ref(&output),
                Default::default(),
                None,
                &REFERENCE_LINEAR_SOLVER,
            )
            .is_err()
        );
        let program = DifferentiableProgram::compile(
            plan.clone(),
            &inputs,
            &[output],
            Default::default(),
            None,
            &FaerLinearSolver,
        )
        .unwrap();
        for offset in [3., -2.] {
            let point = program
                .evaluate(&support::real_point(&program, &[offset]).unwrap())
                .unwrap();
            // With zero forcing and a uniform boundary, u=offset exactly on every
            // represented coordinate, independently of the discrete solve.
            for value in point.primal().output().fixture_coordinates() {
                assert!((value - offset).abs() < 2e-12);
            }
            let tangent = point
                .jvp(&support::derivative_products::tangent(program.identity(), &[2.]).unwrap())
                .unwrap();
            for value in tangent.derivative().fixture_coordinates() {
                assert!((value - 2.).abs() < 2e-12);
            }
            let count = program.identity().output_dimension();
            let weights = (1..=count).map(|i| i as f64).collect::<Vec<_>>();
            let cotangent = point
                .vjp(
                    &support::derivative_products::cotangent(program.identity(), &weights).unwrap(),
                )
                .unwrap();
            let expected = (count * (count + 1) / 2) as f64;
            assert!((cotangent.derivative().fixture_coordinates()[0] - expected).abs() < 2e-10);
            for evidence in [
                point.primal().evidence().steps()[0]
                    .primal_solve(0)
                    .unwrap(),
                tangent.evidence().steps()[0].derivative_solve().unwrap(),
                cotangent.evidence().steps()[0].derivative_solve().unwrap(),
            ] {
                assert_eq!(evidence.solver().id(), "eqiora.faer");
                assert_eq!(evidence.algorithm(), LinearSolver::SparseLu);
                assert!(evidence.true_residual_norm() <= evidence.residual_target());
            }
            let record = point.record().unwrap();
            assert_eq!(
                RecordedDifferentiableEvaluation::from_bytes(
                    &record.to_bytes().unwrap(),
                    &plan,
                    None,
                )
                .unwrap(),
                record
            );
        }
    }
}

#[test]
fn tpfa_cg_jacobi_retains_forward_and_adjoint_diagonals_and_provenance() {
    use eqiora::solver::PreconditionerPolicy;
    let source = COMPONENT.replace(
        "source_scale * math.sin(wave_number * coordinate(0))\n        * math.sin(wave_number * coordinate(1))",
        "0 / 1[m^2]",
    );
    assert_ne!(source, COMPONENT);
    let method = CommonSpatialPolicy::CellCenteredTpfa;
    let (document, _) = document_and_plan_with_source(method, &source);
    let solver = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1e-12,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi);
    let plan = ResolvedCommonPlan::Spatial(Box::new(
        support::common_scalar_plan::plan_for_document_with_solver(
            &document,
            [4, 3],
            method,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap(),
            ),
            &REFERENCE_LINEAR_SOLVER,
        ),
    ));
    assert_eq!(
        plan.operator_properties(),
        Some(LinearOperatorProperties::SymmetricPositiveDefinite)
    );
    let inputs = [document.parameter_ref("boundary_offset").unwrap()];
    let output = document.field_ref("definition.potential").unwrap().erase();
    let program = DifferentiableProgram::compile(
        plan.clone(),
        &inputs,
        &[output],
        Default::default(),
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let point = program
        .evaluate(&support::real_point(&program, &[3.]).unwrap())
        .unwrap();
    let forward = point
        .jvp(&support::derivative_products::tangent(program.identity(), &[2.]).unwrap())
        .unwrap();
    let count = program.identity().output_dimension();
    let weights = (1..=count).map(|index| index as f64).collect::<Vec<_>>();
    let reverse = point
        .vjp(&support::derivative_products::cotangent(program.identity(), &weights).unwrap())
        .unwrap();
    // Zero forcing and a uniform boundary give u=offset on every cell. Thus
    // du/d(offset)=1 independently of the discrete matrix or implementation.
    for value in point.primal().output().fixture_coordinates() {
        assert!((value - 3.).abs() < 2e-10);
    }
    for value in forward.derivative().fixture_coordinates() {
        assert!((value - 2.).abs() < 2e-10);
    }
    assert!(
        (reverse.derivative().fixture_coordinates()[0] - (count * (count + 1) / 2) as f64).abs()
            < 2e-10
    );
    for (evidence, orientation) in [
        point.primal().evidence().steps()[0]
            .primal_solve(0)
            .unwrap(),
        forward.evidence().steps()[0].derivative_solve().unwrap(),
        reverse.evidence().steps()[0].derivative_solve().unwrap(),
    ]
    .into_iter()
    .zip([
        LinearOperatorOrientation::Normal,
        LinearOperatorOrientation::Normal,
        LinearOperatorOrientation::ConjugateTransposed,
    ]) {
        assert_eq!(
            evidence.solver().id(),
            REFERENCE_LINEAR_SOLVER.provider().id().as_str()
        );
        assert_eq!(evidence.algorithm(), LinearSolver::ConjugateGradient);
        assert_eq!(evidence.preconditioner(), PreconditionerPolicy::Jacobi);
        assert_eq!(evidence.orientation(), orientation);
        assert!(evidence.true_residual_norm() <= evidence.residual_target());
    }
    let record = point.record().unwrap();
    assert_eq!(
        RecordedDifferentiableEvaluation::from_bytes(&record.to_bytes().unwrap(), &plan, None)
            .unwrap(),
        record
    );
}
