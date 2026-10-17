//! Spatial products retain the exact ordinary Plan's solver selection.
use super::*;
use eqiora::api::RecordedDifferentiableEvaluation;
use eqiora::solver::{LinearSolverBackend, ReductionPolicy};
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
                point.primal().evidence().primal_solve().unwrap(),
                tangent.evidence().derivative_solve().unwrap(),
                cotangent.evidence().derivative_solve().unwrap(),
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
