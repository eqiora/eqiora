//! Exact complex catalogs use genuine operator classes and independent rational fixtures.
use super::*;
use eqiora::solver::LinearOperatorOrientation;
use num_complex::Complex64 as C;

struct ComplexFixture([C; 4]);
impl CsrStorage<C> for ComplexFixture {
    fn rows(&self) -> usize {
        2
    }
    fn columns(&self) -> usize {
        2
    }
    fn row_offsets(&self) -> &[usize] {
        &[0, 2, 4]
    }
    fn column_indices(&self) -> &[usize] {
        &[0, 1, 0, 1]
    }
    fn values(&self) -> &[C] {
        &self.0
    }
}
fn properties(name: &str) -> LinearOperatorProperties {
    match name {
        "General" => LinearOperatorProperties::General,
        "ComplexSymmetric" => LinearOperatorProperties::ComplexSymmetric,
        "Hermitian" => LinearOperatorProperties::Hermitian,
        "HermitianPositiveDefinite" => LinearOperatorProperties::HermitianPositiveDefinite,
        other => panic!("unlisted complex mathematical class {other}"),
    }
}
fn scalar(value: &Value) -> f64 {
    value.as_f64().unwrap_or_else(|| {
        value["numerator"].as_f64().unwrap() / value["denominator"].as_f64().unwrap()
    })
}
fn complex(value: &Value) -> C {
    C::new(scalar(&value[0]), scalar(&value[1]))
}
fn values(value: &Value) -> Vec<C> {
    value.as_array().unwrap().iter().map(complex).collect()
}
fn tuple(value: &Value) -> SolverPlan {
    assert_eq!(value["scalar_domain"], "Complex");
    assert_eq!(value["scalar_type"], "F64");
    assert_eq!(value["preconditioner"], "Identity");
    plan(
        match value["algorithm"].as_str().unwrap() {
            "BiConjugateGradientStabilized" => LinearSolver::BiConjugateGradientStabilized,
            "ConjugateGradient" => LinearSolver::ConjugateGradient,
            "SparseLu" => LinearSolver::SparseLu,
            other => panic!("unsupported complex tuple {other}"),
        },
        PreconditionerPolicy::Identity,
        match value["reduction"].as_str().unwrap() {
            "Reproducible" => ReductionPolicy::Reproducible,
            "Fast" => ReductionPolicy::Fast,
            other => panic!("unlisted reduction {other}"),
        },
    )
}
fn assert_values(actual: &[C], expected: &Value) {
    let expected = values(expected);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.re - expected.re).abs() <= 2_f64.powi(-40));
        assert!((actual.im - expected.im).abs() <= 2_f64.powi(-40));
    }
}

#[test]
fn complex_catalogs_execute_normal_and_adjoint_with_exact_provider_and_objective() {
    run_derivation();
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    let faer = FaerLinearSolver;
    for fixture in oracle["complex_profiles"].as_array().unwrap() {
        let properties = properties(fixture["properties"].as_str().unwrap());
        let matrix = &fixture["matrix"];
        let storage = ComplexFixture([
            complex(&matrix[0][0]),
            complex(&matrix[0][1]),
            complex(&matrix[1][0]),
            complex(&matrix[1][1]),
        ]);
        let rhs = values(&fixture["right_hand_side"]);
        let system = CanonicalCsrSystemView::new(&storage, &rhs, properties).unwrap();
        let problem = system.linear_problem().unwrap();
        let adjoint_rhs = values(&fixture["adjoint_right_hand_side"]);
        // Execute every catalog tuple directly, independently of ranking. This
        // is the numerical authority for the catalog's complex evidence links.
        for candidate in fixture["candidates"].as_array().unwrap() {
            let backend: &dyn LinearSolverBackend<C> =
                if candidate["solver_provider"]["id"] == "eqiora.reference" {
                    &REFERENCE_LINEAR_SOLVER
                } else {
                    &faer
                };
            let manual = tuple(&candidate["tuple"]);
            assert_solver_provider(backend.provider(), &candidate["solver_provider"]);
            backend
                .capabilities()
                .require_problem(
                    manual,
                    eqiora::ScalarDomain::Complex,
                    eqiora::ScalarType::F64,
                    properties,
                )
                .unwrap();
            let request = LinearSolveRequest::new(backend, manual);
            let solution = request.solve(&problem).unwrap();
            assert_values(solution.values(), &fixture["solution"]);
            assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
            let adjoint = request
                .solve_canonical_oriented(
                    &system,
                    &adjoint_rhs,
                    LinearOperatorOrientation::ConjugateTransposed,
                )
                .unwrap();
            assert_values(adjoint.values(), &fixture["adjoint_solution"]);
            assert!(adjoint.report().true_residual_norm() <= adjoint.report().residual_target());
        }
        for expected in fixture["objectives"].as_array().unwrap() {
            let decision = ResolvedHostSerialSolverPlan::<C>::resolve(
                HostSerialSolverProfile::canonical_csr(properties, None, None),
                objective(expected["objective"].as_str().unwrap()),
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
                &REFERENCE_LINEAR_SOLVER,
                Some(&faer),
            )
            .unwrap();
            assert_eq!(
                decision.selected_candidate_id(),
                expected["selected_candidate_id"].as_str().unwrap()
            );
            let candidate = fixture["candidates"]
                .as_array()
                .unwrap()
                .iter()
                .find(|candidate| {
                    candidate["id"].as_str() == Some(decision.selected_candidate_id())
                })
                .unwrap();
            assert_eq!(decision.solver_plan(), tuple(&candidate["tuple"]));
            assert_solver_provider(decision.solver_provider(), &candidate["solver_provider"]);
            assert_eq!(
                decision.selected_evidence_case(),
                candidate["evidence_case"].as_str().unwrap()
            );
            let trace = expected["ordered_reasons"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| (row[0].as_str().unwrap(), row[1].as_str().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(decision.reasons().collect::<Vec<_>>(), trace);
            let solved = decision.solve(&problem, None).unwrap();
            assert_values(solved.values(), &fixture["solution"]);
            let backend: &dyn LinearSolverBackend<C> =
                if decision.solver_provider() == REFERENCE_LINEAR_SOLVER.provider() {
                    &REFERENCE_LINEAR_SOLVER
                } else {
                    &faer
                };
            let manual = LinearSolveRequest::new(backend, decision.solver_plan())
                .solve(&problem)
                .unwrap();
            assert_eq!(solved, manual);
        }
    }
}

#[test]
fn absent_optional_complex_provider_preserves_reference_and_required_reduction() {
    let faer = FaerLinearSolver;
    for (properties, reference_id, direct_id) in [
        (
            LinearOperatorProperties::General,
            "eqiora.reference.bicgstab-general-identity-reproducible-complex-f64",
            "eqiora.faer.sparse-lu-general-identity-fast-complex-f64",
        ),
        (
            LinearOperatorProperties::HermitianPositiveDefinite,
            "eqiora.reference.cg-hpd-identity-reproducible-complex-f64",
            "eqiora.faer.sparse-lu-hpd-identity-fast-complex-f64",
        ),
    ] {
        for objective in [
            SolverPlanningObjective::Robust,
            SolverPlanningObjective::Fast,
            SolverPlanningObjective::LowMemory,
        ] {
            let selected = ResolvedHostSerialSolverPlan::<C>::resolve(
                HostSerialSolverProfile::canonical_csr(properties, None, None),
                objective,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap();
            assert_eq!(selected.selected_candidate_id(), reference_id);
            assert!(
                selected
                    .reasons()
                    .any(|reason| reason == (direct_id, "catalog.provider-unavailable"))
            );
        }
        let error = ResolvedHostSerialSolverPlan::<C>::resolve(
            HostSerialSolverProfile::canonical_csr(properties, None, Some(ReductionPolicy::Fast)),
            SolverPlanningObjective::Fast,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap_err();
        assert!(
            error
                .message()
                .contains("profile.required-reduction-mismatch")
        );
        assert!(error.message().contains("catalog.provider-unavailable"));
        for objective in [
            SolverPlanningObjective::Robust,
            SolverPlanningObjective::Fast,
            SolverPlanningObjective::LowMemory,
        ] {
            let selected = ResolvedHostSerialSolverPlan::<C>::resolve(
                HostSerialSolverProfile::canonical_csr(
                    properties,
                    None,
                    Some(ReductionPolicy::Fast),
                ),
                objective,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
                &REFERENCE_LINEAR_SOLVER,
                Some(&faer),
            )
            .unwrap();
            assert_eq!(selected.selected_candidate_id(), direct_id);
            assert!(
                selected
                    .reasons()
                    .any(|reason| reason == (reference_id, "profile.required-reduction-mismatch"))
            );
        }
    }
}
