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

    plan(
        match value["algorithm"].as_str().unwrap() {
            "BiConjugateGradientStabilized" => LinearSolver::BiConjugateGradientStabilized,
            "ConjugateGradient" => LinearSolver::ConjugateGradient,
            "SparseLu" => LinearSolver::SparseLu,
            other => panic!("unsupported complex tuple {other}"),
        },
        match value["preconditioner"].as_str().unwrap() {
            "Identity" => PreconditionerPolicy::Identity,
            "Jacobi" => PreconditionerPolicy::Jacobi,
            other => panic!("unlisted preconditioner {other}"),
        },
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
        for (section, supplied) in [("objectives", true), ("without_faer_objectives", false)] {
            for expected in fixture[section].as_array().unwrap() {
                let decision = ResolvedHostSerialSolverPlan::<C>::resolve(
                    HostSerialSolverProfile::canonical_csr(
                        properties,
                        expected["complete_diagonal"].as_bool(),
                        None,
                    ),
                    objective(expected["objective"].as_str().unwrap()),
                    1e-12,
                    1e-14,
                    NonZeroUsize::new(100).unwrap(),
                    &REFERENCE_LINEAR_SOLVER,
                    if supplied { Some(&faer) } else { None },
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

#[test]
fn reference_complex_choices_preserve_exact_normal_and_adjoint_equations() {
    let backend: &dyn LinearSolverBackend<C> = &REFERENCE_LINEAR_SOLVER;
    let oracle: Value = serde_json::from_str(ORACLE).unwrap();
    for fixture in oracle["complex_profiles"].as_array().unwrap() {
        let class = properties(fixture["properties"].as_str().unwrap());
        let matrix = &fixture["matrix"];
        let storage = ComplexFixture([
            complex(&matrix[0][0]),
            complex(&matrix[0][1]),
            complex(&matrix[1][0]),
            complex(&matrix[1][1]),
        ]);
        let rhs = values(&fixture["right_hand_side"]);
        let adjoint_rhs = values(&fixture["adjoint_right_hand_side"]);
        let system = CanonicalCsrSystemView::new(&storage, &rhs, class).unwrap();
        for algorithm in [
            LinearSolver::BiConjugateGradientStabilized,
            LinearSolver::ConjugateGradient,
        ] {
            if algorithm == LinearSolver::ConjugateGradient
                && class != LinearOperatorProperties::HermitianPositiveDefinite
            {
                continue;
            }
            for preconditioner in [PreconditionerPolicy::Identity, PreconditionerPolicy::Jacobi] {
                let selected = plan(algorithm, preconditioner, ReductionPolicy::Reproducible);
                let request = LinearSolveRequest::new(backend, selected);
                for (orientation, right_hand_side, expected) in [
                    (
                        LinearOperatorOrientation::Normal,
                        &rhs,
                        &fixture["solution"],
                    ),
                    (
                        LinearOperatorOrientation::ConjugateTransposed,
                        &adjoint_rhs,
                        &fixture["adjoint_solution"],
                    ),
                ] {
                    let solution = request.solve_canonical_oriented(&system, right_hand_side, orientation)
                        .unwrap_or_else(|error| panic!("{class:?}/{algorithm:?}/{preconditioner:?}/{orientation:?}: {error}"));
                    assert_values(solution.values(), expected);
                    assert!(
                        solution.report().true_residual_norm()
                            <= solution.report().residual_target()
                    );
                    assert_eq!(solution.report().solver_plan(), selected);
                    assert_eq!(solution.report().solver_provider(), backend.provider());
                    assert_eq!(solution.report().orientation(), orientation);
                }
            }
        }
    }
}

#[test]
fn complex_jacobi_distinguishes_stored_zero_from_unproved_diagonal() {
    // det(A)=-(1+i)(2-i)=-3-i !=0; A*[1+2i,-2+i]=[-3-i,-2+6i].
    // Both diagonal slots are stored, but the first numerical entry is zero.
    let storage = ComplexFixture([
        C::new(0., 0.),
        C::new(1., 1.),
        C::new(2., -1.),
        C::new(3., 0.),
    ]);
    let rhs = [C::new(-3., -1.), C::new(-2., 6.)];
    let system =
        CanonicalCsrSystemView::new(&storage, &rhs, LinearOperatorProperties::General).unwrap();
    let problem = system.linear_problem().unwrap();
    let identity = plan(
        LinearSolver::BiConjugateGradientStabilized,
        PreconditionerPolicy::Identity,
        ReductionPolicy::Reproducible,
    );
    let jacobi = identity.with_preconditioner(PreconditionerPolicy::Jacobi);
    let known =
        HostSerialSolverProfile::canonical_csr(LinearOperatorProperties::General, Some(true), None);
    known.require_problem(&problem).unwrap();
    known.require_plan(jacobi).unwrap();
    let unknown =
        HostSerialSolverProfile::canonical_csr(LinearOperatorProperties::General, None, None);
    assert!(
        unknown
            .require_plan(jacobi)
            .unwrap_err()
            .message()
            .contains("profile.complete-diagonal-required")
    );
    let solution = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, identity)
        .solve(&problem)
        .unwrap();
    for (actual, expected) in solution
        .values()
        .iter()
        .zip([C::new(1., 2.), C::new(-2., 1.)])
    {
        assert!((*actual - expected).norm() <= 2f64.powi(-40));
    }
    assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
    let error = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, jacobi)
        .solve(&problem)
        .unwrap_err();
    assert!(
        error.message().contains("finite nonzero diagonal"),
        "{error:?}"
    );
}

#[test]
fn absent_provider_fast_preference_retains_real_spd_specialization() {
    struct Spd;
    impl CsrStorage for Spd {
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
        fn values(&self) -> &[f64] {
            &[4., 1., 1., 3.]
        }
    }
    // Leading minors4,11 prove SPD; A*[1,2]=[6,7]. No shape/cost estimate
    // is supplied, so the declared specialized-method preference applies.
    let system = CanonicalCsrSystemView::new(
        &Spd,
        &[6., 7.],
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let problem = system.linear_problem().unwrap();
    for diagonal in [None, Some(true)] {
        for objective in [
            SolverPlanningObjective::Robust,
            SolverPlanningObjective::Fast,
            SolverPlanningObjective::LowMemory,
        ] {
            let decision = ResolvedHostSerialSolverPlan::resolve(
                HostSerialSolverProfile::canonical_csr(
                    LinearOperatorProperties::SymmetricPositiveDefinite,
                    diagonal,
                    None,
                ),
                objective,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap();
            assert_eq!(
                decision.selected_candidate_id(),
                "eqiora.reference.cg-spd-identity-reproducible-f64"
            );
            assert_eq!(
                decision.solver_plan().algorithm(),
                LinearSolver::ConjugateGradient
            );
            let solution = decision.solve(&problem, None).unwrap();
            for (actual, expected) in solution.values().iter().zip([1., 2.]) {
                assert!((actual - expected).abs() <= 2f64.powi(-40));
            }
            let x = solution.values();
            let residual = (6. - (4. * x[0] + x[1])).hypot(7. - (x[0] + 3. * x[1]));
            assert!(residual <= solution.report().residual_target());
            assert_eq!(
                solution.report().solver_provider(),
                REFERENCE_LINEAR_SOLVER.provider()
            );
        }
    }
}
