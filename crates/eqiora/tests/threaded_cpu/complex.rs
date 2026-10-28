//! Typed complex solves retain the selected pool through numerical lowering.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::realization::Target;
use eqiora::solver::{
    BackendId, CanonicalCsrSystemView, CsrStorage, ExecutionReport, LinearOperatorOrientation,
    LinearOperatorProperties, LinearProblem, LinearSolution, LinearSolveRequest, LinearSolver,
    LinearSolverBackend, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER,
    ReductionPolicy, ReplicatedLinearExecution, SERIAL_EXECUTION_PROVIDER, SolverCapabilities,
    SolverPlan, SolverProvider,
};
use eqiora::{Diagnostic, ScalarDomain};
use eqiora_backend_rayon::{CpuThreadPool, RAYON_EXECUTION, RAYON_EXECUTION_PROVIDER};
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonExecutionTopology, CommonLinearRequest, CommonSolvePolicy,
};
use num_complex::Complex64 as C;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Storage([C; 4]);
impl CsrStorage<C> for Storage {
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
fn plan(algorithm: LinearSolver, preconditioner: PreconditionerPolicy) -> SolverPlan {
    SolverPlan::new(algorithm, 0., 2_f64.powi(-40), 16.try_into().unwrap())
        .unwrap()
        .with_preconditioner(preconditioner)
        .with_reduction(ReductionPolicy::Reproducible)
}

fn property_fixture(properties: LinearOperatorProperties) -> ([C; 4], [C; 2], [C; 2], [C; 2]) {
    let x = [C::new(1., 2.), C::new(-2., 1.)];
    match properties {
        LinearOperatorProperties::HermitianPositiveDefinite => (
            [
                C::new(4., 0.),
                C::new(0., 1.),
                C::new(0., -1.),
                C::new(1., 0.),
            ],
            [C::new(5., 6.), C::new(3., -2.)],
            [C::new(5., 6.), C::new(3., -2.)],
            [C::new(1., 1.), C::new(2., -1.)],
        ),
        LinearOperatorProperties::Hermitian => (
            [
                C::new(4., 0.),
                C::new(0., 3.),
                C::new(0., -3.),
                C::new(1., 0.),
            ],
            [C::new(7., 10.), C::new(5., -4.)],
            [C::new(7., 10.), C::new(5., -4.)],
            [C::new(1., 1.), C::new(2., -1.)],
        ),
        LinearOperatorProperties::General => (
            [
                C::new(2., 1.),
                C::new(1., -2.),
                C::new(3., 1.),
                C::new(4., -1.),
            ],
            [C::new(0., 10.), C::new(-6., 13.)],
            [C::new(-1., 8.), C::new(-12., 6.)],
            x,
        ),
        LinearOperatorProperties::ComplexSymmetric => (
            [
                C::new(2., 1.),
                C::new(0., 0.),
                C::new(0., 0.),
                C::new(4., -1.),
            ],
            [C::new(0., 5.), C::new(-7., 6.)],
            [C::new(4., 3.), C::new(-9., 2.)],
            x,
        ),
        other => panic!("no independent complex fixture for {other:?}"),
    }
}

#[test]
fn complex_reference_policies_preserve_original_values_and_pool_placement() {
    // Pre-run inverse Frobenius bounds: HPD sqrt(19)/3<1.5,
    // indefinite sqrt(35)/5<1.2, general sqrt(37)/sqrt(65)<.76,
    // complex-symmetric sqrt(1/5+1/17)<1. Atol2^-40 therefore
    // implies component norm error <=2^-39 for every listed class.
    let z = C::new(0., 0.);
    for count in [1, 4] {
        let workers = NonZeroUsize::new(count).unwrap();
        let pool = CpuThreadPool::new(workers).unwrap();
        let wrapper = pool
            .solver(
                Target::HostCpu { threads: workers },
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        let typed = LinearSolverBackend::<f64>::complex_backend(&wrapper).unwrap();
        let capabilities = typed.capabilities();
        let advertised = capabilities
            .combinations()
            .iter()
            .filter(|entry| {
                entry.scalar_domain == ScalarDomain::Complex
                    && entry.reduction == ReductionPolicy::Reproducible
            })
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert!(!advertised.is_empty());
        let mut executed = std::collections::BTreeSet::new();
        for capability in &advertised {
            assert_eq!(capability.scalar_type, eqiora::ScalarType::F64);
            assert!(matches!(
                capability.algorithm,
                LinearSolver::ConjugateGradient
                    | LinearSolver::MinimumResidual
                    | LinearSolver::BiConjugateGradientStabilized
            ));
            assert!(matches!(
                capability.preconditioner,
                PreconditionerPolicy::Identity | PreconditionerPolicy::Jacobi
            ));
            let (matrix, rhs, adjoint_rhs, expected) =
                property_fixture(capability.operator_properties);
            let source =
                CanonicalCsrSystemView::new(&Storage(matrix), &rhs, capability.operator_properties)
                    .unwrap();
            let request = plan(capability.algorithm, capability.preconditioner);
            for (orientation, b) in [
                (LinearOperatorOrientation::Normal, rhs),
                (LinearOperatorOrientation::ConjugateTransposed, adjoint_rhs),
            ] {
                let solved = LinearSolveRequest::new(typed, request)
                    .solve_canonical_oriented(&source, &b, orientation)
                    .unwrap();
                let serial = LinearSolveRequest::<C>::new(&REFERENCE_LINEAR_SOLVER, request)
                    .solve_canonical_oriented(&source, &b, orientation)
                    .unwrap();
                assert_eq!(solved.values(), serial.values());
                let x = solved.values();
                let mut residual = [z; 2];
                for i in 0..2 {
                    let ax = (0..2).fold(z, |sum, j| {
                        sum + if orientation == LinearOperatorOrientation::Normal {
                            matrix[2 * i + j] * x[j]
                        } else {
                            matrix[2 * j + i].conj() * x[j]
                        }
                    });
                    residual[i] = b[i] - ax;
                    assert!((x[i] - expected[i]).norm() <= 2_f64.powi(-39));
                }
                assert!(
                    residual[0].norm().hypot(residual[1].norm()) <= request.absolute_tolerance()
                );
                let report = solved.report();
                assert_eq!(report.solver_provider(), REFERENCE_SOLVER_PROVIDER);
                assert_eq!(report.execution_provider(), RAYON_EXECUTION_PROVIDER);
                assert_eq!(report.verification_provider(), RAYON_EXECUTION_PROVIDER);
                assert_eq!(
                    report.execution(),
                    ExecutionReport::host(RAYON_EXECUTION, workers)
                );
                assert_eq!(report.verification(), report.execution());
                assert_eq!(report.solver_plan(), request);
                assert_eq!(report.orientation(), orientation);
                assert!(report.completed_iterations() > 0);
                assert!(report.completed_iterations() <= request.maximum_iterations().get());
                assert!(report.true_residual_norm() <= report.residual_target());
                assert_eq!(
                    report.true_residual_norm(),
                    serial.report().true_residual_norm()
                );
                assert_eq!(
                    report.completed_iterations(),
                    serial.report().completed_iterations()
                );
            }
            assert!(executed.insert(*capability));
        }
        assert_eq!(executed, advertised);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TypedMode {
    Valid,
    Missing,
    Foreign,
    RealOnly,
    FastOnly,
}
#[derive(Debug)]
struct TypedFacade {
    mode: TypedMode,
    calls: AtomicUsize,
}
fn real_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact(
        REFERENCE_LINEAR_SOLVER
            .capabilities()
            .combinations()
            .iter()
            .filter(|entry| entry.scalar_domain == ScalarDomain::Real)
            .copied(),
    )
    .unwrap()
}
impl LinearSolverBackend for TypedFacade {
    fn provider(&self) -> SolverProvider {
        REFERENCE_SOLVER_PROVIDER
    }
    fn capabilities(&self) -> SolverCapabilities {
        // Valid typed support is independent of this real-only inventory.
        // Missing support must also correct an over-advertised source inventory.
        if self.mode == TypedMode::Valid {
            real_capabilities()
        } else {
            REFERENCE_LINEAR_SOLVER.capabilities()
        }
    }
    fn complex_backend(&self) -> Option<&dyn LinearSolverBackend<C>> {
        (self.mode != TypedMode::Missing).then_some(self)
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        panic!("typed Common Run must not fall back to a real adapter")
    }
}
impl LinearSolverBackend<C> for TypedFacade {
    fn provider(&self) -> SolverProvider {
        if self.mode == TypedMode::Foreign {
            SolverProvider::new(BackendId::new("test.foreign"), "1", &[])
        } else {
            REFERENCE_SOLVER_PROVIDER
        }
    }
    fn capabilities(&self) -> SolverCapabilities {
        if self.mode == TypedMode::RealOnly {
            return real_capabilities();
        }
        let source = REFERENCE_LINEAR_SOLVER.capabilities();
        if self.mode == TypedMode::FastOnly {
            SolverCapabilities::exact(source.combinations().iter().copied().map(|mut entry| {
                entry.reduction = ReductionPolicy::Fast;
                entry
            }))
            .unwrap()
        } else {
            source
        }
    }
    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_, C>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution<C>,
    ) -> Result<LinearSolution<C>, Diagnostic> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        <_ as LinearSolverBackend<C>>::solve_with_execution(
            &REFERENCE_LINEAR_SOLVER,
            problem,
            plan,
            execution,
        )
    }
}
fn fixture(
    backend: &dyn LinearSolverBackend,
    workers: NonZeroUsize,
) -> (ModelDocument, CommonAlgebraicPlan) {
    let document=ModelDocument::compile("threaded-complex.eqi",
        "model M(){parameter p:1=3;variable z:complex<1>;relation r{math.complex(1,2)*z=math.complex(3,1)*p;}observable output:1=math.abs2(z)+p;}").unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    // Preserve the existing typed-provider fixture's exact controls.
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-15,
        8.try_into().unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible);
    let request = CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(solver, REFERENCE_SOLVER_PROVIDER)
            .unwrap()
            .with_execution(RAYON_EXECUTION_PROVIDER, workers),
    );
    let plan = CommonAlgebraicPlan::resolve(&model, request, None, &[], None, backend).unwrap();
    (document, plan)
}

#[test]
fn ordinary_complex_common_run_keeps_typed_producer_pool_and_physical_output() {
    let backend = TypedFacade {
        mode: TypedMode::Valid,
        calls: AtomicUsize::new(0),
    };
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = CpuThreadPool::new(workers).unwrap();
    let wrapper = pool
        .solver(Target::HostCpu { threads: workers }, &backend)
        .unwrap();
    let (document, plan) = fixture(&wrapper, workers);
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &wrapper)
        .unwrap();
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    // (1+2i)(p-ip)=(3+i)p. At p3, z=3-3i and |z|²+p=21.
    let values = result.finite_field_values().unwrap();
    assert_eq!(values.len(), 1);
    let (re, im) = values[0].1.component(0).unwrap();
    assert!((C::new(re, im) - C::new(3., -3.)).norm() < 2_f64.powi(-39));
    let value = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
            None,
        )
        .unwrap();
    assert!((value.value().component(0).unwrap().0 - 21.).abs() < 1e-10);
    let evidence = result.solve_evidence(None, None).unwrap();
    assert_eq!(
        evidence.solver().id(),
        REFERENCE_SOLVER_PROVIDER.id().as_str()
    );
    assert_eq!(
        evidence.execution().provider().id(),
        RAYON_EXECUTION.as_str()
    );
    assert_eq!(
        evidence.verification().provider().id(),
        RAYON_EXECUTION.as_str()
    );
    assert_eq!(
        evidence.execution().topology(),
        CommonExecutionTopology::Host { workers: 4 }
    );
    assert_eq!(evidence.reduction(), ReductionPolicy::Reproducible);
    assert_eq!(evidence.maximum_iterations(), 8);
    assert!(evidence.true_residual_norm() <= evidence.residual_target());

    let resolved = eqiora_numerics::ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
    let plan_bytes = resolved.to_bytes().unwrap();
    let reopened_plan =
        eqiora_numerics::ResolvedCommonPlan::from_bytes(&plan_bytes, &wrapper, &[]).unwrap();
    assert_eq!(reopened_plan.to_bytes().unwrap(), plan_bytes);
    let mut unavailable: serde_json::Value = serde_json::from_slice(&plan_bytes).unwrap();
    assert_eq!(
        unavailable["solve"]["linear"]["execution"]["libraries"][0][0],
        "rayon"
    );
    unavailable["solve"]["linear"]["execution"]["libraries"][0][1] = "0.0.0-unavailable".into();
    let error = eqiora_numerics::ResolvedCommonPlan::from_bytes(
        &serde_json::to_vec(&unavailable).unwrap(),
        &wrapper,
        &[],
    )
    .unwrap_err();
    assert!(
        error
            .message()
            .contains("persisted execution provider release or library inventory is unavailable")
    );
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    let bytes = result.to_bytes().unwrap();
    let reopened = eqiora_numerics::CommonResult::from_bytes(&bytes, &resolved).unwrap();
    assert_eq!(reopened.to_bytes().unwrap(), bytes);
    assert_eq!(
        reopened.finite_field_values().unwrap(),
        result.finite_field_values().unwrap()
    );
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);

    let state = plan.initial_state(&[]).unwrap();
    let one_worker = CpuThreadPool::new(NonZeroUsize::MIN).unwrap();
    let wrong = one_worker
        .solver(
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            &backend,
        )
        .unwrap();
    assert!(plan.run_result(&state, &wrong).is_err());
    assert!(plan.run_result(&state, &REFERENCE_LINEAR_SOLVER).is_err());
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);

    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged["content"]["payload"]["solve"]["solve"]["verification"] = serde_json::json!({
        "provider":{"id":SERIAL_EXECUTION_PROVIDER.id().as_str(),
            "implementation_version":SERIAL_EXECUTION_PROVIDER.implementation_version(),"libraries":[]},
        "adapter":SERIAL_EXECUTION_PROVIDER.id().as_str(),
        "topology":{"kind":"host","workers":1},
    });
    // The exact execution-contract gate precedes digest/canonical checks;
    // requiring its diagnostic makes an unrelated integrity denial insufficient.
    let error =
        eqiora_numerics::CommonResult::from_bytes(&serde_json::to_vec(&forged).unwrap(), &resolved)
            .unwrap_err();
    assert!(error.message().contains("execution evidence differs"));
}

#[test]
fn absent_foreign_stale_or_fast_only_typed_implementations_are_not_advertised_or_substituted() {
    let pool = CpuThreadPool::new(NonZeroUsize::new(4).unwrap()).unwrap();
    let reference = pool
        .solver(
            Target::HostCpu {
                threads: pool.workers(),
            },
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    let (_, plan) = fixture(&reference, pool.workers());
    for mode in [
        TypedMode::Missing,
        TypedMode::Foreign,
        TypedMode::RealOnly,
        TypedMode::FastOnly,
    ] {
        let backend = TypedFacade {
            mode,
            calls: AtomicUsize::new(0),
        };
        let wrapper = pool
            .solver(
                Target::HostCpu {
                    threads: pool.workers(),
                },
                &backend,
            )
            .unwrap();
        assert!(LinearSolverBackend::<f64>::complex_backend(&wrapper).is_none());
        assert!(
            LinearSolverBackend::<f64>::capabilities(&wrapper)
                .combinations()
                .iter()
                .all(|entry| entry.scalar_domain == ScalarDomain::Real)
        );
        assert!(
            plan.run_result(&plan.initial_state(&[]).unwrap(), &wrapper)
                .is_err()
        );
        assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    }
    assert!(
        pool.solver(
            Target::HostCpu {
                threads: pool.workers()
            },
            &eqiora_backend_faer::FaerLinearSolver
        )
        .is_err()
    );
}

#[test]
fn nonserial_payload_bounds_and_automatic_ranking_reject_before_backend_work() {
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = CpuThreadPool::new(workers).unwrap();
    let backend = TypedFacade {
        mode: TypedMode::Valid,
        calls: AtomicUsize::new(0),
    };
    let wrapper = pool
        .solver(Target::HostCpu { threads: workers }, &backend)
        .unwrap();
    // Reference real BiCGSTAB has an established serial workspace bound;
    // complex memory is already unproved and would not distinguish this gate.
    let document = ModelDocument::compile(
        "threaded-resource.eqi",
        "model M(){variable z:1;relation r{2*z=6;}}",
    )
    .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let serial = CommonLinearRequest::exact(
        plan(
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
        ),
        REFERENCE_SOLVER_PROVIDER,
    )
    .unwrap()
    .with_cost(
        eqiora::solver::SolverCostRequest::default()
            .with_numeric_payload_limit(NonZeroUsize::new(usize::MAX).unwrap()),
    );
    CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(serial),
        None,
        &[],
        None,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let exact = serial.with_execution(RAYON_EXECUTION_PROVIDER, workers);
    let automatic = CommonLinearRequest::program_controlled(
        1e-13,
        1e-15,
        8.try_into().unwrap(),
        eqiora::solver::SolverPlanningObjective::Robust,
    )
    .unwrap()
    .with_execution(RAYON_EXECUTION_PROVIDER, workers);
    for (request, reason) in [
        (exact, "resource.numeric-payload-bound-unproved"),
        (
            automatic,
            "automatic solver ranking is implemented only for host-serial execution",
        ),
    ] {
        let error = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(request),
            None,
            &[],
            None,
            &wrapper,
        )
        .unwrap_err();
        assert!(error.message().contains(reason), "{error:?}");
        assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    }
}
