//! The ordinary finite lifecycle executes the admitted typed provider instance.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::solver::{
    LinearProblem, LinearSolution, LinearSolver, LinearSolverBackend, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, ReplicatedLinearExecution, SolverCapabilities, SolverPlan, SolverProvider,
};
use eqiora::{Diagnostic, ScalarDomain};
use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
use num_complex::Complex64;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

fn real_only_capabilities() -> SolverCapabilities {
    SolverCapabilities::exact(
        REFERENCE_LINEAR_SOLVER
            .capabilities()
            .combinations()
            .iter()
            .copied()
            .filter(|capability| capability.scalar_domain == ScalarDomain::Real),
    )
    .unwrap()
}

#[derive(Debug, Default)]
struct ObservedReference {
    calls: AtomicUsize,
    missing: bool,
    stale: bool,
    foreign: bool,
    custom: bool,
}

impl ObservedReference {
    fn family_provider(&self) -> SolverProvider {
        if self.custom {
            SolverProvider::new(
                eqiora::solver::BackendId::new("eqiora.test.typed"),
                "1",
                &[],
            )
        } else {
            REFERENCE_LINEAR_SOLVER.provider()
        }
    }
}

impl LinearSolverBackend for ObservedReference {
    fn provider(&self) -> SolverProvider {
        self.family_provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        // Real capabilities are deliberately different. Complex admission must
        // authenticate the typed implementation's capabilities, not these.
        real_only_capabilities()
    }
    fn complex_backend(&self) -> Option<&dyn LinearSolverBackend<Complex64>> {
        (!self.missing).then_some(self)
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        panic!("complex execution reached the real adapter")
    }
}

impl LinearSolverBackend<Complex64> for ObservedReference {
    fn provider(&self) -> SolverProvider {
        if self.foreign {
            SolverProvider::new(
                eqiora::solver::BackendId::new("eqiora.test.foreign"),
                "1",
                &[],
            )
        } else {
            self.family_provider()
        }
    }
    fn capabilities(&self) -> SolverCapabilities {
        if self.stale {
            real_only_capabilities()
        } else {
            REFERENCE_LINEAR_SOLVER.capabilities()
        }
    }
    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_, Complex64>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution<Complex64>, Diagnostic> {
        assert_eq!(problem.scalar_domain(), ScalarDomain::Complex);
        self.calls.fetch_add(1, Ordering::SeqCst);
        <_ as LinearSolverBackend<Complex64>>::solve_with_execution(
            &REFERENCE_LINEAR_SOLVER,
            problem,
            plan,
            execution,
        )
    }
}

fn fixture() -> (ModelDocument, CommonAlgebraicPlan) {
    let document = ModelDocument::compile(
        "typed-provider.eqi",
        "model M(){parameter p:1=3;variable z:complex<1>;relation r{math.complex(1,2)*z=math.complex(3,1)*p;}observable output:1=math.abs2(z)+p;}",
    )
    .unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible);
    let request = CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap(),
    );
    let plan =
        CommonAlgebraicPlan::resolve(&model, request, None, &[], None, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
    (document, plan)
}

#[test]
fn complex_run_borrows_the_supplied_typed_implementation() {
    let (document, plan) = fixture();
    let backend = ObservedReference::default();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &backend)
        .unwrap();
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    // (1+2i)(p-ip)=(3+i)p, hence |z|²+p=2p²+p=21 at p=3.
    let value = result
        .observe(
            plan.model_artifact(),
            document.aliases()["output"].downcast().unwrap(),
            &Default::default(),
        )
        .unwrap();
    assert!((value.value().component(0).unwrap().0 - 21.).abs() < 1e-10);
}

#[test]
fn complex_run_rejects_missing_foreign_or_changed_typed_backend_before_solve() {
    let (_, plan) = fixture();
    let state = plan.initial_state(&[]).unwrap();
    for backend in [
        ObservedReference {
            missing: true,
            ..Default::default()
        },
        ObservedReference {
            stale: true,
            ..Default::default()
        },
        ObservedReference {
            foreign: true,
            ..Default::default()
        },
    ] {
        assert!(plan.run_result(&state, &backend).is_err());
        assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn differentiated_complex_primal_uses_the_same_typed_backend() {
    use eqiora::api::DifferentiableProgram;
    use eqiora_numerics::ResolvedCommonPlan;
    let (document, plan) = fixture();
    static BACKEND: ObservedReference = ObservedReference {
        calls: AtomicUsize::new(0),
        missing: false,
        stale: false,
        foreign: false,
        custom: false,
    };
    let backend = &BACKEND;
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[document.parameter_ref("p").unwrap()],
        &document.observable_ref("output").unwrap(),
        None,
        backend,
    )
    .unwrap();
    let before = backend.calls.load(Ordering::SeqCst);
    let point = program.evaluate(&[5.]).unwrap();
    assert_eq!(backend.calls.load(Ordering::SeqCst), before + 1);
    assert!((point.primal().output()[0] - 55.).abs() < 1e-9);
}

#[test]
fn complex_plan_admits_only_the_selected_providers_matching_typed_capabilities() {
    let (document, _) = fixture();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible);
    for (backend, accepted) in [
        (
            ObservedReference {
                custom: true,
                ..Default::default()
            },
            true,
        ),
        (
            ObservedReference {
                custom: true,
                missing: true,
                ..Default::default()
            },
            false,
        ),
        (
            ObservedReference {
                custom: true,
                stale: true,
                ..Default::default()
            },
            false,
        ),
        (
            ObservedReference {
                custom: true,
                foreign: true,
                ..Default::default()
            },
            false,
        ),
    ] {
        let request = CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(solver, backend.family_provider()).unwrap(),
        );
        let plan = CommonAlgebraicPlan::resolve(&model, request, None, &[], None, &backend);
        assert_eq!(plan.is_ok(), accepted);
        if let Ok(plan) = plan {
            assert_eq!(plan.solver_provider(), backend.family_provider());
        }
        assert_eq!(backend.calls.load(Ordering::SeqCst), 0);
    }
}
