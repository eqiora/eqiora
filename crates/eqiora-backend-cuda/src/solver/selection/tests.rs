use super::*;
use eqiora_device::{DeviceDescriptor, DeviceId, QueueSlot};
use eqiora_execution::CUDA_LINEAR_DEVICE_CAPABILITIES;
use eqiora_solver::{
    LinearOperatorProperties as Properties, LinearSolver, PreconditionerPolicy,
    REFERENCE_SOLVER_PROVIDER, ReductionPolicy, SERIAL_EXECUTION_PROVIDER, SolverCostRequest,
};
use std::num::NonZeroU64;

fn device(ordinal: u16) -> DeviceDescriptor {
    DeviceDescriptor::new(
        DeviceId::new(crate::CUDA_RUNTIME_ID, ordinal),
        "static CUDA snapshot",
        NonZeroU64::new(1 << 30).unwrap(),
        CUDA_LINEAR_DEVICE_CAPABILITIES,
    )
    .unwrap()
}
fn executor(ordinal: u16, queue: u32) -> CudaExecutorDescriptor {
    let device = device(ordinal);
    CudaExecutorDescriptor::new(
        CUDA_LINEAR_SOLVER_PROVIDER,
        CUDA_LINEAR_EXECUTION_PROVIDER,
        device.clone(),
        QueueSlot::new(device.id(), queue),
        CudaLinearSolver::capabilities(),
    )
    .unwrap()
}
fn profile(
    properties: Properties,
    diagonal: Option<bool>,
    reduction: ReductionPolicy,
) -> SolverPlanningProfile {
    SolverPlanningProfile::canonical_csr(properties, diagonal, Some(reduction))
        .with_execution(
            CUDA_LINEAR_EXECUTION_PROVIDER,
            ExecutionReport::cuda(CUDA_LINEAR_EXECUTION_PROVIDER.id(), 0),
        )
        .unwrap()
}
fn select(
    executor: CudaExecutorDescriptor,
    profile: SolverPlanningProfile,
    objective: SolverPlanningObjective,
) -> Result<CudaLinearSolver, Diagnostic> {
    CudaLinearSolver::new(0).select_plan(
        executor,
        profile,
        objective,
        1e-12,
        1e-14,
        NonZeroUsize::new(4096).unwrap(),
    )
}

#[test]
fn explicit_cuda_selection_covers_only_the_three_delivered_real_fast_tuples() {
    for (properties, diagonal, algorithm, preconditioner) in [
        (
            Properties::SymmetricPositiveDefinite,
            Some(true),
            LinearSolver::ConjugateGradient,
            PreconditionerPolicy::Jacobi,
        ),
        (
            Properties::General,
            None,
            LinearSolver::BiConjugateGradientStabilized,
            PreconditionerPolicy::Identity,
        ),
        (
            Properties::SymmetricIndefinite,
            None,
            LinearSolver::MinimumResidual,
            PreconditionerPolicy::Identity,
        ),
    ] {
        for objective in [
            SolverPlanningObjective::Robust,
            SolverPlanningObjective::Fast,
            SolverPlanningObjective::LowMemory,
        ] {
            let selected = select(
                executor(0, 0),
                profile(properties, diagonal, ReductionPolicy::Fast),
                objective,
            )
            .unwrap();
            let selection = selected.selection().unwrap();
            let plan = selection.solver_plan();
            assert_eq!(plan.algorithm(), algorithm);
            assert_eq!(plan.preconditioner(), preconditioner);
            assert_eq!(plan.reduction(), ReductionPolicy::Fast);
            assert_eq!(plan.relative_tolerance(), 1e-12);
            assert_eq!(plan.absolute_tolerance(), 1e-14);
            assert_eq!(plan.maximum_iterations().get(), 4096);
            assert_eq!(selection.solver_provider(), CUDA_LINEAR_SOLVER_PROVIDER);
            assert_eq!(
                selection.execution_provider(),
                CUDA_LINEAR_EXECUTION_PROVIDER
            );
            assert_eq!(
                selection.execution(),
                ExecutionReport::cuda(CUDA_LINEAR_EXECUTION_PROVIDER.id(), 0)
            );
            assert!(!selection.costs().is_empty());
            for cost in selection.costs() {
                assert_eq!(cost.setup_work(), None);
                assert_eq!(cost.solve_work(), None);
                assert_eq!(cost.acceptance_work(), None);
                assert_eq!(cost.total_work(), None);
                assert_eq!(cost.numeric_payload_bytes(), None);
            }
        }
    }
}

#[test]
fn cuda_selection_rejects_unproved_math_resources_and_deployment_substitution_without_runtime() {
    let normal = || {
        profile(
            Properties::SymmetricPositiveDefinite,
            Some(true),
            ReductionPolicy::Fast,
        )
    };
    for rejected in [
        profile(
            Properties::SymmetricPositiveDefinite,
            None,
            ReductionPolicy::Fast,
        ),
        profile(
            Properties::SymmetricPositiveDefinite,
            Some(true),
            ReductionPolicy::Reproducible,
        ),
        normal().with_cost_request(
            SolverCostRequest::default()
                .with_numeric_payload_limit(NonZeroUsize::new(usize::MAX).unwrap()),
        ),
        SolverPlanningProfile::canonical_csr(
            Properties::SymmetricPositiveDefinite,
            Some(true),
            Some(ReductionPolicy::Fast),
        ),
        normal()
            .with_execution(
                CUDA_LINEAR_EXECUTION_PROVIDER,
                ExecutionReport::cuda(CUDA_LINEAR_EXECUTION_PROVIDER.id(), 1),
            )
            .unwrap(),
    ] {
        assert!(select(executor(0, 0), rejected, SolverPlanningObjective::Fast).is_err());
    }
    for rejected in [executor(0, 1), executor(1, 0)] {
        assert!(select(rejected, normal(), SolverPlanningObjective::Fast).is_err());
    }
    let snapshot = device(0);
    let mut invented = *CudaLinearSolver::capabilities()
        .combinations()
        .iter()
        .find(|entry| entry.algorithm == LinearSolver::ConjugateGradient)
        .unwrap();
    invented.preconditioner = PreconditionerPolicy::Identity;
    let invented = CudaExecutorDescriptor::new(
        CUDA_LINEAR_SOLVER_PROVIDER,
        CUDA_LINEAR_EXECUTION_PROVIDER,
        snapshot.clone(),
        QueueSlot::new(snapshot.id(), 0),
        eqiora_solver::SolverCapabilities::exact([invented]).unwrap(),
    )
    .unwrap();
    let error = select(invented, normal(), SolverPlanningObjective::Fast).unwrap_err();
    assert!(
        error
            .message()
            .contains("advertises an unimplemented solver tuple")
    );

    for (solver, execution) in [
        (REFERENCE_SOLVER_PROVIDER, CUDA_LINEAR_EXECUTION_PROVIDER),
        (
            eqiora_solver::SolverProvider::new(
                CUDA_LINEAR_SOLVER_PROVIDER.id(),
                "stale",
                CUDA_LINEAR_SOLVER_PROVIDER.libraries(),
            ),
            CUDA_LINEAR_EXECUTION_PROVIDER,
        ),
        (CUDA_LINEAR_SOLVER_PROVIDER, SERIAL_EXECUTION_PROVIDER),
    ] {
        let rejected = CudaExecutorDescriptor::new(
            solver,
            execution,
            snapshot.clone(),
            QueueSlot::new(snapshot.id(), 0),
            CudaLinearSolver::capabilities(),
        )
        .unwrap();
        assert!(select(rejected, normal(), SolverPlanningObjective::Fast).is_err());
    }
}

#[test]
fn cuda_owner_selection_is_single_use_and_cannot_bypass_retained_binding() {
    let normal = || {
        profile(
            Properties::SymmetricPositiveDefinite,
            Some(true),
            ReductionPolicy::Fast,
        )
    };
    let manual = CudaLinearSolver::new(0);
    assert!(
        manual
            .selection()
            .unwrap_err()
            .message()
            .contains("no selected deployment")
    );
    manual.admit_device(&device(0)).unwrap();
    manual.admit_device(&device(0)).unwrap();

    let selected = select(executor(0, 0), normal(), SolverPlanningObjective::Fast).unwrap();
    let error = selected
        .select_plan(
            executor(0, 0),
            normal(),
            SolverPlanningObjective::Fast,
            1e-12,
            1e-14,
            NonZeroUsize::new(4096).unwrap(),
        )
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("already retains a selected deployment")
    );

    struct One;
    impl eqiora_solver::CsrStorage for One {
        fn rows(&self) -> usize {
            1
        }
        fn columns(&self) -> usize {
            1
        }
        fn row_offsets(&self) -> &[usize] {
            &[0, 1]
        }
        fn column_indices(&self) -> &[usize] {
            &[0]
        }
        fn values(&self) -> &[f64] {
            &[1.]
        }
    }
    let system =
        CanonicalCsrSystemView::new(&One, &[1.], Properties::SymmetricPositiveDefinite).unwrap();
    let selected = select(executor(0, 0), normal(), SolverPlanningObjective::Fast).unwrap();
    let plan = selected.selection().unwrap().solver_plan();
    // A wrong-size guess makes this probe safe even if the state guard regresses:
    // the asserted diagnostic must still come from retained-selection ownership.
    let error = selected.solve(&system, Some(&[0., 0.]), plan).unwrap_err();
    assert!(error.message().contains("retained decision binding"));
    assert_eq!(selected.selection().unwrap().solver_plan(), plan);
}
