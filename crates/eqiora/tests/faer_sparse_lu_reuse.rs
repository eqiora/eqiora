use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::compiler::compile;
use eqiora::diagnostic::codes;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, ResolvedRealization, SemanticRevision, Space, SpatialDimensionSupport,
    Target, TargetCapabilities, VectorLayoutKind, resolve,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties, LinearSolver,
    LinearSolverBackend, PreconditionerPolicy, ReductionPolicy, SERIAL_EXECUTION_PROVIDER,
    SolverPlan,
};
use eqiora_backend_faer::{FAER_SOLVER_PROVIDER, FaerLinearSolver};
use eqiora_execution::{
    AcceptedLinearExecution, AdmittedExecution, DeploymentBinding, HostExecutorDescriptor,
};
use eqiora_numerics::scalar::{
    FinalizedScalarEllipticParameterPoint, finalize_scalar_elliptic_parameter_point,
    lower_scalar_elliptic_cartesian,
};

const SOURCE: &str = r#"model faer_sparse_lu_reuse() {
  domain interval = box(0, 1);
  domain lower_end = boundary(interval, axis = 0, side = lower);
  domain upper_end = boundary(interval, axis = 0, side = upper);

  variable potential: 1 on interval;
  parameter source_scale: 1 / m ^ 2 = 2;
  parameter diffusion: 1 = 1;
  parameter boundary_offset: 1 = 0;
  relation balance on interval {
    -div(diffusion * grad(potential)) - source_scale = 0;
  }
  relation lower_value on lower_end {
    trace(potential) - boundary_offset = 0;
  }
  relation upper_value on upper_end {
    trace(potential) - boundary_offset = 0;
  }
}
"#;

const ABSOLUTE_TOLERANCE: f64 = f64::from_bits(0x3e10_0000_0000_0000);
const SOLUTION_ERROR_CEILING: f64 = f64::from_bits(0x3df0_0000_0000_0000);

struct Fixture {
    plan: SolverPlan,
    points: [FinalizedScalarEllipticParameterPoint; 3],
}

#[test]
fn faer_prepared_run_preserves_scientific_acceptance_and_private_reuse_contract() {
    run_independent_oracle_checker();
    let fixture = fixture();
    let cold = fixture
        .points
        .iter()
        .map(|point| {
            FaerLinearSolver
                .with_prepared_linear(fixture.plan, |solve| solve(admit(point)))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let warm = FaerLinearSolver
        .with_prepared_linear(fixture.plan, |solve| {
            fixture
                .points
                .iter()
                .map(|point| solve(admit(point)))
                .collect::<Result<Vec<_>, _>>()
        })
        .unwrap();
    assert_eq!(warm.len(), 3);
    for (index, (warm, cold)) in warm.iter().zip(&cold).enumerate() {
        assert_same_acceptance(warm, cold);
        assert_point_acceptance(warm, &fixture.points[index], [0.25, 0.5, 0.2][index]);
    }
    assert_ne!(warm[0].receipt().operator(), warm[1].receipt().operator());
}

#[test]
fn faer_prepared_run_is_ephemeral_and_rejects_foreign_candidates() {
    let fixture = fixture();
    let foreign_source = SOURCE.replace("faer_sparse_lu_reuse", "foreign_faer_reuse");
    let foreign_program = compile_program("foreign-faer-prepared.eqi", &foreign_source);
    let foreign_resolved = resolve_plan(&foreign_program, fixture.plan);
    let foreign = finalize_scalar_elliptic_parameter_point(
        lower_scalar_elliptic_cartesian(&foreign_program).unwrap(),
        &foreign_resolved,
    )
    .unwrap();
    let error = FaerLinearSolver
        .with_prepared_linear(fixture.plan, |solve| {
            solve(admit(&fixture.points[0]))?;
            solve(admit(&foreign))
        })
        .unwrap_err();
    assert!(error.message().contains("portable graph"));

    let retry = FaerLinearSolver
        .with_prepared_linear(fixture.plan, |solve| {
            Ok([
                solve(admit(&fixture.points[0]))?,
                solve(admit(&fixture.points[1]))?,
            ])
        })
        .unwrap();
    assert_eq!(retry.len(), 2);
}

#[test]
fn failed_numeric_candidate_cannot_replace_the_last_accepted_factors() {
    let fixture = fixture();
    let singular_storage = Storage {
        values: [0.0],
        right_hand_side: [1.0],
    };
    let singular = CanonicalCsrSystemView::new(
        &singular_storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let accepted = FaerLinearSolver
        .with_prepared_linear(fixture.plan, |solve| {
            solve(admit(&fixture.points[0]))?;
            let error = solve(admit_system(&fixture.points[0], &singular)).unwrap_err();
            assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
            solve(admit(&fixture.points[1]))
        })
        .unwrap();
    assert_point_acceptance(&accepted, &fixture.points[1], 0.5);
}

fn fixture() -> Fixture {
    let plan = sparse_lu_plan();
    let program = compile_program("faer-sparse-lu-reuse.eqi", SOURCE);
    let resolved = resolve_plan(&program, plan);
    let base = lower_scalar_elliptic_cartesian(&program).unwrap();
    let parameters = base.parameter_fields().to_vec();
    let p0 = finalize_scalar_elliptic_parameter_point(base.clone(), &resolved).unwrap();
    let p1 = finalize_scalar_elliptic_parameter_point(
        base.bind_selected_parameters(&parameters, &[1.0, 4.0, 0.0])
            .unwrap(),
        &resolved,
    )
    .unwrap();
    let p2 = finalize_scalar_elliptic_parameter_point(
        base.bind_selected_parameters(&parameters, &[1.25, 2.0, 0.0])
            .unwrap(),
        &resolved,
    )
    .unwrap();
    Fixture {
        plan,
        points: [p0, p1, p2],
    }
}

fn compile_program(file: &str, source: &str) -> KernelProgram {
    let mut compiled = compile(file, source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn sparse_lu_plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::SparseLu,
        0.0,
        ABSOLUTE_TOLERANCE,
        NonZeroUsize::MIN,
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn resolve_plan(program: &KernelProgram, solver: SolverPlan) -> ResolvedRealization {
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(2).unwrap(),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            eqiora::realization::MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::MIN),
        )],
        [VectorLayoutKind::Replicated],
        FaerLinearSolver.capabilities(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap();
    resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(31),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &capabilities,
    )
    .unwrap()
}

fn admit(point: &FinalizedScalarEllipticParameterPoint) -> AdmittedExecution<'_> {
    admit_system(point, point.canonical_csr_system_view())
}

fn admit_system<'a>(
    point: &FinalizedScalarEllipticParameterPoint,
    system: &'a CanonicalCsrSystemView,
) -> AdmittedExecution<'a> {
    let binding = DeploymentBinding::bind_host(
        point.portable_realization(),
        HostExecutorDescriptor::new(
            FAER_SOLVER_PROVIDER,
            SERIAL_EXECUTION_PROVIDER,
            NonZeroUsize::MIN,
            FaerLinearSolver.capabilities(),
        ),
    )
    .unwrap();
    AdmittedExecution::admit_host_linear(point.portable_realization(), system, binding).unwrap()
}

struct Storage {
    values: [f64; 1],
    right_hand_side: [f64; 1],
}

impl CompleteCsrStorage for Storage {
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
        &self.values
    }
    fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }
}

fn assert_point_acceptance(
    accepted: &AcceptedLinearExecution,
    point: &FinalizedScalarEllipticParameterPoint,
    expected: f64,
) {
    assert_eq!(accepted.solution().values().len(), 1);
    assert!((accepted.solution().values()[0] - expected).abs() <= SOLUTION_ERROR_CEILING);
    assert!(accepted.solution().report().true_residual_norm() <= ABSOLUTE_TOLERANCE);
    assert_eq!(
        accepted.receipt().operator(),
        point.canonical_csr_system_view().agreement_fingerprint()
    );
    assert_eq!(accepted.receipt().solver_provider(), FAER_SOLVER_PROVIDER);
}

fn assert_same_acceptance(left: &AcceptedLinearExecution, right: &AcceptedLinearExecution) {
    assert_eq!(left.solution(), right.solution());
    assert_eq!(left.receipt(), right.receipt());
    assert_eq!(
        left.solution().report().true_residual_norm().to_bits(),
        right.solution().report().true_residual_norm().to_bits()
    );
}

fn run_independent_oracle_checker() {
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .unwrap();
    let output = std::process::Command::new("python3")
        .current_dir(repository)
        .arg("verify/numerics/faer-sparse-lu-reuse/run_case.py")
        .arg("--check")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "sparse-LU reuse oracle failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
