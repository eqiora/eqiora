//! Bounded #454 continuity proof; the fixture and falsifiers were frozen before implementation.
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use eqiora::{
    Diagnostic,
    api::ModelDocument,
    artifact::{
        CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1,
        MeshProductionLineageEnvelopeV1, ModelEnvelope,
    },
    compiler::StaticBindingValue,
    diagnostic::codes,
    geometry::GeometryGraph,
    solver::{
        BackendId, CanonicalCsrSystemView, CsrStorage, LinearOperatorProperties, LinearProblem,
        LinearSolution, LinearSolver, LinearSolverBackend, PreconditionerPolicy, ProviderLibrary,
        REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, ReductionPolicy,
        ReplicatedLinearExecution, SERIAL_EXECUTION_PROVIDER, SolveReport, SolverCapabilities,
        SolverPlan, SolverProvider,
    },
};
use eqiora_backend_faer::{
    FAER_ADAPTER_VERSION, FAER_SOLVER_PROVIDER, FAER_VERSION, FaerLinearSolver,
};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, ResolvedCommonPlan,
};

// Same scalar-Poisson source/geometry pattern as support/canonical_poisson.eqi,
// specialized to constant unit source. No provider enters the compiled Model.
const SOURCE: &str = r#"
public component ContinuityPoisson(
 support square:volume(ambient_dimension=2),
 support x_lower:boundary(parent=square), support x_upper:boundary(parent=square),
 support y_lower:boundary(parent=square), support y_upper:boundary(parent=square)
) {
 variable potential:1 on square in h1;
 relation balance on square {-div(grad(potential))=1[1/m^2];}
 relation x_lower_value on x_lower {trace(potential)=0;}
 relation x_upper_value on x_upper {trace(potential)=0;}
 relation y_lower_value on y_lower {trace(potential)=0;}
 relation y_upper_value on y_upper {trace(potential)=0;}
}
"#;

// Integral TPFA cell balance, derived before running either provider:
// h=1/2, interior face conductance h/h=1, boundary conductance h/(h/2)=2.
// Each corner cell has two of each: diagonal 6, two off-diagonals -1.
// Unit source integrates to h^2=1/4. Row sum 4 gives u=1/16.
// Eigenvalues are 4,6,6,8, so ||u-u*||_2 <= ||Au-b||_2/4.
const OFFSETS: &[usize] = &[0, 3, 6, 9, 12];
const COLUMNS: &[usize] = &[0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3];
const VALUES: &[f64] = &[6., -1., -1., -1., 6., -1., -1., 6., -1., -1., -1., 6.];
const RHS: &[f64] = &[0.25; 4];
const COMPONENT_ERROR: f64 = 5.0e-13; // Existing same-problem faer_spatial bound.
const RESIDUAL_TARGET: f64 = 5.0e-13; // max(1e-14, 1e-12 * ||[1/4;4]||_2).
const STALE_FAER_LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("faer", "0.24.3")];

struct Fixture {
    model: ModelEnvelope,
    mesh: AuthenticatedCommonMesh,
}

impl Fixture {
    fn new() -> Self {
        let graph = GeometryGraph::new();
        let square = graph.rectangle([0., 1.], [0., 1.]).unwrap();
        let edges = square.boundaries();
        let geometry = graph
            .build(
                &square,
                &BTreeMap::from([
                    ("square".into(), vec![square.region().into()]),
                    ("x_lower".into(), vec![edges[0].into()]),
                    ("x_upper".into(), vec![edges[1].into()]),
                    ("y_lower".into(), vec![edges[2].into()]),
                    ("y_upper".into(), vec![edges[3].into()]),
                ]),
            )
            .unwrap();
        let body = geometry.entity_set("square").unwrap();
        let bindings = ["square", "x_lower", "x_upper", "y_lower", "y_upper"].map(|name| {
            (
                name,
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set(name).unwrap(),
                    parent: (name != "square").then_some(body),
                },
            )
        });
        let document = ModelDocument::compile_selected(
            "provider-continuity.eqi",
            SOURCE,
            "ContinuityPoisson",
            &bindings,
        )
        .unwrap();
        let cells = CartesianMeshCellsV2::new([2, 2]).unwrap();
        let (mesh, correspondence) =
            GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
                &geometry,
                [2, 2],
            )
            .unwrap();
        let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
            &cells,
            &geometry,
            &mesh,
            &correspondence,
        )
        .unwrap();
        Self {
            model: ModelEnvelope::from_program(document.program()).unwrap(),
            mesh: AuthenticatedCommonMesh::structured_cartesian(
                geometry,
                mesh,
                correspondence,
                production,
            )
            .unwrap(),
        }
    }

    fn resolve(
        &self,
        plan: SolverPlan,
        provider: SolverProvider,
        backend: &dyn LinearSolverBackend,
    ) -> Result<ResolvedCommonPlan, Diagnostic> {
        ResolvedCommonPlan::resolve(
            &self.model,
            self.mesh.clone(),
            CommonSpatialPolicy::CellCenteredTpfa,
            CommonSolvePolicy::Linear(CommonLinearRequest::exact(plan, provider)?),
            None,
            None,
            backend,
            None,
            None,
        )
    }
}

fn plan(reduction: ReductionPolicy) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(4096).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(reduction)
}

#[derive(Debug, Clone, Copy)]
enum Behavior {
    Delegate,
    DoubledOperator,
    ReferenceFallback,
}

#[derive(Debug)]
struct Call {
    system: CanonicalCsrSystemView,
    plan: SolverPlan,
    values: Vec<f64>,
    report: SolveReport,
}

#[derive(Debug)]
struct AuditedBackend<'a> {
    delegate: &'a dyn LinearSolverBackend,
    behavior: Behavior,
    provider: SolverProvider,
    capabilities: SolverCapabilities,
    solve_attempts: AtomicUsize,
    calls: Mutex<Vec<Call>>,
    fallback_calls: Mutex<usize>,
}

impl<'a> AuditedBackend<'a> {
    fn new(delegate: &'a dyn LinearSolverBackend, behavior: Behavior) -> Self {
        Self {
            delegate,
            behavior,
            provider: delegate.provider(),
            capabilities: delegate.capabilities(),
            solve_attempts: AtomicUsize::new(0),
            calls: Mutex::new(Vec::new()),
            fallback_calls: Mutex::new(0),
        }
    }

    fn assert_no_work(&self) {
        assert_eq!(self.solve_attempts.load(Ordering::SeqCst), 0);
        assert!(self.calls.lock().unwrap().is_empty());
        assert_eq!(*self.fallback_calls.lock().unwrap(), 0);
    }
}

impl LinearSolverBackend for AuditedBackend<'_> {
    fn provider(&self) -> SolverProvider {
        self.provider
    }

    fn capabilities(&self) -> SolverCapabilities {
        self.capabilities.clone()
    }

    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        self.solve_attempts.fetch_add(1, Ordering::SeqCst);
        let system = problem
            .canonical_csr_system()
            .expect("ordinary captured CSR boundary");
        let solution = match self.behavior {
            Behavior::Delegate => self
                .delegate
                .solve_with_execution(problem, plan, execution)?,
            Behavior::DoubledOperator => {
                // A plausible stale/scaled-operator adapter bug. The selected production
                // solver truthfully succeeds on 2A; the outer ordinary owner still owns A.
                let storage = DoubledStorage {
                    original: system,
                    values: system.values().iter().map(|value| 2. * value).collect(),
                };
                let wrong = CanonicalCsrSystemView::new(
                    &storage,
                    system.right_hand_side(),
                    problem.properties(),
                )?;
                self.delegate
                    .solve_with_execution(&wrong.linear_problem()?, plan, execution)?
            }
            Behavior::ReferenceFallback => {
                *self.fallback_calls.lock().unwrap() += 1;
                REFERENCE_LINEAR_SOLVER.solve_with_execution(
                    problem,
                    plan.with_reduction(ReductionPolicy::Reproducible),
                    execution,
                )?
            }
        };
        self.calls.lock().unwrap().push(Call {
            system: system.clone(),
            plan,
            values: solution.values().to_vec(),
            report: solution.report().clone(),
        });
        Ok(solution)
    }
}

struct DoubledStorage<'a> {
    original: &'a CanonicalCsrSystemView,
    values: Vec<f64>,
}

impl CsrStorage for DoubledStorage<'_> {
    fn rows(&self) -> usize {
        self.original.rows()
    }
    fn columns(&self) -> usize {
        self.original.columns()
    }
    fn row_offsets(&self) -> &[usize] {
        self.original.row_offsets()
    }
    fn column_indices(&self) -> &[usize] {
        self.original.column_indices()
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
}

fn assert_frozen_system(system: &CanonicalCsrSystemView) {
    assert_eq!(system.rows(), 4);
    assert_eq!(system.columns(), 4);
    assert_eq!(
        system.properties(),
        LinearOperatorProperties::SymmetricPositiveDefinite
    );
    assert_eq!(system.row_offsets(), OFFSETS);
    assert_eq!(system.column_indices(), COLUMNS);
    assert_eq!(system.values(), VALUES);
    assert_eq!(system.right_hand_side(), RHS);
}

fn independent_residual(values: &[f64]) -> f64 {
    assert_eq!(values.len(), 4);
    // Explicit balance equations, independent of captured CSR/operator application.
    let [a, b, c, d] = [values[0], values[1], values[2], values[3]];
    [
        6. * a - b - c - 0.25,
        6. * b - a - d - 0.25,
        6. * c - a - d - 0.25,
        6. * d - b - c - 0.25,
    ]
    .iter()
    .map(|r| r * r)
    .sum::<f64>()
    .sqrt()
}

fn assert_oracle(result: &CommonResult, backend: &AuditedBackend<'_>, expected: SolverPlan) {
    assert_eq!(result.field_count(), 1);
    assert_eq!(result.field_block_count(0), 1);
    let (association, values, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(association, "cell");
    assert_eq!(shape.iter().product::<usize>(), 4);
    assert_eq!(values.len(), 4);
    for value in values {
        assert!((value - 1. / 16.).abs() <= COMPONENT_ERROR);
    }
    assert!(independent_residual(values) <= RESIDUAL_TARGET);
    let calls = backend.calls.lock().unwrap();
    assert_eq!(backend.solve_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(calls.len(), 1);
    assert_eq!(*backend.fallback_calls.lock().unwrap(), 0);
    let call = &calls[0];
    assert_frozen_system(&call.system);
    assert_eq!(call.plan, expected);
    assert_eq!(call.report.solver_plan(), expected);
    assert_eq!(call.report.solver_provider(), backend.delegate.provider());
    assert_eq!(call.report.execution_provider(), SERIAL_EXECUTION_PROVIDER);
    assert_eq!(call.report.residual_target(), RESIDUAL_TARGET);
    assert!(call.report.true_residual_norm() <= RESIDUAL_TARGET);
    assert_eq!(call.values, values);
    let evidence = result.solve_evidence(None, None).unwrap();
    assert_eq!(evidence.algorithm(), LinearSolver::ConjugateGradient);
    assert_eq!(evidence.preconditioner(), PreconditionerPolicy::Identity);
    assert_eq!(evidence.reduction(), expected.reduction());
    assert_eq!(evidence.solver().id(), backend.provider.id().as_str());
    assert_eq!(
        evidence.solver().implementation_version(),
        backend.provider.implementation_version()
    );
    assert_eq!(
        evidence.solver().libraries(),
        backend
            .provider
            .libraries()
            .iter()
            .map(|library| (library.name().to_owned(), library.version().to_owned()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        evidence.execution().provider().id(),
        SERIAL_EXECUTION_PROVIDER.id().as_str()
    );
    assert!(evidence.true_residual_norm() <= RESIDUAL_TARGET);
    assert!(independent_residual(values) <= evidence.residual_target());
}

#[test]
fn ordinary_reference_and_pinned_faer_preserve_the_same_owned_problem() {
    let fixture = Fixture::new();
    let original = fixture.model.canonical_json().unwrap();
    let reference = AuditedBackend::new(&REFERENCE_LINEAR_SOLVER, Behavior::Delegate);
    let faer = AuditedBackend::new(&FaerLinearSolver, Behavior::Delegate);
    let reference_solver = plan(ReductionPolicy::Reproducible);
    let faer_solver = plan(ReductionPolicy::Fast);
    let reference_plan = fixture
        .resolve(reference_solver, REFERENCE_SOLVER_PROVIDER, &reference)
        .unwrap();
    let faer_plan = fixture
        .resolve(faer_solver, FAER_SOLVER_PROVIDER, &faer)
        .unwrap();
    let r = reference_plan.as_spatial().unwrap();
    let f = faer_plan.as_spatial().unwrap();
    assert_ne!(r.identity(), f.identity());
    assert_eq!(r.model_reference().unwrap(), f.model_reference().unwrap());
    assert_eq!(r.mesh_digest(), f.mesh_digest());
    assert_eq!(r.geometry_digest(), f.geometry_digest());
    assert_eq!(r.correspondence_digest(), f.correspondence_digest());
    assert_eq!(r.production_digest(), f.production_digest());
    assert_eq!(r.solver_provider(), REFERENCE_SOLVER_PROVIDER);
    assert_eq!(f.solver_provider(), FAER_SOLVER_PROVIDER);
    assert_eq!(FAER_VERSION, "0.24.4");
    assert_eq!(
        FAER_SOLVER_PROVIDER.implementation_version(),
        FAER_ADAPTER_VERSION
    );
    assert_eq!(
        FAER_SOLVER_PROVIDER.libraries(),
        &[ProviderLibrary::new("faer", "0.24.4")]
    );
    let reference_result = r.run_result(None, &reference).unwrap();
    let faer_result = f.run_result(None, &faer).unwrap();
    assert_oracle(&reference_result, &reference, reference_solver);
    assert_oracle(&faer_result, &faer, faer_solver);
    assert_ne!(reference_result.identity(), faer_result.identity());
    assert_eq!(reference_result.field(0), faer_result.field(0));
    assert_eq!(
        reference.calls.lock().unwrap()[0].system,
        faer.calls.lock().unwrap()[0].system
    );
    assert_eq!(fixture.model.canonical_json().unwrap(), original);
}

#[test]
fn ordinary_acceptance_rejects_success_on_the_wrong_operator() {
    let fixture = Fixture::new();
    for (delegate, reduction) in [
        (
            &REFERENCE_LINEAR_SOLVER as &dyn LinearSolverBackend,
            ReductionPolicy::Reproducible,
        ),
        (&FaerLinearSolver, ReductionPolicy::Fast),
    ] {
        let lying = AuditedBackend::new(delegate, Behavior::DoubledOperator);
        let resolved = fixture
            .resolve(plan(reduction), delegate.provider(), &lying)
            .unwrap();
        let error = resolved
            .as_spatial()
            .unwrap()
            .run_result(None, &lying)
            .unwrap_err();
        assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
        let calls = lying.calls.lock().unwrap();
        assert_eq!(lying.solve_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(calls.len(), 1);
        assert_eq!(*lying.fallback_calls.lock().unwrap(), 0);
        let call = &calls[0];
        assert_frozen_system(&call.system);
        assert_eq!(call.report.solver_provider(), delegate.provider());
        assert!(call.report.true_residual_norm() <= RESIDUAL_TARGET);
        for value in &call.values {
            assert!((value - 1. / 32.).abs() <= COMPONENT_ERROR);
        }
        assert!((independent_residual(&call.values) - 0.25).abs() <= COMPONENT_ERROR);
        assert!(error.message().contains("residual"));
    }
}

#[test]
fn stale_unknown_unsupported_and_silent_fallback_fail_closed() {
    let fixture = Fixture::new();
    let original = fixture.model.canonical_json().unwrap();
    let actual = AuditedBackend::new(&FaerLinearSolver, Behavior::Delegate);
    let stale_adapter = SolverProvider::new(
        FAER_SOLVER_PROVIDER.id(),
        "stale",
        FAER_SOLVER_PROVIDER.libraries(),
    );
    let stale_library = SolverProvider::new(
        FAER_SOLVER_PROVIDER.id(),
        FAER_ADAPTER_VERSION,
        STALE_FAER_LIBRARIES,
    );
    let unknown = SolverProvider::new(BackendId::new("eqiora.unknown"), "1", &[]);
    for unavailable in [stale_adapter, stale_library, unknown] {
        assert_eq!(
            fixture
                .resolve(plan(ReductionPolicy::Fast), unavailable, &actual)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
        actual.assert_no_work();
    }
    assert_eq!(
        fixture
            .resolve(
                plan(ReductionPolicy::Reproducible),
                FAER_SOLVER_PROVIDER,
                &actual
            )
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    actual.assert_no_work();
    let admitted = fixture
        .resolve(plan(ReductionPolicy::Fast), FAER_SOLVER_PROVIDER, &actual)
        .unwrap();
    for unavailable in [stale_adapter, stale_library, unknown] {
        let mut replacement = AuditedBackend::new(&FaerLinearSolver, Behavior::Delegate);
        replacement.provider = unavailable;
        assert_eq!(
            admitted
                .as_spatial()
                .unwrap()
                .run_result(None, &replacement)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
        replacement.assert_no_work();
    }
    let mut changed_capabilities = AuditedBackend::new(&FaerLinearSolver, Behavior::Delegate);
    changed_capabilities.capabilities = REFERENCE_LINEAR_SOLVER.capabilities();
    assert_eq!(
        admitted
            .as_spatial()
            .unwrap()
            .run_result(None, &changed_capabilities)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    changed_capabilities.assert_no_work();
    let fallback = AuditedBackend::new(&FaerLinearSolver, Behavior::ReferenceFallback);
    assert_eq!(
        admitted
            .as_spatial()
            .unwrap()
            .run_result(None, &fallback)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
    let calls = fallback.calls.lock().unwrap();
    assert_eq!(fallback.solve_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(calls.len(), 1);
    assert_eq!(*fallback.fallback_calls.lock().unwrap(), 1);
    assert_eq!(calls[0].report.solver_provider(), REFERENCE_SOLVER_PROVIDER);
    assert_frozen_system(&calls[0].system);
    actual.assert_no_work();
    assert_eq!(fixture.model.canonical_json().unwrap(), original);
}
