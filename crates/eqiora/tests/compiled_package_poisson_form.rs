use eqiora_core::ScalarType;
use std::f64::consts::PI;
use std::fmt;
use std::num::{NonZeroU16, NonZeroUsize};
use std::sync::Mutex;

use eqiora::Diagnostic;
use eqiora::assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan,
    AssemblyResult, AssemblyWork, LocalContribution, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora::compiler::compile;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::meshing::QuadratureRule;
use eqiora::package::{
    BundleRoleV1, InMemoryPackageStore, ModelPackageIdentityV1, NormalizedRelativePath,
    PackageManifestV1, PackageReleaseV1, PackageSourcesV1, PackagedModelDocument, ResolutionNodeV1,
    ResolutionRecordV1, SourceFileV1, prepare_package_release_v1,
};
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, ResolvedRealization, SemanticRevision, Space, Target, VectorLayoutKind,
    resolve,
};
use eqiora::solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use eqiora_meshing::CartesianMesh;
use eqiora_numerics::scalar::{
    ResolvedScalarEllipticCartesianSolution, lower_scalar_elliptic_cartesian,
    solve_resolved_scalar_elliptic_cartesian,
    solve_resolved_scalar_elliptic_cartesian_with_assembly,
    solve_scalar_elliptic_cartesian_fem_with_assembly,
};
use eqiora_sem::KernelProgram;

const MANIFEST: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/org.example.poisson/package.json");
const SOURCE: &str =
    include_str!("../../../crates/eqiora-api/packages/org.example.poisson/src/main.eqi");
const README: &[u8] =
    include_bytes!("../../../crates/eqiora-api/packages/org.example.poisson/README.md");
const SHADOW_RELATIVE: f64 = 1.0e-15;
const MAXIMUM_BALANCE: f64 = 2.0e-11;

#[test]
fn e2_actual_package_reaches_verification_only_q1_and_retains_convergence_and_balance() {
    let packaged = compile_package();
    let program = packaged.model().program();
    let mut errors = Vec::new();
    for cells in [4, 8, 16, 32] {
        let resolved = resolved(program, cells);
        let (_, solution) =
            solve_resolved_scalar_elliptic_cartesian(program, &resolved, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
        let ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) = solution else {
            panic!("Q1 Realization returned a non-FEM solution");
        };
        let source = solution.integrated_source();
        let reaction = solution.boundary_reaction_sum();
        let relative_imbalance = (reaction + source).abs() / (reaction.abs() + source.abs());
        assert!(relative_imbalance <= MAXIMUM_BALANCE);
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 4).unwrap();
        errors.push(
            solution
                .field()
                .l2_error(
                    &|point: &[f64]| (PI * point[0]).sin() * (PI * point[1]).sin(),
                    &quadrature,
                )
                .unwrap(),
        );
    }
    for pair in errors.windows(2) {
        assert!((pair[0] / pair[1]).log2() >= 1.9, "{errors:?}");
    }
}

#[test]
fn e2_compiled_and_existing_paths_match_every_local_csr_and_rhs() {
    let program = compile_program(SOURCE).unwrap();
    let resolved = resolved(&program, 4);
    let compiled_backend = RecordingBackend::default();
    let (model, compiled_solution) = solve_resolved_scalar_elliptic_cartesian_with_assembly(
        &program,
        &resolved,
        &compiled_backend,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let ResolvedScalarEllipticCartesianSolution::FiniteElement(compiled_solution) =
        compiled_solution
    else {
        panic!("Q1 Realization returned a non-FEM solution");
    };

    let mesh = CartesianMesh::uniform(model.bounds(), &[4, 4]).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let source = |point: &[f64]| model.source().evaluate(point).unwrap_or(f64::NAN);
    let coefficient = |point: &[f64]| {
        model
            .coefficient_expression()
            .evaluate(point)
            .unwrap_or(f64::NAN)
    };
    let legacy_backend = RecordingBackend::default();
    let legacy_solution = solve_scalar_elliptic_cartesian_fem_with_assembly(
        &mesh,
        &coefficient,
        &source,
        &|_: &[f64]| 0.0,
        &quadrature,
        &legacy_backend,
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, resolved.plan().solver()),
    )
    .unwrap();

    assert_relative(
        compiled_solution.field().vertex_values(),
        legacy_solution.field().vertex_values(),
        SHADOW_RELATIVE,
    );
    compare_capture(
        &compiled_backend.take(),
        &legacy_backend.take(),
        SHADOW_RELATIVE,
    );

    let under_integrated = resolved_with_points(&program, 4, 1);
    let error = solve_resolved_scalar_elliptic_cartesian_with_assembly(
        &program,
        &under_integrated,
        &REFERENCE_ASSEMBLY_BACKEND,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap_err();
    assert!(
        error.message().contains("realization compatibility"),
        "{error:?}"
    );
}

#[test]
fn e2_role_assignment_rejects_missing_duplicate_and_reclassified_roles() {
    let missing = SOURCE.replace(
        "  relation y_upper_value on y_upper { trace(potential) = 0; }\n",
        "",
    );
    assert_role_gate(compile_program(&missing));

    let duplicate = SOURCE.replace(
        "  relation y_upper_value on y_upper { trace(potential) = 0; }\n",
        concat!(
            "  relation y_upper_value on y_upper { trace(potential) = 0; }\n",
            "  relation y_upper_value_duplicate on y_upper { trace(potential) = 0; }\n",
        ),
    );
    assert_role_gate(compile_program(&duplicate));

    let reclassified = SOURCE.replace(
        "  parameter source_scale: 1 / m ^ 2 = 19.739208802178716;",
        "  variable source_scale: 1 / m ^ 2 on square;",
    );
    assert_role_gate(compile_program(&reclassified));
}

#[test]
fn e2_conservation_and_shadow_reject_source_omission_and_dof_shift() {
    let program = compile_program(SOURCE).unwrap();
    let resolved = resolved(&program, 4);
    let baseline = RecordingBackend::default();
    solve_resolved_scalar_elliptic_cartesian_with_assembly(
        &program,
        &resolved,
        &baseline,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let baseline = baseline.take();

    let omitted = MutatingBackend::new(Mutation::OmitSource);
    let (_, solution) = solve_resolved_scalar_elliptic_cartesian_with_assembly(
        &program,
        &resolved,
        &omitted,
        &REFERENCE_LINEAR_SOLVER,
    )
    .unwrap();
    let ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) = solution else {
        panic!("Q1 Realization returned a non-FEM solution");
    };
    let semantic_source = omitted.original_source_sum();
    let imbalance = (solution.boundary_reaction_sum() + semantic_source).abs()
        / (solution.boundary_reaction_sum().abs() + semantic_source.abs());
    assert!(imbalance > MAXIMUM_BALANCE);

    let shifted = MutatingBackend::new(Mutation::ShiftDof);
    let shifted_result = solve_resolved_scalar_elliptic_cartesian_with_assembly(
        &program,
        &resolved,
        &shifted,
        &REFERENCE_LINEAR_SOLVER,
    );
    match shifted_result {
        Ok(_) => {
            let shifted = shifted.take();
            assert!(!captures_match(&baseline, &shifted, SHADOW_RELATIVE));
        }
        Err(_) => {
            // An invalid shifted map is also rejected before a result escapes.
        }
    }
}

fn compile_package() -> PackagedModelDocument {
    let manifest = PackageManifestV1::from_json(MANIFEST).unwrap();
    let sources = PackageSourcesV1::new(
        manifest,
        vec![
            SourceFileV1::new(
                NormalizedRelativePath::parse("README.md").unwrap(),
                BundleRoleV1::Documentation,
                README.to_vec(),
            ),
            SourceFileV1::new(
                NormalizedRelativePath::parse("src/main.eqi").unwrap(),
                BundleRoleV1::ModelSource,
                SOURCE.as_bytes().to_vec(),
            ),
        ],
    )
    .unwrap();
    let release = prepare_package_release_v1(sources, &[]).unwrap();
    let (store, resolution) = install(&release);
    PackagedModelDocument::compile_locked(&store, &resolution, "Main").unwrap()
}

fn install(release: &PackageReleaseV1) -> (InMemoryPackageStore, ResolutionRecordV1) {
    let identity: ModelPackageIdentityV1 = release.package_identity().unwrap();
    let mut store = InMemoryPackageStore::default();
    let source = store.insert(release).unwrap();
    let resolution = ResolutionRecordV1::new(
        identity.clone(),
        vec![ResolutionNodeV1::new(identity, source)],
        vec![],
    )
    .unwrap();
    (store, resolution)
}

fn compile_program(source: &str) -> Result<KernelProgram, Diagnostic> {
    let mut compiled = compile("compiled-package-poisson.eqi", source)
        .map_err(|errors| errors.into_iter().next().unwrap())?;
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store
        .commit(transaction)
        .map_err(|errors| errors.into_iter().next().unwrap())?;
    KernelProgram::from_snapshot(&store.snapshot(), model)
        .map_err(|errors| errors.into_iter().next().unwrap())
}

fn resolved(program: &KernelProgram, cells: usize) -> ResolvedRealization {
    resolved_with_points(program, cells, 2)
}

fn resolved_with_points(
    program: &KernelProgram,
    cells: usize,
    points_per_axis: usize,
) -> ResolvedRealization {
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).unwrap(),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(points_per_axis).unwrap(),
            },
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(cells * cells * 8).unwrap(),
        )
        .unwrap(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(1),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap()
}

fn assert_role_gate(result: Result<KernelProgram, Diagnostic>) {
    let program = result.unwrap();
    let error = lower_scalar_elliptic_cartesian(&program).unwrap_err();
    assert!(error.message().contains("role assignment"), "{error:?}");
}

#[derive(Debug, Clone)]
struct Capture {
    locals: Vec<LocalContribution>,
    systems: Vec<eqiora::assembly::LinearSystem>,
}

#[derive(Debug, Default)]
struct RecordingBackend {
    capture: Mutex<Option<Capture>>,
}

impl RecordingBackend {
    fn take(&self) -> Capture {
        self.capture.lock().unwrap().take().unwrap()
    }
}

impl AssemblyBackend for RecordingBackend {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        let locals = Mutex::new(Vec::new());
        let recording = RecordingWork {
            inner: work,
            locals: &locals,
        };
        let result = REFERENCE_ASSEMBLY_BACKEND.assemble(plan, &recording)?;
        let systems = result.clone().into_parts().0;
        *self.capture.lock().unwrap() = Some(Capture {
            locals: locals.into_inner().unwrap(),
            systems,
        });
        Ok(result)
    }
}

struct RecordingWork<'a> {
    inner: &'a dyn AssemblyWork,
    locals: &'a Mutex<Vec<LocalContribution>>,
}

impl fmt::Debug for RecordingWork<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RecordingWork")
    }
}

impl AssemblyWork for RecordingWork<'_> {
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.inner.packet_set_identity()
    }

    fn packet_count(&self) -> usize {
        self.inner.packet_count()
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic> {
        let packet = self.inner.evaluate(packet_index)?;
        self.locals.lock().unwrap().push(packet.local().clone());
        Ok(packet)
    }
}

#[derive(Debug, Clone, Copy)]
enum Mutation {
    OmitSource,
    ShiftDof,
}

#[derive(Debug)]
struct MutatingBackend {
    mutation: Mutation,
    original_source_sum: Mutex<f64>,
    capture: Mutex<Option<Capture>>,
}

impl MutatingBackend {
    fn new(mutation: Mutation) -> Self {
        Self {
            mutation,
            original_source_sum: Mutex::new(0.0),
            capture: Mutex::new(None),
        }
    }

    fn original_source_sum(&self) -> f64 {
        *self.original_source_sum.lock().unwrap()
    }

    fn take(&self) -> Capture {
        self.capture.lock().unwrap().take().unwrap()
    }
}

impl AssemblyBackend for MutatingBackend {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        *self.original_source_sum.lock().unwrap() = 0.0;
        let locals = Mutex::new(Vec::new());
        let mutated = MutatingWork {
            inner: work,
            mutation: self.mutation,
            source_sum: &self.original_source_sum,
            locals: &locals,
        };
        let result = REFERENCE_ASSEMBLY_BACKEND.assemble(plan, &mutated)?;
        let systems = result.clone().into_parts().0;
        *self.capture.lock().unwrap() = Some(Capture {
            locals: locals.into_inner().unwrap(),
            systems,
        });
        Ok(result)
    }
}

struct MutatingWork<'a> {
    inner: &'a dyn AssemblyWork,
    mutation: Mutation,
    source_sum: &'a Mutex<f64>,
    locals: &'a Mutex<Vec<LocalContribution>>,
}

impl fmt::Debug for MutatingWork<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MutatingWork")
    }
}

impl AssemblyWork for MutatingWork<'_> {
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.inner.packet_set_identity()
    }

    fn packet_count(&self) -> usize {
        self.inner.packet_count()
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic> {
        let packet = self.inner.evaluate(packet_index)?;
        if matches!(self.mutation, Mutation::OmitSource) {
            *self.source_sum.lock().unwrap() += packet.local().rhs().iter().sum::<f64>();
        }
        let local = match self.mutation {
            Mutation::OmitSource => LocalContribution::new(
                packet.local().rows(),
                packet.local().columns(),
                packet.local().matrix().to_vec(),
                vec![0.0; packet.local().rows()],
            )?,
            Mutation::ShiftDof => packet.local().clone(),
        };
        self.locals.lock().unwrap().push(local.clone());
        let mappings = packet
            .mappings()
            .iter()
            .map(|mapping| mutate_mapping(mapping, self.mutation, packet_index))
            .collect::<Result<Vec<_>, _>>()?;
        AssemblyPacket::new(local, mappings)
    }
}

fn mutate_mapping(
    mapping: &TargetAssemblyMap,
    mutation: Mutation,
    packet_index: usize,
) -> Result<TargetAssemblyMap, Diagnostic> {
    if !matches!(mutation, Mutation::ShiftDof) || packet_index != 0 || mapping.target().index() != 1
    {
        return Ok(mapping.clone());
    }
    let mut equations = mapping.map().equations().to_vec();
    let mut unknowns = mapping.map().unknowns().to_vec();
    equations.swap(0, 1);
    unknowns.swap(0, 1);
    Ok(TargetAssemblyMap::new(
        mapping.target(),
        AssemblyMap::new(equations, unknowns)?,
    ))
}

fn compare_capture(left: &Capture, right: &Capture, tolerance: f64) {
    assert_eq!(left.locals.len(), right.locals.len());
    for (left, right) in left.locals.iter().zip(&right.locals) {
        assert_relative(left.matrix(), right.matrix(), tolerance);
        assert_relative(left.rhs(), right.rhs(), tolerance);
    }
    assert_eq!(left.systems.len(), right.systems.len());
    for (left, right) in left.systems.iter().zip(&right.systems) {
        assert_eq!(left.matrix().row_offsets(), right.matrix().row_offsets());
        assert_eq!(
            left.matrix().column_indices(),
            right.matrix().column_indices()
        );
        assert_relative(left.matrix().values(), right.matrix().values(), tolerance);
        assert_relative(left.rhs(), right.rhs(), tolerance);
    }
}

fn captures_match(left: &Capture, right: &Capture, tolerance: f64) -> bool {
    if left.systems.len() != right.systems.len() {
        return false;
    }
    left.systems
        .iter()
        .zip(&right.systems)
        .all(|(left, right)| {
            left.matrix().row_offsets() == right.matrix().row_offsets()
                && left.matrix().column_indices() == right.matrix().column_indices()
                && maximum_relative(left.matrix().values(), right.matrix().values()) <= tolerance
                && maximum_relative(left.rhs(), right.rhs()) <= tolerance
        })
}

fn assert_relative(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        maximum_relative(actual, expected) <= tolerance,
        "actual={actual:?}, expected={expected:?}"
    );
}

fn maximum_relative(actual: &[f64], expected: &[f64]) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}
