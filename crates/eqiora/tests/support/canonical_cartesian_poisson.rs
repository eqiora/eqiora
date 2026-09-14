use eqiora_core::ScalarType;
use std::f64::consts::PI;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::compiler::compile;
use eqiora::entity::EntityKind;
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::meshing::QuadratureRule;
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshKind, MeshPolicy,
    QuadraturePolicy, RealizationCapabilities, RealizationPlan, RealizationRequest,
    RealizationRequirements, RealizationRevision, SemanticRevision, Space, SpatialDimensionSupport,
    Target, TargetCapabilities, VectorLayoutKind,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverCapabilities, SolverCapability, SolverPlan,
};
use eqiora_numerics::{
    scalar::ResolvedScalarEllipticCartesianSolution,
    scalar::solve_resolved_scalar_elliptic_cartesian,
};

pub const SOURCE: &str = include_str!("canonical_poisson.eqi");
pub const CELLS_PER_AXIS: usize = 16;
pub const CPU_CUDA_ABSOLUTE: f64 = 2.0e-12;
pub const CPU_CUDA_RELATIVE: f64 = 2.0e-12;
pub const MAXIMUM_L2_ERROR: f64 = 2.0e-3;
pub const MAXIMUM_RELATIVE_BALANCE: f64 = 2.0e-11;

pub const METHODS: [(u64, DiscretizationMethod, &str); 2] = [
    (1, DiscretizationMethod::ContinuousGalerkin, "q1-fem"),
    (
        2,
        DiscretizationMethod::CellCenteredFiniteVolume,
        "cell-centered-tpfa",
    ),
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MethodMetrics {
    pub l2_error: f64,
    pub boundary_quantity: f64,
    pub integrated_source: f64,
    pub relative_balance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuConformance {
    pub maximum_absolute_error: f64,
    pub maximum_scaled_error: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    pub model_ulid: String,
    pub symbols: Vec<SourceSymbol>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSymbol {
    pub name: String,
    pub kind: &'static str,
    pub ulid: String,
}

#[allow(dead_code)] // Each integration target compiles only the helpers it consumes.
pub fn compile_program() -> Result<KernelProgram, String> {
    compile_program_with_identity().map(|(program, _)| program)
}

pub fn compile_program_with_identity() -> Result<(KernelProgram, SourceIdentity), String> {
    compile_program_with_identity_from_source("canonical-cartesian-poisson-cuda.eqi", SOURCE)
}

#[allow(dead_code)] // Used by feature-gated integration targets with their own evidence fixture.
pub fn compile_program_from_source(
    source_name: &str,
    source: &str,
) -> Result<KernelProgram, String> {
    compile_program_with_identity_from_source(source_name, source).map(|(program, _)| program)
}

fn compile_program_with_identity_from_source(
    source_name: &str,
    source: &str,
) -> Result<(KernelProgram, SourceIdentity), String> {
    let mut compiled = compile(source_name, source)
        .map_err(|diagnostics| format!("canonical source did not compile: {diagnostics:?}"))?;
    if compiled.len() != 1 {
        return Err(format!(
            "canonical source produced {} transactions, expected one",
            compiled.len()
        ));
    }
    let (transaction, model, symbols) = compiled.remove(0).into_parts();
    let identity = SourceIdentity {
        model_ulid: model.ulid().to_string(),
        symbols: symbols
            .iter()
            .map(|(name, id)| {
                Ok(SourceSymbol {
                    name: name.to_owned(),
                    kind: semantic_kind_name(id.kind())?,
                    ulid: id.ulid().to_string(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    let mut store = InMemoryGraphStore::new();
    store
        .commit(transaction)
        .map_err(|diagnostics| format!("canonical transaction did not commit: {diagnostics:?}"))?;
    KernelProgram::from_snapshot(&store.snapshot(), model)
        .map(|program| (program, identity))
        .map_err(|diagnostics| format!("canonical program validation failed: {diagnostics:?}"))
}

pub fn cuda_solver_contract() -> SolverCapabilities {
    SolverCapabilities::exact([
        SolverCapability {
            algorithm: LinearSolver::ConjugateGradient,
            operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        },
        SolverCapability {
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        },
        SolverCapability {
            algorithm: LinearSolver::MinimumResidual,
            operator_properties: LinearOperatorProperties::SymmetricIndefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Fast,
            scalar_type: ScalarType::F64,
        },
    ])
    .expect("the fixed evidence capability set is nonempty")
}

pub fn exact_capabilities(
    solver: SolverCapabilities,
    targets: TargetCapabilities,
) -> RealizationCapabilities {
    let plan = solver_plan(ReductionPolicy::Fast);
    solver
        .require_problem(
            plan,
            ScalarType::F64,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .expect("the provider implements the exact Poisson solver tuple");
    let solver = SolverCapabilities::exact([SolverCapability {
        algorithm: plan.algorithm(),
        operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
        preconditioner: plan.preconditioner(),
        reduction: plan.reduction(),
        scalar_type: ScalarType::F64,
    }])
    .expect("the selected Poisson solver tuple is exact");
    RealizationCapabilities::cartesian_product(
        [
            DiscretizationMethod::ContinuousGalerkin,
            DiscretizationMethod::CellCenteredFiniteVolume,
        ],
        [(
            MeshKind::GeneratedCartesian,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is nonzero")),
        )],
        [VectorLayoutKind::Replicated],
        solver,
        targets,
    )
    .expect("the fixed evidence capability intersection is valid")
}

pub fn request(
    program: &KernelProgram,
    method: DiscretizationMethod,
    device: u16,
    revision: u64,
) -> Result<RealizationRequest, String> {
    let (space, quadrature) = method_policy(method)?;
    let plan = RealizationPlan::new(
        space,
        Discretization::new(
            method,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(CELLS_PER_AXIS)
                    .expect("the fixed cell count is nonzero"),
            },
            quadrature,
        ),
        solver_plan(ReductionPolicy::Fast),
        Target::CudaGpu { device },
        ExecutionSchedule::Offline,
    )
    .map_err(|diagnostic| diagnostic.to_string())?;
    Ok(RealizationRequest::explicit(
        program.model(),
        SemanticRevision::new(program.revision().0),
        RealizationRevision::new(revision),
        plan,
    ))
}

pub fn host_request(
    program: &KernelProgram,
    method: DiscretizationMethod,
    revision: u64,
) -> Result<RealizationRequest, String> {
    let (space, quadrature) = method_policy(method)?;
    let plan = RealizationPlan::new(
        space,
        Discretization::new(
            method,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(CELLS_PER_AXIS)
                    .expect("the fixed cell count is nonzero"),
            },
            quadrature,
        ),
        solver_plan(ReductionPolicy::Reproducible),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .map_err(|diagnostic| diagnostic.to_string())?;
    Ok(RealizationRequest::explicit(
        program.model(),
        SemanticRevision::new(program.revision().0),
        RealizationRevision::new(revision),
        plan,
    ))
}

pub fn solver_plan(reduction: ReductionPolicy) -> SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(4096).expect("the fixed iteration limit is nonzero"),
    )
    .expect("the fixed evidence solver plan is valid")
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(reduction)
}

pub fn requirements() -> RealizationRequirements {
    RealizationRequirements::new(
        NonZeroUsize::new(2).expect("two is nonzero"),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    )
}

pub fn reference_cpu_solution(
    program: &KernelProgram,
    method: DiscretizationMethod,
    revision: u64,
) -> Result<ResolvedScalarEllipticCartesianSolution, String> {
    let request = host_request(program, method, revision)?;
    let resolved = eqiora::realization::resolve(
        &request,
        requirements(),
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .map_err(|diagnostic| diagnostic.to_string())?;
    solve_resolved_scalar_elliptic_cartesian(program, &resolved, &REFERENCE_LINEAR_SOLVER)
        .map(|(_, solution)| solution)
        .map_err(|diagnostic| diagnostic.to_string())
}

pub fn algebraic_values(solution: &ResolvedScalarEllipticCartesianSolution) -> &[f64] {
    match solution {
        ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) => {
            solution.algebraic_values()
        }
        ResolvedScalarEllipticCartesianSolution::FiniteVolume(solution) => solution.cell_values(),
    }
}

pub fn method_metrics(
    method: DiscretizationMethod,
    solution: &ResolvedScalarEllipticCartesianSolution,
) -> Result<MethodMetrics, String> {
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 4)
        .map_err(|diagnostic| diagnostic.to_string())?;
    let exact = |coordinate: &[f64]| (PI * coordinate[0]).sin() * (PI * coordinate[1]).sin();
    let (l2_error, boundary_quantity, integrated_source) = match (method, solution) {
        (
            DiscretizationMethod::ContinuousGalerkin,
            ResolvedScalarEllipticCartesianSolution::FiniteElement(solution),
        ) => (
            solution
                .field()
                .l2_error(&exact, &quadrature)
                .map_err(|diagnostic| diagnostic.to_string())?,
            solution.boundary_reaction_sum(),
            solution.integrated_source(),
        ),
        (
            DiscretizationMethod::CellCenteredFiniteVolume,
            ResolvedScalarEllipticCartesianSolution::FiniteVolume(solution),
        ) => (
            solution
                .reconstruction()
                .l2_error(&exact, &quadrature)
                .map_err(|diagnostic| diagnostic.to_string())?,
            solution.boundary_flux_sum(),
            solution.integrated_source(),
        ),
        _ => return Err("resolved method and reconstructed solution differ".to_owned()),
    };
    let relative_balance = (boundary_quantity + integrated_source).abs()
        / (boundary_quantity.abs() + integrated_source.abs()).max(f64::MIN_POSITIVE);
    for (label, value) in [
        ("L2 error", l2_error),
        ("boundary quantity", boundary_quantity),
        ("integrated source", integrated_source),
        ("relative balance", relative_balance),
    ] {
        if !value.is_finite() {
            return Err(format!("{label} is non-finite"));
        }
    }
    if l2_error >= MAXIMUM_L2_ERROR {
        return Err(format!(
            "{method:?} L2 error {l2_error:e} is not below {MAXIMUM_L2_ERROR:e}"
        ));
    }
    if relative_balance >= MAXIMUM_RELATIVE_BALANCE {
        return Err(format!(
            "{method:?} relative balance {relative_balance:e} is not below {MAXIMUM_RELATIVE_BALANCE:e}"
        ));
    }
    Ok(MethodMetrics {
        l2_error,
        boundary_quantity,
        integrated_source,
        relative_balance,
    })
}

pub fn cpu_conformance(
    reference: &ResolvedScalarEllipticCartesianSolution,
    candidate: &ResolvedScalarEllipticCartesianSolution,
) -> Result<CpuConformance, String> {
    reference_conformance(
        reference,
        candidate,
        CPU_CUDA_ABSOLUTE,
        CPU_CUDA_RELATIVE,
        "CPU/CUDA",
    )
}

#[allow(dead_code)] // Feature-gated backends share the same explicit comparison contract.
pub fn reference_conformance(
    reference: &ResolvedScalarEllipticCartesianSolution,
    candidate: &ResolvedScalarEllipticCartesianSolution,
    absolute: f64,
    relative: f64,
    comparison: &str,
) -> Result<CpuConformance, String> {
    if !absolute.is_finite() || absolute < 0.0 || !relative.is_finite() || relative < 0.0 {
        return Err(format!(
            "{comparison} tolerances must be finite and non-negative"
        ));
    }
    let reference_values = algebraic_values(reference);
    let candidate_values = algebraic_values(candidate);
    if reference_values.len() != candidate_values.len() {
        return Err(format!(
            "{comparison} value counts differ: {} and {}",
            reference_values.len(),
            candidate_values.len()
        ));
    }
    let scalar_pairs = match (reference, candidate) {
        (
            ResolvedScalarEllipticCartesianSolution::FiniteElement(reference),
            ResolvedScalarEllipticCartesianSolution::FiniteElement(candidate),
        ) => [
            (
                reference.boundary_reaction_sum(),
                candidate.boundary_reaction_sum(),
            ),
            (reference.integrated_source(), candidate.integrated_source()),
        ],
        (
            ResolvedScalarEllipticCartesianSolution::FiniteVolume(reference),
            ResolvedScalarEllipticCartesianSolution::FiniteVolume(candidate),
        ) => [
            (reference.boundary_flux_sum(), candidate.boundary_flux_sum()),
            (reference.integrated_source(), candidate.integrated_source()),
        ],
        _ => return Err(format!("{comparison} methods differ")),
    };
    let mut maximum_absolute_error = 0.0_f64;
    let mut maximum_scaled_error = 0.0_f64;
    for (index, (reference, candidate)) in reference_values
        .iter()
        .copied()
        .zip(candidate_values.iter().copied())
        .chain(scalar_pairs)
        .enumerate()
    {
        if !reference.is_finite() || !candidate.is_finite() {
            return Err(format!("{comparison} value {index} is non-finite"));
        }
        let error = (candidate - reference).abs();
        let tolerance = absolute + relative * reference.abs();
        if error > tolerance {
            return Err(format!(
                "{comparison} value {index} differs: reference {reference:e}, candidate {candidate:e}, tolerance {tolerance:e}"
            ));
        }
        maximum_absolute_error = maximum_absolute_error.max(error);
        maximum_scaled_error = maximum_scaled_error.max(error / tolerance);
    }
    Ok(CpuConformance {
        maximum_absolute_error,
        maximum_scaled_error,
    })
}

#[allow(dead_code)] // The collector and host replay compile this shared support independently.
pub fn method_from_tag(tag: &str) -> Option<(u64, DiscretizationMethod)> {
    METHODS
        .iter()
        .find(|(_, _, expected)| *expected == tag)
        .map(|(revision, method, _)| (*revision, *method))
}

fn method_policy(method: DiscretizationMethod) -> Result<(Space, QuadraturePolicy), String> {
    match method {
        DiscretizationMethod::ContinuousGalerkin => Ok((
            Space::continuous_lagrange(NonZeroU16::MIN),
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).expect("two is nonzero"),
            },
        )),
        DiscretizationMethod::CellCenteredFiniteVolume => {
            Ok((Space::cell_constant(), QuadraturePolicy::CellCentroid))
        }
    }
}

fn semantic_kind_name(kind: EntityKind) -> Result<&'static str, String> {
    match kind {
        EntityKind::Domain => Ok("domain"),
        EntityKind::Representation => Ok("representation"),
        EntityKind::Field => Ok("field"),
        EntityKind::Parameter => Ok("parameter"),
        EntityKind::Port => Ok("port"),
        EntityKind::Relation => Ok("relation"),
        EntityKind::Activation => Ok("activation"),
        EntityKind::Connection => Ok("connection"),
        EntityKind::ClockDomain => Ok("clock-domain"),
        _ => Err(format!(
            "source compiler returned non-semantic symbol kind {kind:?}"
        )),
    }
}
