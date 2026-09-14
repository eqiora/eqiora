use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
use std::collections::BTreeMap;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_assembly::{AssemblyBackend, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, ScalarType};
use eqiora_meshing::{
    MeshEntity, MeshTopology, SimplicialMesh, simplex_centroid_rule, triangle_duffy_gauss_legendre,
};
use eqiora_realization::{
    AlgebraicBlockScale, Discretization, DiscretizationMethod, ExecutionSchedule,
    FieldSpaceBinding, FieldwiseRealizationPlan, FieldwiseRealizationRequirements,
    FieldwiseSpatialDiscretization, MeshArtifactReference, MeshPolicy, PlacementRequirementNode,
    PortableRealizationGraph, PositivePhysicalScale, QuadraturePolicy, RealizationRequirements,
    ResolvedFieldwiseRealization, SolveRoot, Space, SymmetricCongruenceScaling, Target,
    VectorLayoutKind,
};
use eqiora_schema::kernel::BoundarySide;
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, LinearSolverBackend, PreconditionerPolicy,
    ReductionPolicy, SolverPlan,
};

use super::{
    FinalizedSteadyStokesMini2dProblem, SteadyIncompressibleStokesCartesianModel2d,
    SteadyStokesMiniSolution2d, lower_steady_incompressible_stokes_cartesian_2d,
};
use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};
use crate::canonical_stokes::api::{SteadyIncompressibleStokesModel2d, StokesBoundaryKey2d};
use crate::simplicial_stokes::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
    SimplicialMiniStokesBoundaryFacet2d,
    finalize_simplicial_mini_stokes_2d_with_boundary_and_assembly,
};

const DIMENSION: usize = 2;
const DUFFY_POINTS_PER_AXIS: usize = 3;

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

/// Three native characteristic quantities for coherent-SI incompressible flow.
///
/// Gauge and weak-functional scales are derived exactly as `G = U / L` and
/// `Theta = P U L`; callers cannot supply either value independently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncompressibleFlowScaleProfile2d {
    length: PositivePhysicalScale,
    velocity: PositivePhysicalScale,
    pressure: PositivePhysicalScale,
    gauge: PositivePhysicalScale,
    weak_functional: PositivePhysicalScale,
}

impl IncompressibleFlowScaleProfile2d {
    /// Validate positive finite `L`, `U`, and `P` and derive `G` and `Theta`.
    ///
    /// # Errors
    /// Returns `EQ0807` for a non-positive, non-finite, or dimensionally
    /// incompatible quantity, including overflow in a derived scale.
    pub fn new(
        length: DynQuantity,
        velocity: DynQuantity,
        pressure: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        require_dimension(length, LENGTH, "Stokes length scale L")?;
        require_dimension(velocity, VELOCITY, "Stokes velocity scale U")?;
        require_dimension(pressure, PRESSURE, "Stokes pressure scale P")?;
        let length = PositivePhysicalScale::new(length).map_err(realization_error)?;
        let velocity = PositivePhysicalScale::new(velocity).map_err(realization_error)?;
        let pressure = PositivePhysicalScale::new(pressure).map_err(realization_error)?;
        let gauge = PositivePhysicalScale::new(velocity.quantity().try_div(length.quantity())?)
            .map_err(realization_error)?;
        let weak_functional = PositivePhysicalScale::new(
            pressure
                .quantity()
                .try_mul(velocity.quantity())?
                .try_mul(length.quantity())?,
        )
        .map_err(realization_error)?;
        Ok(Self {
            length,
            velocity,
            pressure,
            gauge,
            weak_functional,
        })
    }

    /// Characteristic physical length `L`.
    #[must_use]
    pub const fn length(self) -> DynQuantity {
        self.length.quantity()
    }

    /// Characteristic physical velocity `U`.
    #[must_use]
    pub const fn velocity(self) -> DynQuantity {
        self.velocity.quantity()
    }

    /// Characteristic physical pressure `P`.
    #[must_use]
    pub const fn pressure(self) -> DynQuantity {
        self.pressure.quantity()
    }

    /// Derived gauge-multiplier scale `G = U / L`.
    #[must_use]
    pub const fn gauge(self) -> DynQuantity {
        self.gauge.quantity()
    }

    /// Derived intrinsic-2D weak-functional scale `Theta = P U L`.
    #[must_use]
    pub const fn weak_functional(self) -> DynQuantity {
        self.weak_functional.quantity()
    }

    pub(super) const fn length_value(self) -> f64 {
        self.length.quantity().value()
    }

    pub(super) const fn velocity_value(self) -> f64 {
        self.velocity.quantity().value()
    }

    pub(super) const fn pressure_value(self) -> f64 {
        self.pressure.quantity().value()
    }

    pub(super) const fn gauge_value(self) -> f64 {
        self.gauge.quantity().value()
    }
}

/// Compatibility name for the existing steady-Stokes realization surface.
pub type SteadyStokesScaleProfile2d = IncompressibleFlowScaleProfile2d;

/// Exact lowerer requirements for the admitted mixed Stokes path.
#[must_use]
pub fn steady_stokes_fieldwise_requirements_2d(
    model: &SteadyIncompressibleStokesCartesianModel2d,
) -> FieldwiseRealizationRequirements {
    steady_stokes_fieldwise_requirements_for_model_2d(model.common())
}

pub(super) fn steady_stokes_fieldwise_requirements_for_model_2d(
    model: &SteadyIncompressibleStokesModel2d,
) -> FieldwiseRealizationRequirements {
    FieldwiseRealizationRequirements::new(
        domain_id(model),
        [velocity_id(model), pressure_id(model)],
        RealizationRequirements::new(
            NonZeroUsize::new(DIMENSION).expect("two is non-zero"),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
    )
    .expect("a lowered Stokes model owns two distinct algebraic Fields")
}

/// Build the sole admitted field-wise MINI plan from exact canonical roles.
///
/// `q` remains immutable canonical coefficient data and receives no discrete
/// unknown block. The plan derives gauge and functional scales from `L/U/P`,
/// fixes positive degree-four Duffy assembly, and admits only the exact
/// reference-MINRES and host-sparse-LU solver tuples verified for this path,
/// without a fluid-named generic space tag.
///
/// # Errors
/// Returns `EQ0807` for an unsupported solver tuple or invalid derived plan.
pub fn steady_stokes_mini_plan_2d(
    model: &SteadyIncompressibleStokesCartesianModel2d,
    mesh: MeshArtifactReference,
    scales: SteadyStokesScaleProfile2d,
    solver: SolverPlan,
) -> Result<FieldwiseRealizationPlan, Diagnostic> {
    steady_stokes_mini_plan_for_model_2d(model.common(), mesh, scales, solver)
}

mod plan;
pub(super) use plan::steady_stokes_algebraic_structure;
pub(super) use plan::steady_stokes_mini_plan_for_model_2d;

/// Finalize one resolved coherent-SI Stokes Model through reference assembly.
///
/// `mesh_artifact` authenticates the reference retained by `resolved`; a bare
/// [`SimplicialMesh`] has no content digest. Artifact-backed callers must first
/// validate the Realization against a mesh envelope, then pass the reference
/// and mesh reconstructed from that same accepted envelope.
///
/// # Errors
/// Rejects Model/Field/mesh/space/constraint/scale/solver drift before local
/// assembly and preserves numerical diagnostics from the dimensionless path.
pub fn finalize_resolved_steady_stokes_mini_2d(
    program: &KernelProgram,
    resolved: &ResolvedFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
) -> Result<
    (
        SteadyIncompressibleStokesCartesianModel2d,
        FinalizedSteadyStokesMini2dProblem,
    ),
    Diagnostic,
> {
    finalize_resolved_steady_stokes_mini_2d_with_assembly(
        program,
        resolved,
        mesh_artifact,
        mesh,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

/// Finalize one resolved coherent-SI Stokes Model through explicit assembly.
///
/// The production local operator is assembled directly on normalized
/// coordinates with dimensionless viscosity and force-potential gradient. No
/// raw matrix with heterogeneous SI row dimensions is materialized.
/// `mesh_artifact` proves reference agreement, while the caller retains
/// responsibility for obtaining `mesh` from that exact validated artifact.
///
/// # Errors
/// Preserves all reference finalization failures plus the selected assembly
/// adapter's complete-operation diagnostic.
fn finalize_resolved_steady_stokes_mini_2d_with_assembly(
    program: &KernelProgram,
    resolved: &ResolvedFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
    assembly: &dyn AssemblyBackend,
) -> Result<
    (
        SteadyIncompressibleStokesCartesianModel2d,
        FinalizedSteadyStokesMini2dProblem,
    ),
    Diagnostic,
> {
    let model = lower_steady_incompressible_stokes_cartesian_2d(program)?;
    let scales = resolved_stokes_scales(program, resolved, mesh_artifact, model.common())?;
    let normalized = normalize_mesh(&model, mesh, scales.length_value())?;
    let boundary = numerical_boundary(model.common(), &normalized, scales.pressure_value())?;
    let essential_velocity =
        |coordinate_hat| cartesian_essential_velocity(model.common(), scales, coordinate_hat);
    let finalized = finalize_lowered_steady_stokes_mini_2d_with_assembly(
        resolved,
        mesh_artifact,
        model.common(),
        mesh,
        &normalized.mesh,
        &boundary,
        scales,
        &essential_velocity,
        assembly,
    )?;
    Ok((model, finalized))
}

pub(super) fn resolved_stokes_scales(
    program: &KernelProgram,
    resolved: &ResolvedFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    model: &SteadyIncompressibleStokesModel2d,
) -> Result<SteadyStokesScaleProfile2d, Diagnostic> {
    if program.model() != resolved.model()
        || program.revision().0 != resolved.semantic_revision().get()
    {
        return Err(invalid_realization(
            "resolved field-wise realization does not reference this exact Semantic Model revision",
        ));
    }
    requires_zero_integral_constraint(model)?;
    let realization_graph = resolved.portable_graph()?;
    require_exact_plan(model, resolved, &realization_graph, mesh_artifact)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_lowered_steady_stokes_mini_2d_with_assembly<B>(
    resolved: &ResolvedFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    model: &SteadyIncompressibleStokesModel2d,
    physical_mesh: &SimplicialMesh,
    normalized_mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    scales: SteadyStokesScaleProfile2d,
    essential_velocity: &B,
    assembly: &dyn AssemblyBackend,
) -> Result<FinalizedSteadyStokesMini2dProblem, Diagnostic>
where
    B: Fn([f64; DIMENSION]) -> Result<[f64; DIMENSION], Diagnostic> + Sync,
{
    let realization_graph = resolved.portable_graph()?;
    let quadrature = triangle_duffy_gauss_legendre(DUFFY_POINTS_PER_AXIS)?;
    let facet_quadrature = simplex_centroid_rule(DIMENSION - 1)?;
    let dimensionless_viscosity = model.dynamic_viscosity() * scales.velocity_value()
        / (scales.pressure_value() * scales.length_value());
    if !dimensionless_viscosity.is_finite() || dimensionless_viscosity <= 0.0 {
        return Err(invalid_realization(
            "derived dimensionless Stokes viscosity is not finite and positive",
        ));
    }
    let lower = [model.bounds()[0][0], model.bounds()[1][0]];
    let parameter_tangent = vec![0.0; model.force_potential_expression().parameter_fields().len()];
    let force_potential = model.force_potential_expression();
    let dimensionless_force = |coordinate_hat: [f64; DIMENSION]| {
        let coordinate = [
            lower[0] + scales.length_value() * coordinate_hat[0],
            lower[1] + scales.length_value() * coordinate_hat[1],
        ];
        let mut result = [0.0; DIMENSION];
        for axis in 0..DIMENSION {
            let mut coordinate_tangent = [0.0; DIMENSION];
            coordinate_tangent[axis] = scales.length_value();
            let (_, derivative_hat) = force_potential.evaluate_jvp(
                &coordinate,
                &coordinate_tangent,
                &parameter_tangent,
            )?;
            result[axis] = derivative_hat / scales.pressure_value();
        }
        Ok(result)
    };
    let block_system = super::block::steady_stokes_block_system(
        model,
        resolved,
        mesh_artifact,
        normalized_mesh,
        boundary,
        scales,
    )?;
    let checked_assembly = block_system.checked_backend(assembly);
    let SolveRoot::Linear(root) = realization_graph.root() else {
        return Err(invalid_realization(
            "steady Stokes requires a linear portable Realization root",
        ));
    };
    let linear = realization_graph
        .linear_solve(root)
        .ok_or_else(|| invalid_realization("steady Stokes portable linear root is absent"))?;
    let inner = finalize_simplicial_mini_stokes_2d_with_boundary_and_assembly(
        normalized_mesh,
        dimensionless_viscosity,
        &dimensionless_force,
        boundary,
        essential_velocity,
        &quadrature,
        &facet_quadrature,
        &checked_assembly,
        linear.plan(),
        realization_graph.systems()[0].partition(),
        resolved.plan().target(),
    )?;
    let inner = inner.with_block_system(&block_system)?;
    let finalized = FinalizedSteadyStokesMini2dProblem::new(
        inner,
        physical_mesh.clone(),
        velocity_id(model),
        pressure_id(model),
        force_potential_id(model),
        scales,
    );
    Ok(finalized)
}

/// Admit callback transport only after exact agreement with complete Model replay.
pub(super) fn validate_model_owned_essential_transport<B>(
    normalized_mesh: &SimplicialMesh,
    replayed: &[Option<[f64; DIMENSION]>],
    transport: &B,
) -> Result<BTreeMap<(u64, u64), [f64; DIMENSION]>, Diagnostic>
where
    B: Fn([f64; DIMENSION]) -> Result<[f64; DIMENSION], Diagnostic> + Sync,
{
    if replayed.len() != normalized_mesh.vertices().len() {
        return Err(invalid_realization(
            "Model-owned essential replay does not cover the exact mesh vertex inventory",
        ));
    }
    let mut lookup = BTreeMap::new();
    for (coordinate, expected) in normalized_mesh.vertices().iter().zip(replayed) {
        let Some(expected) = expected else {
            continue;
        };
        let [x, y] = coordinate.as_slice() else {
            return Err(invalid_realization(
                "Model-owned essential replay requires intrinsic 2D coordinates",
            ));
        };
        let coordinate = [*x, *y];
        let carried = transport(coordinate)?;
        if carried != *expected {
            return Err(invalid_realization(
                "essential callback value differs from the exhaustive Model-owned replay",
            ));
        }
        if lookup.insert((x.to_bits(), y.to_bits()), carried).is_some() {
            return Err(invalid_realization(
                "essential callback coordinates are not unique in the accepted mesh",
            ));
        }
    }
    if lookup.is_empty() {
        return Err(invalid_realization(
            "Model-owned essential replay contains no accepted vertex",
        ));
    }
    Ok(lookup)
}

/// Solve one resolved coherent-SI Stokes Model through reference assembly.
///
/// # Errors
/// Preserves exact finalization, backend, residual, and reconstruction diagnostics.
pub fn solve_resolved_steady_stokes_mini_2d(
    program: &KernelProgram,
    resolved: &ResolvedFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
    backend: &dyn LinearSolverBackend,
) -> Result<
    (
        SteadyIncompressibleStokesCartesianModel2d,
        SteadyStokesMiniSolution2d,
    ),
    Diagnostic,
> {
    solve_resolved_steady_stokes_mini_2d_with_assembly(
        program,
        resolved,
        mesh_artifact,
        mesh,
        &REFERENCE_ASSEMBLY_BACKEND,
        backend,
    )
}

/// Solve through independently selected assembly and linear execution adapters.
///
/// # Errors
/// Preserves every typed admission, adapter, and physical reconstruction failure.
fn solve_resolved_steady_stokes_mini_2d_with_assembly(
    program: &KernelProgram,
    resolved: &ResolvedFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
    assembly: &dyn AssemblyBackend,
    backend: &dyn LinearSolverBackend,
) -> Result<
    (
        SteadyIncompressibleStokesCartesianModel2d,
        SteadyStokesMiniSolution2d,
    ),
    Diagnostic,
> {
    let (model, finalized) = finalize_resolved_steady_stokes_mini_2d_with_assembly(
        program,
        resolved,
        mesh_artifact,
        mesh,
        assembly,
    )?;
    let solved = backend.solve(&finalized.linear_problem()?, finalized.solver_plan())?;
    Ok((model, finalized.finish(solved)?))
}

fn require_exact_plan(
    model: &SteadyIncompressibleStokesModel2d,
    resolved: &ResolvedFieldwiseRealization,
    graph: &PortableRealizationGraph,
    mesh_artifact: MeshArtifactReference,
) -> Result<SteadyStokesScaleProfile2d, Diagnostic> {
    let expected_requirements = steady_stokes_fieldwise_requirements_for_model_2d(model);
    if resolved.requirements() != &expected_requirements {
        return Err(invalid_realization(
            "field-wise requirements differ from the exact Stokes Domain and unknown-Field inventory",
        ));
    }
    let SolveRoot::Linear(root) = graph.root() else {
        return Err(invalid_realization(
            "steady Stokes portable graph requires one linear solve root",
        ));
    };
    let linear = graph
        .linear_solve(root)
        .ok_or_else(|| invalid_realization("steady Stokes graph linear root is absent"))?;
    if graph.lineage().model() != resolved.model()
        || graph.lineage().semantic_revision() != resolved.semantic_revision()
        || graph.domains().len() != 1
        || graph.domains()[0].domain() != domain_id(model)
        || graph.fields().len() != 2
        || graph.systems().len() != 1
        || graph.placement(linear.placement())
            != Some(PlacementRequirementNode::HostWorkers {
                workers_per_partition: NonZeroUsize::MIN,
            })
    {
        return Err(invalid_realization(
            "steady Stokes portable graph lineage, exact Field inventory, or placement drifted",
        ));
    }
    let plan = resolved.plan();
    let velocity = velocity_id(model);
    let pressure = pressure_id(model);
    let scale_for = |block| {
        graph.systems()[0]
            .congruence_scaling()
            .ok_or_else(|| invalid_realization("steady Stokes graph requires congruence scaling"))?
            .block_scales()
            .iter()
            .find(|entry| entry.block() == block)
            .map(|entry| entry.scale().quantity())
            .ok_or_else(|| {
                invalid_realization("Stokes plan is missing an exact algebraic block scale")
            })
    };
    let scales = SteadyStokesScaleProfile2d::new(
        plan.spatial().coordinate_length_scale().quantity(),
        scale_for(AlgebraicBlock::Field(velocity))?,
        scale_for(AlgebraicBlock::Field(pressure))?,
    )?;
    let expected =
        steady_stokes_mini_plan_for_model_2d(model, mesh_artifact, scales, plan.solver())?;
    if plan != &expected {
        return Err(invalid_realization(
            "field-wise plan differs from the exact coherent-SI MINI Stokes contract",
        ));
    }
    Ok(scales)
}

pub(super) struct NormalizedCartesianSimplicialMesh2d {
    pub(super) mesh: SimplicialMesh,
    pub(super) boundary_facets: Vec<(MeshEntity, usize, BoundarySide)>,
}

fn normalize_mesh(
    model: &SteadyIncompressibleStokesCartesianModel2d,
    mesh: &SimplicialMesh,
    length: f64,
) -> Result<NormalizedCartesianSimplicialMesh2d, Diagnostic> {
    normalize_cartesian_mesh(model.bounds(), mesh, length, "Stokes")
}

pub(super) fn normalize_cartesian_mesh(
    bounds: &[[f64; 2]; DIMENSION],
    mesh: &SimplicialMesh,
    length: f64,
    equation_family: &str,
) -> Result<NormalizedCartesianSimplicialMesh2d, Diagnostic> {
    if mesh.topological_dimension() != DIMENSION {
        return Err(invalid_realization(format!(
            "coherent-SI MINI {equation_family} requires an intrinsic 2D mesh"
        )));
    }
    let mut side_coverage = [[false; 2]; DIMENSION];
    for (vertex, coordinates) in mesh.vertices().iter().enumerate() {
        if coordinates.len() != DIMENSION
            || coordinates
                .iter()
                .enumerate()
                .any(|(axis, value)| *value < bounds[axis][0] || *value > bounds[axis][1])
        {
            return Err(invalid_realization(format!(
                "imported {equation_family} mesh has a vertex outside the exact Cartesian Domain"
            )));
        }
        let on_box = (0..DIMENSION).any(|axis| {
            coordinates[axis] == bounds[axis][0] || coordinates[axis] == bounds[axis][1]
        });
        let on_mesh_boundary = mesh
            .is_boundary_entity(MeshEntity::new(0, vertex))
            .expect("mesh owns every vertex");
        if on_box != on_mesh_boundary {
            return Err(invalid_realization(format!(
                "imported {equation_family} mesh boundary vertices do not coincide exactly with the Cartesian Domain boundary"
            )));
        }
    }
    let facet_count = mesh
        .entity_count(DIMENSION - 1)
        .expect("2D mesh owns edge entities");
    let mut boundary_facets = Vec::new();
    for facet in 0..facet_count {
        let facet = MeshEntity::new(DIMENSION - 1, facet);
        if !mesh
            .is_boundary_entity(facet)
            .expect("mesh owns every edge")
        {
            continue;
        }
        let vertices = mesh
            .entity_vertices(facet)
            .expect("accepted boundary edge owns vertices");
        let mut matched = None;
        for (axis, axis_bounds) in bounds.iter().enumerate() {
            for (side, bound) in axis_bounds.iter().enumerate() {
                if vertices
                    .iter()
                    .all(|vertex| mesh.vertices()[vertex.index()][axis] == *bound)
                {
                    if matched.is_some() {
                        return Err(invalid_realization(format!(
                            "a {equation_family} boundary facet ambiguously belongs to multiple Cartesian sides"
                        )));
                    }
                    matched = Some((axis, side));
                }
            }
        }
        let Some((axis, side)) = matched else {
            return Err(invalid_realization(format!(
                "a {equation_family} mesh boundary facet does not lie on an exact Cartesian side"
            )));
        };
        side_coverage[axis][side] = true;
        boundary_facets.push((
            facet,
            axis,
            if side == 0 {
                BoundarySide::Lower
            } else {
                BoundarySide::Upper
            },
        ));
    }
    if side_coverage.iter().flatten().any(|covered| !covered) {
        return Err(invalid_realization(format!(
            "imported {equation_family} mesh does not cover every exact Cartesian boundary side"
        )));
    }
    let vertices = mesh
        .vertices()
        .iter()
        .map(|coordinates| {
            (0..DIMENSION)
                .map(|axis| (coordinates[axis] - bounds[axis][0]) / length)
                .collect()
        })
        .collect();
    let mesh = SimplicialMesh::new(
        DIMENSION,
        vertices,
        mesh.cells().to_vec(),
        mesh.quality_gate(),
    )
    .map_err(|error| invalid_realization(error.message()))?;
    Ok(NormalizedCartesianSimplicialMesh2d {
        mesh,
        boundary_facets,
    })
}

fn numerical_boundary(
    model: &SteadyIncompressibleStokesModel2d,
    normalized: &NormalizedCartesianSimplicialMesh2d,
    pressure_scale: f64,
) -> Result<SimplicialMiniStokesBoundary2d, Diagnostic> {
    let facets = normalized
        .boundary_facets
        .iter()
        .map(|(facet, axis, side)| {
            let key = StokesBoundaryKey2d::CartesianSide {
                axis: *axis,
                side: *side,
            };
            let entry = model
                .boundary_entry(&key)
                .ok_or_else(|| {
                    invalid_realization(format!(
                        "lowered 2D Stokes boundary inventory omits axis {axis} {side:?}"
                    ))
                })?;
            let condition = match entry.disposition {
                PhysicalBoundaryDisposition::TraceZero => {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                }
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace =>
                {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                }
                PhysicalBoundaryDisposition::FluxZero => {
                    let pressure = model
                        .normal_pressure_for(&key)
                        .ok_or_else(|| {
                            invalid_realization(format!(
                                "axis {axis} {side:?} is not an admitted normal-pressure boundary"
                            ))
                        })?
                        .expression()
                        .constant_value()
                        .ok_or_else(|| {
                            invalid_realization(format!(
                                "axis {axis} {side:?} normal pressure is coordinate-dependent; this realization admits only a spatial constant"
                            ))
                        })?;
                    let mut traction = [0.0; DIMENSION];
                    traction[*axis] = match side {
                        BoundarySide::Lower => pressure / pressure_scale,
                        BoundarySide::Upper => -pressure / pressure_scale,
                    };
                    SimplicialMiniStokesBoundaryCondition2d::ConstantTraction {
                        value: traction,
                    }
                }
                PhysicalBoundaryDisposition::Prescribed(law) => {
                    let pressure = model
                        .normal_pressure_for(&key)
                        .ok_or_else(|| {
                            invalid_realization(format!(
                                "prescribed Stokes {:?} law {} on axis {axis} {side:?} is outside the normal-pressure realization",
                                law.quantity(),
                                law.relation()
                            ))
                        })?
                        .expression()
                        .constant_value()
                        .ok_or_else(|| {
                            invalid_realization(format!(
                                "axis {axis} {side:?} normal pressure is coordinate-dependent; this realization admits only a spatial constant"
                            ))
                        })?;
                    let mut traction = [0.0; DIMENSION];
                    traction[*axis] = match side {
                        BoundarySide::Lower => pressure / pressure_scale,
                        BoundarySide::Upper => -pressure / pressure_scale,
                    };
                    SimplicialMiniStokesBoundaryCondition2d::ConstantTraction {
                        value: traction,
                    }
                }
                PhysicalBoundaryDisposition::PortBinding { connection, port } => {
                    return Err(invalid_realization(format!(
                        "live Stokes PortBinding {connection} through Port {port} on axis {axis} {side:?} requires an explicit trace-space interface Realization"
                    )));
                }
            };
            Ok(SimplicialMiniStokesBoundaryFacet2d::new(*facet, condition))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    SimplicialMiniStokesBoundary2d::new(&normalized.mesh, facets)
        .map_err(|error| invalid_realization(error.message()))
}

fn cartesian_essential_velocity(
    model: &SteadyIncompressibleStokesModel2d,
    scales: SteadyStokesScaleProfile2d,
    coordinate_hat: [f64; DIMENSION],
) -> Result<[f64; DIMENSION], Diagnostic> {
    let lower = [model.bounds()[0][0], model.bounds()[1][0]];
    let length = scales.length_value();
    let physical = [
        lower[0] + length * coordinate_hat[0],
        lower[1] + length * coordinate_hat[1],
    ];
    let mut selected = None;
    for axis in 0..DIMENSION {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let side_index = usize::from(side == BoundarySide::Upper);
            let normalized_bound = (model.bounds()[axis][side_index] - lower[axis]) / length;
            if coordinate_hat[axis] != normalized_bound {
                continue;
            }
            let key = StokesBoundaryKey2d::CartesianSide { axis, side };
            let disposition = model
                .boundary_entry(&key)
                .expect("lowered Cartesian Stokes model owns every exact side")
                .disposition;
            let value = match disposition {
                PhysicalBoundaryDisposition::TraceZero => Some([0.0; DIMENSION]),
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace =>
                {
                    let mut outward = [0.0; DIMENSION];
                    outward[axis] = if side == BoundarySide::Lower {
                        -1.0
                    } else {
                        1.0
                    };
                    Some(
                        model
                            .prescribed_velocity(&key, Some(outward), &physical)?
                            .ok_or_else(|| {
                                invalid_realization(format!(
                                    "prescribed velocity Relation {} has no retained trace expression",
                                    law.relation()
                                ))
                            })?,
                    )
                }
                _ => None,
            };
            let Some(value) = value else {
                continue;
            };
            if selected.is_some_and(|existing| existing != value) {
                return Err(invalid_realization(
                    "essential velocity prescriptions disagree at a shared Cartesian corner",
                ));
            }
            selected = Some(value);
        }
    }
    selected
        .map(|value| value.map(|component| component / scales.velocity_value()))
        .ok_or_else(|| {
            invalid_realization(
                "an essential boundary vertex is absent from the canonical trace inventory",
            )
        })
}

fn require_mini_solver(solver: SolverPlan) -> Result<(), Diagnostic> {
    let admitted = solver.preconditioner() == PreconditionerPolicy::Identity
        && matches!(
            (solver.algorithm(), solver.reduction()),
            (LinearSolver::MinimumResidual, ReductionPolicy::Reproducible)
                | (LinearSolver::SparseLu, ReductionPolicy::Fast)
        );
    if !admitted {
        return Err(invalid_realization(
            "coherent-SI MINI Stokes requires identity-preconditioned reproducible MINRES or host-fast sparse LU",
        ));
    }
    Ok(())
}

fn requires_zero_integral_constraint(
    model: &SteadyIncompressibleStokesModel2d,
) -> Result<bool, Diagnostic> {
    let mut trace_boundaries = 0_usize;
    let mut pressure_boundaries = 0_usize;
    let mut boundary_count = 0_usize;
    for (key, entry) in model.boundary_entries() {
        boundary_count += 1;
        match entry.disposition {
            PhysicalBoundaryDisposition::TraceZero => trace_boundaries += 1,
            PhysicalBoundaryDisposition::FluxZero => {
                if model.normal_pressure_for(key).is_none() {
                    return Err(invalid_realization(format!(
                        "flux-zero Stokes boundary {key:?} is not an admitted normal-pressure law"
                    )));
                }
                pressure_boundaries += 1;
            }
            PhysicalBoundaryDisposition::Prescribed(law) => match law.quantity() {
                PhysicalBoundaryQuantity::Trace => trace_boundaries += 1,
                PhysicalBoundaryQuantity::Flux => {
                    if model.normal_pressure_for(key).is_none() {
                        return Err(invalid_realization(format!(
                            "prescribed Stokes {:?} law {} on {key:?} is outside the normal-pressure realization",
                            law.quantity(),
                            law.relation()
                        )));
                    }
                    pressure_boundaries += 1;
                }
            },
            PhysicalBoundaryDisposition::PortBinding { connection, port } => {
                return Err(invalid_realization(format!(
                    "live Stokes PortBinding {connection} through Port {port} on {key:?} requires an explicit trace-space interface Realization"
                )));
            }
        }
    }
    match (trace_boundaries, pressure_boundaries) {
        (trace, 0) if trace == boundary_count => Ok(true),
        (trace, pressure) if trace > 0 && pressure > 0 && trace + pressure == boundary_count => {
            Ok(false)
        }
        (0, pressure) if pressure == boundary_count => Err(invalid_realization(
            "the bounded MINI Stokes realization requires positive-measure essential velocity boundary; pure traction remains unsupported",
        )),
        _ => Err(invalid_realization(
            "Stokes boundary meaning is neither complete velocity trace nor an admitted velocity/static-pressure partition",
        )),
    }
}

fn require_dimension(
    value: DynQuantity,
    expected: DimExponents,
    label: &str,
) -> Result<(), Diagnostic> {
    if value.dim() != expected {
        return Err(invalid_realization(format!(
            "{label} has incompatible physical dimension {:?}",
            value.dim()
        )));
    }
    Ok(())
}

fn domain_id(model: &SteadyIncompressibleStokesModel2d) -> Id<kinds::Domain> {
    model
        .domain()
        .downcast()
        .expect("lowered Stokes Domain retains its entity kind")
}

fn velocity_id(model: &SteadyIncompressibleStokesModel2d) -> Id<kinds::Field> {
    model
        .velocity()
        .downcast()
        .expect("lowered Stokes velocity retains its Field kind")
}

fn pressure_id(model: &SteadyIncompressibleStokesModel2d) -> Id<kinds::Field> {
    model
        .pressure()
        .downcast()
        .expect("lowered Stokes pressure retains its Field kind")
}

fn force_potential_id(model: &SteadyIncompressibleStokesModel2d) -> Id<kinds::Field> {
    model
        .force_potential()
        .downcast()
        .expect("lowered Stokes force potential retains its Field kind")
}

fn realization_error(error: Diagnostic) -> Diagnostic {
    invalid_realization(error.message())
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
#[path = "realization/tests.rs"]
mod tests;
