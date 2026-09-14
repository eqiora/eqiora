//! Coherent-SI reference realization of the fixed-domain transient flow subset.

use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint, AlgebraicStructure};
pub(super) mod boundary;

use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_artifact::SimplicialMeshEnvelopeV1;
use eqiora_assembly::{AssemblyBackend, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, ScalarType};
use eqiora_meshing::{
    QuadratureRule, SimplicialMesh, simplex_duffy_gauss_legendre, triangle_duffy_gauss_legendre,
};
use eqiora_realization::{
    AlgebraicBlockScale, BackwardEulerRelationStep, CoordinateTreatment, Discretization,
    DiscretizationMethod, EnergySkewConvection, ExecutionSchedule, FieldSpaceBinding,
    FieldwiseRealizationPlan, FieldwiseRealizationRequirements, FieldwiseSpatialDiscretization,
    MeshArtifactReference, MeshPolicy, NonlinearSolvePlan, PlacementRequirementNode,
    PortableRealizationGraph, PositivePhysicalScale, QuadraturePolicy, RealizationRequirements,
    ResolvedTransientFieldwiseRealization, SolveRoot, Space, SymmetricCongruenceScaling,
    SystemBlock, Target, TransformationNode, TransientFieldwiseRealizationPlan,
    TransientFieldwiseRealizationRequirements, VectorLayoutKind,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearOperatorProperties, LinearSolverBackend, SolverPlan};

use super::navier_stokes::TransientIncompressibleNavierStokesModel2d;
use super::realization::{NormalizedCartesianSimplicialMesh2d, normalize_cartesian_mesh};

mod prepared;
use super::{
    IncompressibleFlowScaleProfile2d, TransientIncompressibleNavierStokesCartesianModel2d,
    lower_transient_incompressible_navier_stokes_cartesian_2d,
};
use crate::discrete_block::DiscreteBlockSystem;
use crate::simplicial_elliptic::SimplicialP1Field;
use crate::simplicial_navier_stokes::{
    MiniNavierStokesStepPlan2d, PreparedStepStructure, SimplicialMiniNavierStokesState2d,
    SimplicialMiniNavierStokesStepEvidence2d,
    advance_simplicial_mini_navier_stokes_2d_with_prepared_structure, prepare_step_structure,
};
use crate::simplicial_stokes::{
    SimplicialMiniStokesPressureReference2d, SimplicialMiniVelocityField2d,
};
use crate::step_count::NonZeroStepCount;
use prepared::prepare_resolved_transient_navier_stokes_mini_run_2d_with_assembly;
pub(crate) use prepared::{
    PreparedResolvedTransientMiniRun2d, prepare_resolved_transient_navier_stokes_mini_run_2d,
};

const DIMENSION: usize = 2;
const DUFFY_POINTS_PER_AXIS: usize = 5;

/// Repeated-step request, deliberately separate from Realization identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransientNavierStokesRun2d {
    step_count: NonZeroStepCount,
}

impl TransientNavierStokesRun2d {
    /// Request a non-zero number of accepted applications of one finalized step.
    #[must_use]
    pub const fn new(step_count: NonZeroStepCount) -> Self {
        Self { step_count }
    }

    /// Requested accepted-step count.
    #[must_use]
    pub const fn step_count(self) -> NonZeroStepCount {
        self.step_count
    }
}

/// Coherent-SI initial state admitted from an already accepted steady solve.
///
/// This type cannot be passed to the dimensionless MINI kernel. It retains
/// exact Field and mesh identities so scaling and provenance drift fail before
/// numerical normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientNavierStokesInitialState2d {
    pub(super) time: DynQuantity,
    pub(super) mesh_artifact: MeshArtifactReference,
    pub(super) velocity_field: Id<kinds::Field>,
    pub(super) pressure_field: Id<kinds::Field>,
    pub(super) velocity: SimplicialMiniVelocityField2d,
    pub(super) pressure: SimplicialP1Field,
    pub(super) pressure_reference: super::SteadyStokesPressureReference2d,
}

impl TransientNavierStokesInitialState2d {
    /// Bind coherent-SI fields to exact transient Model and mesh identities.
    ///
    /// This constructor validates shape and provenance only. The selected
    /// Realization reassembles continuity, pressure-mean, and gauge conditions
    /// before the first Newton iteration.
    ///
    /// # Errors
    /// Returns `EQ0807` for invalid time or mismatched velocity/pressure mesh.
    pub fn new(
        model: &TransientIncompressibleNavierStokesCartesianModel2d,
        time: DynQuantity,
        mesh_artifact: MeshArtifactReference,
        velocity: SimplicialMiniVelocityField2d,
        pressure: SimplicialP1Field,
        pressure_reference: super::SteadyStokesPressureReference2d,
    ) -> Result<Self, Diagnostic> {
        Self::new_for_model(
            &model.common_projection(),
            time,
            mesh_artifact,
            velocity,
            pressure,
            pressure_reference,
        )
    }

    pub(crate) fn new_for_model(
        model: &TransientIncompressibleNavierStokesModel2d,
        time: DynQuantity,
        mesh_artifact: MeshArtifactReference,
        velocity: SimplicialMiniVelocityField2d,
        pressure: SimplicialP1Field,
        pressure_reference: super::SteadyStokesPressureReference2d,
    ) -> Result<Self, Diagnostic> {
        if time.dim() != time_dimension() || !time.value().is_finite() || time.value() < 0.0 {
            return Err(invalid_realization(
                "transient initial time must be finite, non-negative, and have physical time dimension",
            ));
        }
        if velocity.mesh() != pressure.mesh() {
            return Err(invalid_realization(
                "transient initial velocity and pressure must share one exact mesh",
            ));
        }
        if pressure_reference
            .gauge_multiplier()
            .is_some_and(|value| !value.is_finite())
        {
            return Err(invalid_realization(
                "transient initial pressure-reference evidence must be finite",
            ));
        }
        Ok(Self {
            time,
            mesh_artifact,
            velocity_field: velocity_id(model),
            pressure_field: pressure_id(model),
            velocity,
            pressure,
            pressure_reference,
        })
    }

    #[must_use]
    pub const fn velocity(&self) -> &SimplicialMiniVelocityField2d {
        &self.velocity
    }

    #[must_use]
    pub const fn pressure(&self) -> &SimplicialP1Field {
        &self.pressure
    }

    #[must_use]
    pub const fn pressure_reference(&self) -> super::SteadyStokesPressureReference2d {
        self.pressure_reference
    }
}

/// One accepted coherent-SI transient state with exact semantic provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTransientNavierStokesState2d {
    time: DynQuantity,
    velocity_field: Id<kinds::Field>,
    pressure_field: Id<kinds::Field>,
    velocity: SimplicialMiniVelocityField2d,
    pressure: SimplicialP1Field,
    pressure_reference: super::SteadyStokesPressureReference2d,
    named_boundary_forces_on_domain: Vec<(String, [f64; DIMENSION])>,
}

impl ResolvedTransientNavierStokesState2d {
    /// Accepted physical model time.
    #[must_use]
    pub const fn time(&self) -> DynQuantity {
        self.time
    }

    /// Exact Semantic velocity Field.
    #[must_use]
    pub const fn velocity_field(&self) -> Id<kinds::Field> {
        self.velocity_field
    }

    /// Exact Semantic pressure Field.
    #[must_use]
    pub const fn pressure_field(&self) -> Id<kinds::Field> {
        self.pressure_field
    }

    /// Physical MINI velocity in metres per second.
    #[must_use]
    pub const fn velocity(&self) -> &SimplicialMiniVelocityField2d {
        &self.velocity
    }

    /// Physical P1 pressure in pascals.
    #[must_use]
    pub const fn pressure(&self) -> &SimplicialP1Field {
        &self.pressure
    }

    /// Physical pressure closure retained at acceptance.
    #[must_use]
    pub const fn pressure_reference(&self) -> super::SteadyStokesPressureReference2d {
        self.pressure_reference
    }

    /// Intrinsic-2D force exerted on the fluid domain by one authenticated boundary.
    #[must_use]
    pub fn named_boundary_force_on_domain(&self, name: &str) -> Option<[f64; DIMENSION]> {
        self.named_boundary_forces_on_domain
            .iter()
            .find_map(|(candidate, force)| (candidate == name).then_some(*force))
    }

    pub(crate) fn named_boundary_forces_on_domain(&self) -> &[(String, [f64; DIMENSION])] {
        &self.named_boundary_forces_on_domain
    }
}

/// Exact lowerer requirements for the admitted transient MINI path.
#[must_use]
pub fn transient_navier_stokes_fieldwise_requirements_2d(
    model: &TransientIncompressibleNavierStokesCartesianModel2d,
) -> TransientFieldwiseRealizationRequirements {
    transient_navier_stokes_fieldwise_requirements_for_2d(&model.common_projection())
}

pub(super) fn transient_navier_stokes_fieldwise_requirements_for_2d(
    model: &TransientIncompressibleNavierStokesModel2d,
) -> TransientFieldwiseRealizationRequirements {
    let fieldwise = FieldwiseRealizationRequirements::new(
        domain_id(model),
        [velocity_id(model), pressure_id(model)],
        RealizationRequirements::new(
            NonZeroUsize::new(DIMENSION).expect("two is non-zero"),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
    )
    .expect("a lowered transient flow model owns two distinct algebraic Fields");
    TransientFieldwiseRealizationRequirements::new(
        fieldwise,
        momentum_id(model),
        velocity_id(model),
    )
    .expect("the transient velocity is part of the exact Field inventory")
}

/// Build the sole admitted transient MINI/P1 reference Realization.
///
/// The result owns method, mesh, quadrature, scaling, time discretization,
/// nonlinear policy, linear solver, and placement. Repeated step count remains
/// a separate [`TransientNavierStokesRun2d`].
///
/// # Errors
/// Returns `EQ0807` for an invalid scale, solver, or composed transient plan.
pub fn transient_navier_stokes_mini_plan_2d(
    model: &TransientIncompressibleNavierStokesCartesianModel2d,
    mesh: MeshArtifactReference,
    scales: IncompressibleFlowScaleProfile2d,
    time_step: DynQuantity,
    nonlinear: NonlinearSolvePlan,
    solver: SolverPlan,
) -> Result<TransientFieldwiseRealizationPlan, Diagnostic> {
    transient_navier_stokes_mini_plan_for_2d(
        &model.common_projection(),
        mesh,
        scales,
        time_step,
        nonlinear,
        solver,
    )
}

pub(crate) fn transient_mini_algebraic_structure(
    model: &TransientIncompressibleNavierStokesModel2d,
) -> Result<AlgebraicStructure, Diagnostic> {
    let pressure = pressure_id(model);
    AlgebraicStructure::new(
        [velocity_id(model), pressure],
        boundary::pressure_uses_gauge(model)?
            .then_some(AlgebraicConstraint::ZeroIntegral { field: pressure }),
    )
}

pub(super) fn transient_navier_stokes_mini_plan_for_2d(
    model: &TransientIncompressibleNavierStokesModel2d,
    mesh: MeshArtifactReference,
    scales: IncompressibleFlowScaleProfile2d,
    time_step: DynQuantity,
    nonlinear: NonlinearSolvePlan,
    solver: SolverPlan,
) -> Result<TransientFieldwiseRealizationPlan, Diagnostic> {
    let velocity = velocity_id(model);
    let pressure = pressure_id(model);
    let structure = transient_mini_algebraic_structure(model)?;
    let spatial = FieldwiseSpatialDiscretization::new(
        domain_id(model),
        PositivePhysicalScale::new(scales.length()).map_err(realization_error)?,
        [
            FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
            FieldSpaceBinding::new(pressure, Space::continuous_lagrange(NonZeroU16::MIN)),
        ],
        structure.constraints().iter().copied(),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial { artifact: mesh },
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(DUFFY_POINTS_PER_AXIS)
                    .expect("five is non-zero"),
            },
        ),
    )
    .map_err(realization_error)?;
    let mut block_scales = vec![
        AlgebraicBlockScale::new(
            AlgebraicBlock::Field(velocity),
            PositivePhysicalScale::new(scales.velocity()).map_err(realization_error)?,
        ),
        AlgebraicBlockScale::new(
            AlgebraicBlock::Field(pressure),
            PositivePhysicalScale::new(scales.pressure()).map_err(realization_error)?,
        ),
    ];
    if !structure.constraints().is_empty() {
        block_scales.push(AlgebraicBlockScale::new(
            AlgebraicBlock::ConstraintMultiplier { field: pressure },
            PositivePhysicalScale::new(scales.gauge()).map_err(realization_error)?,
        ));
    }
    let scaling = SymmetricCongruenceScaling::new(
        block_scales,
        PositivePhysicalScale::new(scales.weak_functional()).map_err(realization_error)?,
    )
    .map_err(realization_error)?;
    let fieldwise = FieldwiseRealizationPlan::new(
        spatial,
        scaling,
        LinearOperatorProperties::General,
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .map_err(realization_error)?;
    TransientFieldwiseRealizationPlan::new(
        fieldwise,
        BackwardEulerRelationStep::new(momentum_id(model), velocity, time_step)
            .map_err(realization_error)?,
        EnergySkewConvection::new(momentum_id(model), velocity),
        nonlinear,
    )
    .map_err(realization_error)
}

/// Coherent-SI states plus dimensionless nonlinear acceptance evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTransientNavierStokesTrajectory2d {
    model: TransientIncompressibleNavierStokesCartesianModel2d,
    realization: ResolvedTransientFieldwiseRealization,
    realization_graph: PortableRealizationGraph,
    solver_backend: eqiora_solver::BackendId,
    mesh_artifact: MeshArtifactReference,
    scales: IncompressibleFlowScaleProfile2d,
    states: Vec<ResolvedTransientNavierStokesState2d>,
    steps: Vec<SimplicialMiniNavierStokesStepEvidence2d>,
    validated_block_materializations: usize,
}

impl ResolvedTransientNavierStokesTrajectory2d {
    /// Exact package-neutral model roles used by every step.
    #[must_use]
    pub const fn model(&self) -> &TransientIncompressibleNavierStokesCartesianModel2d {
        &self.model
    }

    /// Exact resolved Realization and its independent revision provenance.
    #[must_use]
    pub const fn realization(&self) -> &ResolvedTransientFieldwiseRealization {
        &self.realization
    }

    /// Common portable DAG that drove numerical admission for every step.
    #[must_use]
    pub const fn realization_graph(&self) -> &PortableRealizationGraph {
        &self.realization_graph
    }

    /// Exact solver adapter that executed every accepted Newton linearization.
    #[must_use]
    pub const fn solver_backend(&self) -> eqiora_solver::BackendId {
        self.solver_backend
    }

    /// Fixed imported mesh content identity.
    #[must_use]
    pub const fn mesh_artifact(&self) -> MeshArtifactReference {
        self.mesh_artifact
    }

    /// Coherent-SI initial state followed by accepted states.
    #[must_use]
    pub fn states(&self) -> &[ResolvedTransientNavierStokesState2d] {
        &self.states
    }

    /// Dimensionless true-residual and conservation evidence per step.
    #[must_use]
    pub fn steps(&self) -> &[SimplicialMiniNavierStokesStepEvidence2d] {
        &self.steps
    }

    /// Number of CSR assemblies bound to and revalidated against one exact
    /// private block-system identity during this run.
    #[must_use]
    pub const fn validated_block_materialization_count(&self) -> usize {
        self.validated_block_materializations
    }

    /// Exact scaling used for numerical coordinates and reconstruction.
    #[must_use]
    pub const fn scales(&self) -> IncompressibleFlowScaleProfile2d {
        self.scales
    }
}

/// Advance one exact canonical model through the reference assembler.
///
/// # Errors
/// Rejects semantic, boundary, mesh, initial-state, scaling, nonlinear, JVP,
/// Krylov, or reconstruction drift before returning accepted evidence.
#[allow(clippy::too_many_arguments)]
pub fn advance_resolved_transient_navier_stokes_mini_2d(
    program: &KernelProgram,
    resolved: &ResolvedTransientFieldwiseRealization,
    mesh: &SimplicialMeshEnvelopeV1,
    initial: TransientNavierStokesInitialState2d,
    run: TransientNavierStokesRun2d,
    solver: &dyn LinearSolverBackend,
) -> Result<ResolvedTransientNavierStokesTrajectory2d, Diagnostic> {
    advance_resolved_transient_navier_stokes_mini_2d_with_assembly(
        program,
        resolved,
        mesh,
        initial,
        run,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
}

/// Advance through an explicit assembly adapter and common linear backend.
///
/// The accepted envelope keeps content identity and reconstructed mesh data
/// indivisible at this public boundary.
///
/// # Errors
/// Preserves all canonical, fixed-mesh, block-system, assembly, nonlinear,
/// linearization, solver, and physical reconstruction diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn advance_resolved_transient_navier_stokes_mini_2d_with_assembly(
    program: &KernelProgram,
    resolved: &ResolvedTransientFieldwiseRealization,
    mesh: &SimplicialMeshEnvelopeV1,
    initial: TransientNavierStokesInitialState2d,
    run: TransientNavierStokesRun2d,
    assembly: &dyn AssemblyBackend,
    solver: &dyn LinearSolverBackend,
) -> Result<ResolvedTransientNavierStokesTrajectory2d, Diagnostic> {
    prepare_resolved_transient_navier_stokes_mini_run_2d_with_assembly(
        program, resolved, mesh, assembly,
    )?
    .advance(initial, run, solver)
}

pub(super) fn normalize_state(
    state: &TransientNavierStokesInitialState2d,
    mesh: &SimplicialMesh,
    scales: IncompressibleFlowScaleProfile2d,
    with_gauge: bool,
) -> Result<SimplicialMiniNavierStokesState2d, Diagnostic> {
    let velocity = SimplicialMiniVelocityField2d::new(
        mesh.clone(),
        state
            .velocity
            .vertex_values()
            .iter()
            .map(|value| {
                [
                    value[0] / scales.velocity_value(),
                    value[1] / scales.velocity_value(),
                ]
            })
            .collect(),
        state
            .velocity
            .cell_bubble_values()
            .iter()
            .map(|value| {
                [
                    value[0] / scales.velocity_value(),
                    value[1] / scales.velocity_value(),
                ]
            })
            .collect(),
    )?;
    let pressure = SimplicialP1Field::new(
        mesh.clone(),
        state
            .pressure
            .vertex_values()
            .iter()
            .map(|value| value / scales.pressure_value())
            .collect(),
    )?;
    let pressure_reference = match (state.pressure_reference, with_gauge) {
        (super::SteadyStokesPressureReference2d::ZeroIntegral { multiplier }, true) => {
            SimplicialMiniStokesPressureReference2d::ZeroIntegral {
                multiplier: multiplier / scales.gauge_value(),
            }
        }
        (super::SteadyStokesPressureReference2d::BoundaryTraction, false) => {
            SimplicialMiniStokesPressureReference2d::BoundaryTraction
        }
        _ => {
            return Err(invalid_realization(
                "transient initial pressure closure differs from the boundary-determined pressure regime",
            ));
        }
    };
    SimplicialMiniNavierStokesState2d::new(
        state.time.value() * scales.velocity_value() / scales.length_value(),
        velocity,
        pressure,
        pressure_reference,
    )
    .map_err(|error| invalid_realization(error.message()))
}

pub(super) fn reconstruct_state(
    state: &SimplicialMiniNavierStokesState2d,
    mesh: &SimplicialMesh,
    model: &TransientIncompressibleNavierStokesModel2d,
    scales: IncompressibleFlowScaleProfile2d,
    step: Option<&crate::simplicial_navier_stokes::SimplicialMiniNavierStokesStepEvidence2d>,
) -> Result<ResolvedTransientNavierStokesState2d, Diagnostic> {
    let velocity = SimplicialMiniVelocityField2d::new(
        mesh.clone(),
        state
            .velocity()
            .vertex_values()
            .iter()
            .map(|value| {
                [
                    value[0] * scales.velocity_value(),
                    value[1] * scales.velocity_value(),
                ]
            })
            .collect(),
        state
            .velocity()
            .cell_bubble_values()
            .iter()
            .map(|value| {
                [
                    value[0] * scales.velocity_value(),
                    value[1] * scales.velocity_value(),
                ]
            })
            .collect(),
    )?;
    let pressure = SimplicialP1Field::new(
        mesh.clone(),
        state
            .pressure()
            .vertex_values()
            .iter()
            .map(|value| value * scales.pressure_value())
            .collect(),
    )?;
    let pressure_reference = match state.pressure_reference() {
        SimplicialMiniStokesPressureReference2d::ZeroIntegral { multiplier } => {
            super::SteadyStokesPressureReference2d::ZeroIntegral {
                multiplier: multiplier * scales.gauge_value(),
            }
        }
        SimplicialMiniStokesPressureReference2d::BoundaryTraction => {
            super::SteadyStokesPressureReference2d::BoundaryTraction
        }
    };
    let time = DynQuantity::new(
        state.time() * scales.length_value() / scales.velocity_value(),
        time_dimension(),
    );
    if !time.value().is_finite() {
        return Err(invalid_realization(
            "transient physical reconstruction produced non-finite time",
        ));
    }
    let force_scale = scales.pressure_value() * scales.length_value();
    let named_boundary_forces_on_domain = step
        .into_iter()
        .flat_map(|evidence| evidence.named_boundary_reactions())
        .map(|(name, force)| (name.clone(), force.map(|component| component * force_scale)))
        .collect::<Vec<_>>();
    if named_boundary_forces_on_domain
        .iter()
        .flat_map(|(_, force)| force)
        .any(|component| !component.is_finite())
    {
        return Err(invalid_realization(
            "transient physical boundary-force reconstruction produced a non-finite value",
        ));
    }
    Ok(ResolvedTransientNavierStokesState2d {
        time,
        velocity_field: velocity_id(model),
        pressure_field: pressure_id(model),
        velocity,
        pressure,
        pressure_reference,
        named_boundary_forces_on_domain,
    })
}

pub(crate) fn require_complete_zero_trace(
    model: &TransientIncompressibleNavierStokesCartesianModel2d,
) -> Result<(), Diagnostic> {
    let entries = model.boundary_inventory().entries().collect::<Vec<_>>();
    if entries.len() != 2 * DIMENSION
        || entries.iter().any(|(_, entry)| {
            entry.disposition() != crate::canonical_boundary::PhysicalBoundaryDisposition::TraceZero
        })
    {
        return Err(invalid_realization(
            "bounded transient Navier--Stokes realization requires complete homogeneous velocity trace",
        ));
    }
    Ok(())
}

pub(super) fn require_exact_transient_plan(
    model: &TransientIncompressibleNavierStokesModel2d,
    resolved: &ResolvedTransientFieldwiseRealization,
    graph: &PortableRealizationGraph,
    mesh_artifact: MeshArtifactReference,
) -> Result<
    (
        IncompressibleFlowScaleProfile2d,
        crate::simplicial_navier_stokes::MiniNavierStokesStepPlan2d,
    ),
    Diagnostic,
> {
    let with_gauge = boundary::pressure_uses_gauge(model)?;
    let expected_requirements = transient_navier_stokes_fieldwise_requirements_for_2d(model);
    if resolved.requirements() != &expected_requirements {
        return Err(invalid_realization(
            "transient Realization requirements differ from the exact flow lowerer inventory",
        ));
    }
    let velocity = velocity_id(model);
    let pressure = pressure_id(model);
    if graph.lineage().model() != resolved.model()
        || graph.lineage().semantic_revision() != resolved.semantic_revision()
        || graph.domains().len() != 1
        || graph.domains()[0].domain() != domain_id(model)
        || graph.domains()[0].discretization()
            != Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: mesh_artifact,
                },
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(DUFFY_POINTS_PER_AXIS)
                        .expect("five is non-zero"),
                },
            )
    {
        return Err(invalid_realization(
            "transient portable graph lineage or Domain discretization differs from the exact flow lowerer",
        ));
    }
    let exact_space = |field, expected| {
        graph
            .fields()
            .iter()
            .find(|binding| binding.field() == field)
            .is_some_and(|binding| binding.space() == expected)
    };
    let [system] = graph.systems() else {
        return Err(invalid_realization(
            "transient portable graph must contain one monolithic algebraic system",
        ));
    };
    let algebraic_blocks = system
        .blocks()
        .iter()
        .map(|block| match *block {
            SystemBlock::Field(id) => graph
                .field(id)
                .map(|field| AlgebraicBlock::Field(field.field()))
                .ok_or_else(|| invalid_realization("portable system has an absent Field block")),
            SystemBlock::ConstraintMultiplier(constraint) => {
                Ok(AlgebraicBlock::ConstraintMultiplier {
                    field: constraint.field(),
                })
            }
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let mut expected_blocks = vec![
        AlgebraicBlock::Field(velocity),
        AlgebraicBlock::Field(pressure),
    ];
    if with_gauge {
        expected_blocks.push(AlgebraicBlock::ConstraintMultiplier { field: pressure });
    }
    expected_blocks.sort_by_key(|block| match *block {
        AlgebraicBlock::Field(field) => (0_u8, field.ulid()),
        AlgebraicBlock::ConstraintMultiplier { field } => (1_u8, field.ulid()),
    });
    if !exact_space(velocity, Space::simplex_p1_bubble())
        || !exact_space(pressure, Space::continuous_lagrange(NonZeroU16::MIN))
        || algebraic_blocks != expected_blocks
        || system.operator_properties() != LinearOperatorProperties::General
        || system.scalar_type() != ScalarType::F64
        || system.partition() != VectorLayoutKind::Replicated
    {
        return Err(invalid_realization(
            "transient flow requires exact MINI/P1, boundary-selected pressure closure, replicated-f64 monolithic Realization graph choices",
        ));
    }
    let SolveRoot::Nonlinear(nonlinear_id) = graph.root() else {
        return Err(invalid_realization(
            "transient flow requires a nonlinear portable solve root",
        ));
    };
    let nonlinear_node = graph
        .nonlinear_solve(nonlinear_id)
        .ok_or_else(|| invalid_realization("transient nonlinear root is absent"))?;
    let linear_node = graph
        .linear_solve(nonlinear_node.linearization())
        .ok_or_else(|| invalid_realization("transient linearization solve is absent"))?;
    let placement = graph
        .placement(linear_node.placement())
        .ok_or_else(|| invalid_realization("transient placement requirement is absent"))?;
    if placement
        != (PlacementRequirementNode::HostWorkers {
            workers_per_partition: NonZeroUsize::MIN,
        })
        || linear_node.schedule() != ExecutionSchedule::Offline
    {
        return Err(invalid_realization(
            "transient reference requires one portable host worker and offline scheduling",
        ));
    }
    let velocity_representation = system
        .blocks()
        .iter()
        .find_map(|block| match *block {
            SystemBlock::Field(id)
                if graph
                    .field(id)
                    .is_some_and(|field| field.field() == velocity) =>
            {
                Some(id)
            }
            _ => None,
        })
        .ok_or_else(|| invalid_realization("portable graph has no velocity system block"))?;
    if graph.transformations()
        != [
            TransformationNode::BackwardEulerDerivative {
                relation: momentum_id(model),
                state: velocity_representation,
                duration: resolved.plan().time_step().duration(),
            },
            TransformationNode::EnergySkewConvection {
                relation: momentum_id(model),
                velocity: velocity_representation,
            },
        ]
    {
        return Err(invalid_realization(
            "transient portable graph must bind Backward Euler and energy-skew transformations to the exact momentum Relation and velocity Field",
        ));
    }
    let scale_for = |block| {
        system
            .congruence_scaling()
            .ok_or_else(|| {
                invalid_realization("transient portable graph requires congruence scaling")
            })?
            .block_scales()
            .iter()
            .find(|entry| entry.block() == block)
            .map(|entry| entry.scale().quantity())
            .ok_or_else(|| invalid_realization("transient plan is missing an exact block scale"))
    };
    let scales = IncompressibleFlowScaleProfile2d::new(
        match graph.domains()[0].coordinates() {
            CoordinateTreatment::Scaled(scale) => scale.quantity(),
            CoordinateTreatment::Physical => {
                return Err(invalid_realization(
                    "transient coherent-SI graph requires an explicit coordinate length scale",
                ));
            }
        },
        scale_for(AlgebraicBlock::Field(velocity))?,
        scale_for(AlgebraicBlock::Field(pressure))?,
    )?;
    if (with_gauge
        && scale_for(AlgebraicBlock::ConstraintMultiplier { field: pressure })? != scales.gauge())
        || system
            .congruence_scaling()
            .ok_or_else(|| {
                invalid_realization("transient portable graph requires congruence scaling")
            })?
            .weak_functional_scale()
            .quantity()
            != scales.weak_functional()
    {
        return Err(invalid_realization(
            "transient plan pressure-closure or weak-functional scale is not derived exactly from L/U/P",
        ));
    }
    let nonlinear = nonlinear_node.plan();
    let density = model.mass_density() * scales.velocity_value().powi(2) / scales.pressure_value();
    let viscosity = model.dynamic_viscosity() * scales.velocity_value()
        / (scales.pressure_value() * scales.length_value());
    let time_step = resolved.plan().time_step().duration().value() * scales.velocity_value()
        / scales.length_value();
    let numerical = crate::simplicial_navier_stokes::MiniNavierStokesStepPlan2d::new(
        density,
        viscosity,
        time_step,
        nonlinear.relative_tolerance(),
        nonlinear.absolute_tolerance(),
        nonlinear.maximum_iterations(),
        nonlinear.maximum_line_search_steps(),
        linear_node.plan(),
        resolved.plan().fieldwise().target(),
    )
    .map_err(|error| invalid_realization(error.message()))?;
    Ok((scales, numerical))
}

pub(super) fn domain_id(model: &TransientIncompressibleNavierStokesModel2d) -> Id<kinds::Domain> {
    model
        .domain
        .downcast()
        .expect("transient lowerer retains a Domain identity")
}

pub(super) fn velocity_id(model: &TransientIncompressibleNavierStokesModel2d) -> Id<kinds::Field> {
    model
        .velocity
        .downcast()
        .expect("transient lowerer retains a velocity Field identity")
}

pub(super) fn pressure_id(model: &TransientIncompressibleNavierStokesModel2d) -> Id<kinds::Field> {
    model
        .pressure
        .downcast()
        .expect("transient lowerer retains a pressure Field identity")
}

pub(super) fn momentum_id(
    model: &TransientIncompressibleNavierStokesModel2d,
) -> Id<kinds::Relation> {
    model
        .momentum_relation
        .downcast()
        .expect("transient lowerer retains a momentum Relation identity")
}

const fn time_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension")
}

fn realization_error(error: Diagnostic) -> Diagnostic {
    invalid_realization(error.message())
}

pub(super) fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
