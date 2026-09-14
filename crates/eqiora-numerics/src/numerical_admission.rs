//! Private Model-driven admission of exact common Mesh resources and numerical policy.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::canonical::{
    ScalarEllipticCartesianBoundary, geometry_cartesian_support,
    project_scalar_conservation_for_differentiation,
};
use crate::canonical_elasticity::{
    IsotropicElasticityContinuum, finalize_isotropic_elasticity_cartesian_q1_on_mesh,
    lower_isotropic_elasticity_geometry_2d, recognize_isotropic_elasticity_geometry_mathematics,
};
use crate::canonical_fsi::{
    FixedReferenceFsiCartesianModel2d, FixedReferenceFsiScaleProfile2d,
    PreparedResolvedFixedReferenceFsiRun2d, fixed_reference_fsi_plan_2d,
    fixed_reference_fsi_requirements_2d, lower_fixed_reference_fsi_geometry_2d,
    prepare_resolved_fixed_reference_fsi_run_2d,
};
use crate::canonical_stokes::{
    CellCenteredNavierStokesInitialState2d, IncompressibleScalingReceipt2d,
    IncompressibleScalingRequest2d, PreparedResolvedTransientCellCenteredRun2d,
    PreparedResolvedTransientGeometryMiniRun2d, PreparedResolvedTransientMiniRun2d,
    ResolvedIncompressibleScaling2d, ResolvedTransientNavierStokesState2d,
    TransientIncompressibleNavierStokesCartesianModel2d, TransientNavierStokesGeometryBinding2d,
    TransientNavierStokesInitialState2d, TransientNavierStokesRun2d,
    integral_conservative_correspondence,
    lower_transient_incompressible_navier_stokes_cartesian_2d,
    prepare_resolved_transient_navier_stokes_cell_centered_run_2d,
    prepare_resolved_transient_navier_stokes_geometry_mini_run_2d,
    prepare_resolved_transient_navier_stokes_mini_run_2d,
    recognize_steady_incompressible_stokes_geometry_mathematics,
    recognize_transient_incompressible_navier_stokes_geometry_mathematics,
    resolve_complete_manual_incompressible_scaling_2d, resolve_fixed_reference_fsi_scaling_2d,
    solve_resolved_steady_stokes_geometry_mini_2d, transient_navier_stokes_cell_centered_plan_2d,
    transient_navier_stokes_cell_centered_requirements_2d,
    transient_navier_stokes_fieldwise_requirements_2d, transient_navier_stokes_mini_plan_2d,
};
use crate::cartesian_elasticity::CartesianLinearElasticity2dSolution;
use crate::cartesian_elliptic::{
    CartesianBoundaryValue, finalize_scalar_elliptic_cartesian_fvm,
    linearize_scalar_elliptic_cartesian_fem, linearize_scalar_elliptic_cartesian_fem_output,
    linearize_scalar_elliptic_cartesian_fvm, linearize_scalar_elliptic_cartesian_fvm_output,
};
use crate::common::{AssembledLinearizedRelation, SpatialDesignCoordinate};
use crate::common_ode::{CommonOdePlan, CommonTsitouras45};
use crate::finalized_spatial::FinalizedScalarEllipticCartesianProblem;
use crate::fluid::{
    CellCenteredPressureField2d, CellCenteredVelocityField2d, IncompressibleFlowScaleProfile2d,
    SimplicialMiniVelocityField2d, SteadyStokesGeometryBinding2d, SteadyStokesPressureReference2d,
};
use crate::fsi::{
    FixedReferenceFsiPartition, FixedReferenceFsiState, ResolvedFixedReferenceFsiSolution2d,
};
use crate::scalar::{CartesianScalarFieldLinearization, ResolvedScalarEllipticCartesianSolution};
use crate::scalar_conservation::{
    ScalarConservationDescriptor, ScalarExteriorLaw, ScalarRegionSupport,
    recognize_scalar_conservation_on_supports,
};
use crate::simplicial_elliptic::SimplicialP1Field;
use crate::step_count::NonZeroStepCount;
use eqiora_artifact::{
    CanonicalModelArtifact, CartesianMeshEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1,
    MeshProductionLineageEnvelopeV1, ModelEnvelope, SimplicialMeshEnvelopeV1,
};
use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_compiler::AuthoredFormulationProjection;
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, ScalarType};
use eqiora_execution::{
    AdmittedExecution, DeploymentBinding, ExecutionReceipt, HostExecutorDescriptor,
};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_io_gmsh::{Msh41Policy, import_msh41};
use eqiora_meshing::{CellId, MeshEntity, MeshTopology, QuadratureRule, SimplicialMesh};
use eqiora_realization::{
    CoupledFieldwiseRealizationRequest, Discretization, DiscretizationMethod, ExecutionSchedule,
    FieldwiseRealizationRequest, MeshArtifactReference, MeshKind, MeshPolicy, NonlinearSolvePlan,
    PortableRealizationGraph, QuadraturePolicy, RealizationCapabilities, RealizationLineage,
    RealizationRevision, ResolvedCoupledFieldwiseRealization, ResolvedFieldwiseRealization,
    ResolvedTransientCellCenteredIncompressibleFlowRealization,
    ResolvedTransientFieldwiseRealization, SemanticRevision, Space, SpaceFamily,
    SpatialDimensionSupport, Target, TargetCapabilities,
    TransientCellCenteredIncompressibleFlowCapabilities,
    TransientCellCenteredIncompressibleFlowRealizationRequest,
    TransientFieldwiseRealizationRequest, VectorLayoutKind, resolve_coupled_fieldwise,
    resolve_fieldwise, resolve_transient_cell_centered_incompressible_flow,
    resolve_transient_fieldwise,
};
use eqiora_schema::kernel::BoundarySide;
use eqiora_sem::KernelProgram;
use eqiora_solver::{
    ExecutionProvider, LinearOperatorProperties, LinearSolveRequest, LinearSolver,
    LinearSolverBackend, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy,
    ResolvedHostSerialSolverPlan, SERIAL_EXECUTION_PROVIDER, SolverCapabilities, SolverCapability,
    SolverPlan, SolverPlanningObjective, SolverProvider,
};
use eqiora_time::TimeBackendIdentity;

use sha2::{Digest, Sha256};

pub use crate::form_compiler::vocabulary::FormulationKind;

const APPLICATION_REALIZATION_REVISION: u64 = 134;
const COMMON_SCALAR_REALIZATION_REVISION: u64 = 170;
const COMMON_ELASTICITY_REALIZATION_REVISION: u64 = 171;
const TRANSIENT_REALIZATION_REVISION: u64 = 166;
const COMMON_TRANSIENT_RESOLVER_EPOCH: u64 = 1;
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
const POLICY_DOMAIN: &[u8] = b"eqiora.private-native-numerical-admission/v1\0";
/// Closed spatial choice requested from the Model-first common resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommonSpatialPolicy {
    Q1,
    P1,
    CellCenteredTpfa,
    MiniP1,
    CellCentered,
}

/// One exact Domain-scoped spatial policy binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonScopedSpatialPolicy {
    model: eqiora_artifact::ArtifactDigest,
    domain: eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
    policy: CommonSpatialPolicy,
}

impl CommonScopedSpatialPolicy {
    #[must_use]
    pub fn new(
        model: eqiora_artifact::ArtifactDigest,
        domain: eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
        policy: CommonSpatialPolicy,
    ) -> Self {
        Self {
            model,
            domain,
            policy,
        }
    }
    #[must_use]
    pub const fn model(&self) -> &eqiora_artifact::ArtifactDigest {
        &self.model
    }
    #[must_use]
    pub const fn domain(&self) -> eqiora_core::Id<eqiora_core::entity::kinds::Domain> {
        self.domain
    }
    #[must_use]
    pub const fn policy(&self) -> CommonSpatialPolicy {
        self.policy
    }
}

/// Closed numerical-method request consumed by the common Model-first resolver.
///
/// Formulation and spatial policy remain distinct choices. The sum type makes
/// exact Formulation override available only for the current uniform-method
/// consumers and cannot represent an unsupported exact scoped request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommonMethodRequest {
    /// Use one uniform spatial policy and select the effective Formulation automatically.
    Uniform(CommonSpatialPolicy),
    /// Require one exact Formulation for one uniform spatial policy.
    Exact {
        /// Requested finite-dimensional spatial method.
        spatial: CommonSpatialPolicy,
        /// Exact mathematical Formulation that resolution must admit unchanged.
        formulation: FormulationKind,
    },
    /// Use exact Domain-scoped spatial policies with automatic Formulation ownership.
    Scoped(Vec<CommonScopedSpatialPolicy>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommonSpatialRequest {
    Uniform(CommonSpatialPolicy),
    Scoped(Vec<CommonScopedSpatialPolicy>),
}

impl CommonMethodRequest {
    fn split(self) -> (CommonSpatialRequest, Option<FormulationKind>) {
        match self {
            Self::Uniform(spatial) => (CommonSpatialRequest::Uniform(spatial), None),
            Self::Exact {
                spatial,
                formulation,
            } => (CommonSpatialRequest::Uniform(spatial), Some(formulation)),
            Self::Scoped(spatial) => (CommonSpatialRequest::Scoped(spatial), None),
        }
    }
}

/// Immutable coherent-SI values for one supported exact Field association.
#[derive(Debug, Clone, PartialEq)]
pub enum CommonInitialValues {
    Scalar(Box<[f64]>),
    Vector2(Box<[[f64; 2]]>),
}

/// One exact Model/Field-bound initial assignment with bounded associations.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonInitialField {
    model: eqiora_artifact::ArtifactDigest,
    field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
    vertex: Option<CommonInitialValues>,
    cell: Option<CommonInitialValues>,
}

impl CommonInitialField {
    pub fn new(
        model: eqiora_artifact::ArtifactDigest,
        field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
        vertex: Option<CommonInitialValues>,
        cell: Option<CommonInitialValues>,
    ) -> Result<Self, Diagnostic> {
        if vertex.is_none() && cell.is_none() {
            return Err(invalid(
                "InitialField requires vertex_values or cell_values",
            ));
        }
        let finite = |values: &CommonInitialValues| match values {
            CommonInitialValues::Scalar(values) => values.iter().all(|value| value.is_finite()),
            CommonInitialValues::Vector2(values) => {
                values.iter().flatten().all(|value| value.is_finite())
            }
        };
        if vertex.as_ref().is_some_and(|values| !finite(values))
            || cell.as_ref().is_some_and(|values| !finite(values))
        {
            return Err(invalid(
                "InitialField values must be finite coherent-SI numbers",
            ));
        }
        Ok(Self {
            model,
            field,
            vertex,
            cell,
        })
    }
    #[must_use]
    pub const fn model(&self) -> &eqiora_artifact::ArtifactDigest {
        &self.model
    }
    #[must_use]
    pub const fn field(&self) -> eqiora_core::Id<eqiora_core::entity::kinds::Field> {
        self.field
    }
    #[must_use]
    pub const fn vertex(&self) -> Option<&CommonInitialValues> {
        self.vertex.as_ref()
    }
    #[must_use]
    pub const fn cell(&self) -> Option<&CommonInitialValues> {
        self.cell.as_ref()
    }
}

impl From<CommonSpatialPolicy> for CommonMethodRequest {
    fn from(value: CommonSpatialPolicy) -> Self {
        Self::Uniform(value)
    }
}

/// Closed time integration policy requested from the common resolver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonBackwardEuler {
    step: DynQuantity,
}

impl CommonBackwardEuler {
    /// Construct one positive coherent-SI operator step.
    pub fn from_seconds(step_s: f64) -> Result<Self, Diagnostic> {
        if !step_s.is_finite() || step_s <= 0.0 || step_s.to_bits() == (-0.0_f64).to_bits() {
            return Err(invalid(
                "BackwardEuler step_s must be finite and strictly positive",
            ));
        }
        Ok(Self {
            step: DynQuantity::new(step_s, TIME),
        })
    }

    /// Exact physical duration of one operator construction.
    #[must_use]
    pub const fn step(self) -> DynQuantity {
        self.step
    }
}

/// Explicit linear-solver intent admitted by the common resolver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonLinearRequest {
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: NonZeroUsize,
    selection: CommonLinearSelection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CommonLinearSelection {
    Exact {
        plan: SolverPlan,
        provider: SolverProvider,
    },
    ProgramControlled(SolverPlanningObjective),
}

impl CommonLinearRequest {
    /// Request exactly one caller-owned plan and complete provider release.
    /// Admission never substitutes another algorithm, provider, or objective.
    pub fn exact(plan: SolverPlan, provider: SolverProvider) -> Result<Self, Diagnostic> {
        provider.validate()?;
        Self::validated(
            plan.relative_tolerance(),
            plan.absolute_tolerance(),
            plan.maximum_iterations(),
            CommonLinearSelection::Exact { plan, provider },
        )
    }

    /// Request ranking by the versioned host-serial policy after hard admission.
    pub fn program_controlled(
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: NonZeroUsize,
        objective: SolverPlanningObjective,
    ) -> Result<Self, Diagnostic> {
        Self::validated(
            relative_tolerance,
            absolute_tolerance,
            maximum_iterations,
            CommonLinearSelection::ProgramControlled(objective),
        )
    }

    fn validated(
        relative_tolerance: f64,
        absolute_tolerance: f64,
        maximum_iterations: NonZeroUsize,
        selection: CommonLinearSelection,
    ) -> Result<Self, Diagnostic> {
        if !relative_tolerance.is_finite()
            || !absolute_tolerance.is_finite()
            || relative_tolerance < 0.0
            || absolute_tolerance < 0.0
            || relative_tolerance.to_bits() == (-0.0_f64).to_bits()
            || absolute_tolerance.to_bits() == (-0.0_f64).to_bits()
            || (relative_tolerance == 0.0 && absolute_tolerance == 0.0)
        {
            return Err(invalid(
                "solver tolerances must be finite and non-negative, without signed zero, with at least one positive",
            ));
        }
        Ok(Self {
            relative_tolerance,
            absolute_tolerance,
            maximum_iterations,
            selection,
        })
    }

    /// Requested relative residual tolerance.
    #[must_use]
    pub const fn relative_tolerance(self) -> f64 {
        self.relative_tolerance
    }

    /// Requested absolute residual tolerance.
    #[must_use]
    pub const fn absolute_tolerance(self) -> f64 {
        self.absolute_tolerance
    }

    /// Requested iteration ceiling.
    #[must_use]
    pub const fn maximum_iterations(self) -> NonZeroUsize {
        self.maximum_iterations
    }

    /// Explicit ranked objective; exact manual requests have no objective.
    #[must_use]
    pub const fn objective(self) -> Option<SolverPlanningObjective> {
        match self.selection {
            CommonLinearSelection::ProgramControlled(objective) => Some(objective),
            CommonLinearSelection::Exact { .. } => None,
        }
    }

    /// Exact caller-owned plan and provider, absent for ranked requests.
    #[must_use]
    pub const fn exact_request(self) -> Option<(SolverPlan, SolverProvider)> {
        match self.selection {
            CommonLinearSelection::Exact { plan, provider } => Some((plan, provider)),
            CommonLinearSelection::ProgramControlled(_) => None,
        }
    }
}

/// Closed linear or Newton/linear hierarchy with explicit solver intent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommonSolvePolicy {
    /// One linear solve for a steady linearized problem.
    Linear(CommonLinearRequest),
    /// One bounded Newton policy owning its explicit nested linear request.
    Newton {
        /// Nonlinear convergence/globalization policy.
        nonlinear: NonlinearSolvePlan,
        /// Exact or program-controlled linear request.
        linear: CommonLinearRequest,
    },
}

/// Pressure representative retained by an admitted transient Plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommonPressureGauge2d {
    /// Pressure is represented by one zero-integral constraint.
    ZeroIntegral,
    /// Natural traction determines the absolute pressure representative.
    BoundaryTraction,
}

/// Closed result of Model-first common numerical resolution.
///
/// This is the single native sum for every admitted common Plan. Adapters may
/// project it, but must not create a parallel Plan-kind authority.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedCommonPlan {
    /// Finite scalar affine physical Plan.
    Algebraic(Box<CommonAlgebraicPlan>),
    /// Explicit no-Mesh ODE Plan.
    Ode(Box<CommonOdePlan>),
    /// Scalar elliptic spatial Plan.
    Scalar(Box<CommonScalarPlan>),
    /// Linear-elasticity spatial Plan.
    Elasticity(Box<CommonElasticityPlan>),
    /// Steady incompressible-flow spatial Plan.
    SteadyStokes(Box<CommonSteadyStokesPlan>),
    /// Transient incompressible-flow spatial Plan.
    TransientFlow(Box<CommonTransientFlowPlan>),
    /// Fixed-reference fluid-structure interaction spatial Plan.
    Fsi(Box<CommonFsiPlan>),
}

impl ResolvedCommonPlan {
    /// Inspect the effective mathematical Formulation when the admitted
    /// capability owns one of the current proof-carrying consumers.
    #[must_use]
    pub fn formulation(&self) -> Option<CommonFormulationDescription> {
        match self {
            Self::Scalar(plan) => plan.formulation(),
            Self::SteadyStokes(plan) => Some(plan.formulation()),
            Self::TransientFlow(plan) => Some(plan.formulation()),
            Self::Algebraic(_) | Self::Ode(_) | Self::Elasticity(_) | Self::Fsi(_) => None,
        }
    }
}

/// Opaque native fixed-reference FSI Plan owning exact common resources.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonFsiPlan {
    recognized: RecognizedNativeAdmission,
    partition: FixedReferenceFsiPartition<2>,
    resolved: ResolvedCoupledFieldwiseRealization,
    portable: PortableRealizationGraph,
    scaling: FixedReferenceFsiScaleProfile2d,
    scaling_receipt: IncompressibleScalingReceipt2d,
    temporal: CommonBackwardEuler,
    linear: NativeLinearPolicy,
    lineage: CommonSpatialPlanLineage,
    field_ids: Vec<String>,
    domain_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct CommonSpatialPlanLineage {
    identity: String,
    model_id: String,
    model_revision: u64,
    resource_digests: ResourceDigests,
    realization_digest: String,
}

impl CommonSpatialPlanLineage {
    fn new(
        identity: String,
        model_id: String,
        model_revision: u64,
        resource_digests: ResourceDigests,
        realization_digest: String,
    ) -> Self {
        Self {
            identity,
            model_id,
            model_revision,
            resource_digests,
            realization_digest,
        }
    }

    fn identity(&self) -> &str {
        &self.identity
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    const fn model_revision(&self) -> u64 {
        self.model_revision
    }

    fn geometry_digest(&self) -> &str {
        &self.resource_digests.geometry
    }

    fn mesh_digest(&self) -> &str {
        &self.resource_digests.mesh
    }

    fn correspondence_digest(&self) -> &str {
        &self.resource_digests.correspondence
    }

    fn production_digest(&self) -> &str {
        &self.resource_digests.production
    }

    fn realization_digest(&self) -> &str {
        &self.realization_digest
    }
}

/// Resolve one canonical no-Mesh explicit ODE through the common native Plan sum.
pub fn resolve_common_ode_plan(
    model: &ModelEnvelope,
    kernel: &KernelProgram,
    temporal: CommonTsitouras45,
    backend: TimeBackendIdentity,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    CommonOdePlan::resolve(model, kernel, temporal, backend)
        .map(|plan| ResolvedCommonPlan::Ode(Box::new(plan)))
}

#[derive(Debug, Clone, PartialEq)]
enum CommonTransientResolvedSpatial {
    MiniP1(ResolvedTransientFieldwiseRealization),
    CellCentered(ResolvedTransientCellCenteredIncompressibleFlowRealization),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommonTransientFormulation {
    MixedGalerkin(Box<crate::form_compiler::vocabulary::MixedGalerkinCorrespondence>),
    IntegralConservative(Box<crate::form_compiler::vocabulary::IntegralConservativeCorrespondence>),
}

/// How formulation selection entered resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormulationSelectionMode {
    /// The versioned resolver selected the exact effective formulation.
    Automatic,
    /// The caller requested the exact effective formulation.
    Exact,
    /// Fresh source supplied one checked authored formulation.
    Authored,
}

impl FormulationSelectionMode {
    const fn identity(self) -> &'static [u8] {
        match self {
            Self::Automatic => b"automatic",
            Self::Exact => b"exact",
            Self::Authored => b"authored",
        }
    }
}

/// Inspectable mathematical form selected automatically for one common Plan.
///
/// Field roles and finite-dimensional spaces remain available from the
/// capability-specific Plan. This description owns only the mathematical
/// form, boundary treatment, closed transformation-rule inventory, and the
/// reason the resolver selected it; mesh, quadrature, numerical flux, solver,
/// provider, and placement remain Realization concerns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommonFormulationDescription {
    requested: FormulationSelectionMode,
    kind: FormulationKind,
    boundary_treatment: &'static str,
    rule_ids: Box<[&'static str]>,
    selection_reason_codes: Box<[&'static str]>,
    requested_source_identity: Option<String>,
}

impl CommonTransientFormulation {
    const fn identity(&self) -> &'static [u8] {
        match self {
            Self::MixedGalerkin(_) => b"mixed-galerkin-formulation/v1",
            Self::IntegralConservative(_) => b"integral-conservative-formulation/v1",
        }
    }
}

/// Opaque transient-flow Plan owning exact caller resources and numerical policy.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTransientFlowPlan {
    admission: NativeNumericalAdmission,
    resolved: CommonTransientResolvedSpatial,
    portable: PortableRealizationGraph,
    formulation: CommonTransientFormulation,
    formulation_selection: FormulationSelectionMode,
    scaling: ResolvedIncompressibleScaling2d,
    temporal: CommonBackwardEuler,
    nonlinear: NonlinearSolvePlan,
    lineage: CommonSpatialPlanLineage,
    velocity_field_id: String,
    pressure_field_id: String,
    velocity_space: Space,
    pressure_space: Space,
    gauge: CommonPressureGauge2d,
}

#[derive(Debug, Clone, PartialEq)]
enum CommonStateKind {
    Scalar(Box<[f64]>),
    MiniP1(Box<TransientNavierStokesInitialState2d>),
    CellCentered(Box<CellCenteredNavierStokesInitialState2d>),
    Fsi {
        state: Box<FixedReferenceFsiState<2>>,
        accepted: Option<Box<ResolvedFixedReferenceFsiSolution2d>>,
    },
}

/// Opaque coherent-SI state for one exact common transient state space.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonState {
    state_space_identity: String,
    identity: String,
    time_s: f64,
    model: Arc<ModelEnvelope>,
    resources: Arc<NativeMeshResources>,
    kind: CommonStateKind,
    named_boundary_forces_on_domain: Vec<(String, [f64; 2])>,
}

/// Canonical private execution request for one exact transient Plan and State.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonTransientRunRequest {
    plan: ResolvedCommonPlan,
    schedule: CommonRunSchedule,
}

/// Canonical common-worker request for one exact FSI Plan and State.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonFsiRunRequest {
    plan: CommonFsiPlan,
    schedule: CommonRunSchedule,
}

#[derive(Debug, Clone, PartialEq)]
struct CommonRunSchedule {
    state: CommonState,
    accepted_steps: NonZeroUsize,
    output_steps: Vec<usize>,
    identity: String,
}

/// Opaque common scalar Plan owning authenticated Model, Mesh, and policy state.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonScalarPlan {
    admission: NativeNumericalAdmission,
    portable: PortableRealizationGraph,
    formulation: Option<CommonFormulationDescription>,
    authored_formulation: Option<AuthoredFormulationProjection>,
    lineage: CommonSpatialPlanLineage,
    fields: Box<
        [(
            eqiora_core::Id<eqiora_core::entity::kinds::Field>,
            eqiora_core::ValueType,
        )],
    >,
    cells: Box<[usize]>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonScalarRunOutput {
    pub(crate) fields: Vec<(
        eqiora_core::Id<eqiora_core::entity::kinds::Field>,
        eqiora_core::ValueType,
        Vec<f64>,
    )>,
    pub(crate) solve_report: eqiora_solver::SolveReport,
    pub(crate) assembly_report: eqiora_assembly::AssemblyReport,
}

/// One accepted scalar Parameter point produced through an exact common Plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonScalarDifferentiationPoint {
    relation: AssembledLinearizedRelation,
    output: CartesianScalarFieldLinearization,
    receipt: ExecutionReceipt,
}

impl CommonScalarDifferentiationPoint {
    /// Consume the point into its relation, complete Field projection, and solve receipt.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        AssembledLinearizedRelation,
        CartesianScalarFieldLinearization,
        ExecutionReceipt,
    ) {
        (self.relation, self.output, self.receipt)
    }
}

/// Opaque linear-elasticity Plan owning exact Model, Mesh, and policy state.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonElasticityPlan {
    admission: NativeNumericalAdmission,
    portable: PortableRealizationGraph,
    lineage: CommonSpatialPlanLineage,
    displacement_field_id: String,
    cells: [usize; 2],
}

/// Opaque steady-Stokes Plan owning one authenticated exact-cylinder occurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonSteadyStokesPlan {
    admission: NativeNumericalAdmission,
    binding: SteadyStokesGeometryBinding2d,
    resolved: ResolvedFieldwiseRealization,
    portable: PortableRealizationGraph,
    formulation_selection: FormulationSelectionMode,
    scaling: ResolvedIncompressibleScaling2d,
    lineage: CommonSpatialPlanLineage,
    velocity_field_id: String,
    pressure_field_id: String,
    velocity_space: Space,
    pressure_space: Space,
}

/// Plan-authenticated scientific observations for one common steady-Stokes solve.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonSteadyStokesObservation {
    pressure_minimum: f64,
    pressure_maximum: f64,
    exact_bounds: [[f64; 2]; 2],
    cylinder_force_on_fluid: [f64; 2],
    inlet_flux: f64,
    outlet_flux: f64,
    net_flux: f64,
    constrained_reaction: [f64; 2],
    integrated_body_force: [f64; 2],
    integrated_boundary_traction: [f64; 2],
    momentum_closure: [f64; 2],
    continuity_residual_norm: f64,
}

/// Exact paired output produced by one common steady-Stokes Plan execution.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonSteadyStokesRunOutput {
    plan_identity: String,
    solution: crate::fluid::SteadyStokesMiniSolution2d,
    observation: CommonSteadyStokesObservation,
}

impl CommonSteadyStokesRunOutput {
    #[must_use]
    pub(crate) fn plan_identity(&self) -> &str {
        &self.plan_identity
    }

    #[must_use]
    pub(crate) fn into_parts(
        self,
    ) -> (
        crate::fluid::SteadyStokesMiniSolution2d,
        CommonSteadyStokesObservation,
    ) {
        (self.solution, self.observation)
    }
}

impl CommonSteadyStokesObservation {
    #[must_use]
    pub(crate) const fn pressure_minimum(&self) -> f64 {
        self.pressure_minimum
    }
    #[must_use]
    pub(crate) const fn pressure_maximum(&self) -> f64 {
        self.pressure_maximum
    }
    #[must_use]
    pub(crate) const fn exact_bounds(&self) -> [[f64; 2]; 2] {
        self.exact_bounds
    }
    #[must_use]
    pub(crate) const fn cylinder_force_on_fluid(&self) -> [f64; 2] {
        self.cylinder_force_on_fluid
    }
    #[must_use]
    pub(crate) const fn inlet_flux(&self) -> f64 {
        self.inlet_flux
    }
    #[must_use]
    pub(crate) const fn outlet_flux(&self) -> f64 {
        self.outlet_flux
    }
    #[must_use]
    pub(crate) const fn net_flux(&self) -> f64 {
        self.net_flux
    }
    #[must_use]
    pub(crate) const fn constrained_reaction(&self) -> [f64; 2] {
        self.constrained_reaction
    }
    #[must_use]
    pub(crate) const fn integrated_body_force(&self) -> [f64; 2] {
        self.integrated_body_force
    }
    #[must_use]
    pub(crate) const fn integrated_boundary_traction(&self) -> [f64; 2] {
        self.integrated_boundary_traction
    }
    #[must_use]
    pub(crate) const fn momentum_closure(&self) -> [f64; 2] {
        self.momentum_closure
    }
    #[must_use]
    pub(crate) const fn continuity_residual_norm(&self) -> f64 {
        self.continuity_residual_norm
    }
}

/// Plan-authenticated scientific observations for one common elasticity solve.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonElasticityObservation {
    constrained_reaction: [f64; 2],
    integrated_body_force: [f64; 2],
    exact_bounds: [[f64; 2]; 2],
}

/// Exact paired output produced by one common elasticity Plan execution.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonElasticityRunOutput {
    plan_identity: String,
    solution: CartesianLinearElasticity2dSolution,
    observation: CommonElasticityObservation,
}

impl CommonElasticityRunOutput {
    #[must_use]
    pub(crate) fn plan_identity(&self) -> &str {
        &self.plan_identity
    }

    #[must_use]
    pub(crate) fn into_parts(
        self,
    ) -> (
        CartesianLinearElasticity2dSolution,
        CommonElasticityObservation,
    ) {
        (self.solution, self.observation)
    }
}

impl CommonElasticityObservation {
    #[must_use]
    pub(crate) const fn constrained_reaction(&self) -> [f64; 2] {
        self.constrained_reaction
    }
    #[must_use]
    pub(crate) const fn integrated_body_force(&self) -> [f64; 2] {
        self.integrated_body_force
    }
    #[must_use]
    pub(crate) const fn exact_bounds(&self) -> [[f64; 2]; 2] {
        self.exact_bounds
    }
}

mod algebraic;
pub use algebraic::{CommonAlgebraicPlan, CommonAlgebraicState};
mod derived;
/// Resolve Model mathematics first, then admit the requested numerical policies.
mod elasticity;
mod formulation;
mod fsi;
pub use fsi::{CommonFsiConnectionInventory, CommonFsiDomainInventory};
mod mesh_artifact;
mod native;
mod plan_artifact;
mod resolve;
mod resolved;
mod scalar;
pub(super) use scalar::ExecutableScalarEquations;
mod solver_planning;
mod spatial_planning;
mod state;
mod state_artifact;
mod steady_stokes;
mod transient;
pub use native::AuthenticatedCommonMesh;
use native::*;
pub use resolve::resolve_common_plan;
#[cfg(test)]
mod tests;
