use eqiora_assembly::AssemblyReport;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, OntologyId};
use eqiora_realization::{
    CellCenteredConvectionScheme, RealizationRevision,
    ResolvedTransientCellCenteredTransportRealization, SemanticRevision,
};
use eqiora_schema::Model;
use eqiora_solver::{LinearProblem, LinearSolution, SolveReport, SolverPlan};

use super::reconstruction::AffineFaceTrace;
use crate::finalized_spatial::FinalizedLinearCore;
use eqiora_meshing::CartesianMesh;

/// Physical role derived from canonical boundary meaning and outward velocity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarTransportBoundaryRole {
    /// Negative outward volume flux with an exact prescribed trace.
    Inflow,
    /// Positive outward volume flux with an exact prescribed diffusive flux.
    Outflow,
    /// Zero outward volume flux with an exact prescribed diffusive flux.
    ImpermeableWall,
}

/// One accepted cell-centered state at a physical model time.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarTransportCellState2d {
    pub(super) model: OntologyId<Model>,
    pub(super) semantic_revision: SemanticRevision,
    pub(super) realization_revision: RealizationRevision,
    pub(super) field: Id<kinds::Field>,
    pub(super) mesh: CartesianMesh,
    pub(super) time: DynQuantity,
    pub(super) value_dimension: DimExponents,
    pub(super) values: Vec<f64>,
}

impl ScalarTransportCellState2d {
    /// Exact Semantic Model owning this state.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Exact Semantic Model revision captured before initialization.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Exact Realization revision whose mesh and space carry the state.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.realization_revision
    }

    /// Exact transported Semantic Field.
    #[must_use]
    pub const fn field(&self) -> Id<kinds::Field> {
        self.field
    }

    /// Generated Cartesian mesh carrying the cell values.
    #[must_use]
    pub const fn mesh(&self) -> &CartesianMesh {
        &self.mesh
    }

    /// Dimensioned physical model time.
    #[must_use]
    pub const fn time(&self) -> DynQuantity {
        self.time
    }

    /// Coherent-SI physical dimension of every cell value.
    #[must_use]
    pub const fn value_dimension(&self) -> DimExponents {
        self.value_dimension
    }

    /// Cell values in canonical top-cell order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Inspectable acceptance evidence for one typed Euler-family FVM step.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarTransportFvmStepEvidence2d {
    pub(super) convection_scheme: CellCenteredConvectionScheme,
    pub(super) maximum_courant_number: f64,
    pub(super) limited_face_count: usize,
    pub(super) advective_face_value_range: Option<[f64; 2]>,
    pub(super) advective_face_bound_defect: f64,
    pub(super) advective_face_bound_tolerance: Option<f64>,
    pub(super) old_mass: f64,
    pub(super) new_mass: f64,
    pub(super) outward_boundary_flux: f64,
    pub(super) conservation_defect: f64,
    pub(super) conservation_tolerance: f64,
    pub(super) minimum_value: f64,
    pub(super) maximum_value: f64,
    pub(super) replayed_residual_norm: f64,
    pub(super) maximum_assembly_replay_defect: f64,
    pub(super) assembly_replay_tolerance: f64,
    pub(super) maximum_operator_replay_defect: f64,
    pub(super) operator_replay_tolerance: f64,
    pub(super) interior_face_count: usize,
    pub(super) periodic_face_count: usize,
    pub(super) boundary_face_count: usize,
    pub(super) inflow_face_count: usize,
    pub(super) outflow_face_count: usize,
    pub(super) wall_face_count: usize,
    pub(super) maximum_interior_cancellation_defect: f64,
    pub(super) assembly_report: AssemblyReport,
    pub(super) solve_report: SolveReport,
}

impl ScalarTransportFvmStepEvidence2d {
    /// Exact convection treatment executed by this step.
    #[must_use]
    pub const fn convection_scheme(&self) -> CellCenteredConvectionScheme {
        self.convection_scheme
    }

    /// Largest explicit advective Courant number; zero for implicit upwind.
    #[must_use]
    pub const fn maximum_courant_number(&self) -> f64 {
        self.maximum_courant_number
    }

    /// Number of face reconstructions where minmod or hull limiting was active.
    #[must_use]
    pub const fn limited_face_count(&self) -> usize {
        self.limited_face_count
    }

    /// Smallest/largest active advective face states, or `None` for zero flow.
    #[must_use]
    pub const fn advective_face_value_range(&self) -> Option<[f64; 2]> {
        self.advective_face_value_range
    }

    /// Maximum violation of the previous-state plus boundary-trace hull.
    #[must_use]
    pub const fn advective_face_bound_defect(&self) -> f64 {
        self.advective_face_bound_defect
    }

    /// Roundoff bound used to accept limited face values, or `None` when the
    /// selected scheme makes no bounded-reconstruction claim.
    #[must_use]
    pub const fn advective_face_bound_tolerance(&self) -> Option<f64> {
        self.advective_face_bound_tolerance
    }

    /// Integrated transported quantity before the step.
    #[must_use]
    pub const fn old_mass(&self) -> f64 {
        self.old_mass
    }

    /// Integrated transported quantity after the step.
    #[must_use]
    pub const fn new_mass(&self) -> f64 {
        self.new_mass
    }

    /// Sum of final outward advective-minus-diffusive boundary fluxes.
    #[must_use]
    pub const fn outward_boundary_flux(&self) -> f64 {
        self.outward_boundary_flux
    }

    /// `(new_mass - old_mass) / dt + outward_boundary_flux`.
    #[must_use]
    pub const fn conservation_defect(&self) -> f64 {
        self.conservation_defect
    }

    /// Solver-aware finite-precision bound used to accept conservation.
    #[must_use]
    pub const fn conservation_tolerance(&self) -> f64 {
        self.conservation_tolerance
    }

    /// Minimum accepted cell value.
    #[must_use]
    pub const fn minimum_value(&self) -> f64 {
        self.minimum_value
    }

    /// Maximum accepted cell value.
    #[must_use]
    pub const fn maximum_value(&self) -> f64 {
        self.maximum_value
    }

    /// Norm from an independent replay of the finalized canonical CSR.
    #[must_use]
    pub const fn replayed_residual_norm(&self) -> f64 {
        self.replayed_residual_norm
    }

    /// Largest rowwise difference between finalized CSR and an independent
    /// physical cell/face residual reconstruction.
    #[must_use]
    pub const fn maximum_assembly_replay_defect(&self) -> f64 {
        self.maximum_assembly_replay_defect
    }

    /// Floating-point bound used to accept the independent assembly replay.
    #[must_use]
    pub const fn assembly_replay_tolerance(&self) -> f64 {
        self.assembly_replay_tolerance
    }

    /// Largest coefficient or right-hand-side difference between the captured
    /// CSR and an independent reconstruction of the complete physical operator.
    #[must_use]
    pub const fn maximum_operator_replay_defect(&self) -> f64 {
        self.maximum_operator_replay_defect
    }

    /// Floating-point bound used to accept the complete operator replay.
    #[must_use]
    pub const fn operator_replay_tolerance(&self) -> f64 {
        self.operator_replay_tolerance
    }

    /// Number of exactly-once interior face packets.
    #[must_use]
    pub const fn interior_face_count(&self) -> usize {
        self.interior_face_count
    }

    /// Number of exactly-once coupled packets derived from periodic pairs.
    #[must_use]
    pub const fn periodic_face_count(&self) -> usize {
        self.periodic_face_count
    }

    /// Number of exactly-once exterior face packets.
    #[must_use]
    pub const fn boundary_face_count(&self) -> usize {
        self.boundary_face_count
    }

    /// Number of boundary faces classified as inflow.
    #[must_use]
    pub const fn inflow_face_count(&self) -> usize {
        self.inflow_face_count
    }

    /// Number of boundary faces classified as outflow.
    #[must_use]
    pub const fn outflow_face_count(&self) -> usize {
        self.outflow_face_count
    }

    /// Number of boundary faces classified as impermeable walls.
    #[must_use]
    pub const fn wall_face_count(&self) -> usize {
        self.wall_face_count
    }

    /// Largest independently evaluated `F + (-F)` interior scatter defect.
    #[must_use]
    pub const fn maximum_interior_cancellation_defect(&self) -> f64 {
        self.maximum_interior_cancellation_defect
    }

    /// Complete ordered assembly evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Complete solver/backend/execution evidence.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }
}

/// One accepted state transition and its exact resolved Realization.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarTransportFvmStep2d {
    pub(super) realization: ResolvedTransientCellCenteredTransportRealization,
    pub(super) state: ScalarTransportCellState2d,
    pub(super) evidence: ScalarTransportFvmStepEvidence2d,
}

impl ScalarTransportFvmStep2d {
    /// Exact two-layer Realization admitted before assembly.
    #[must_use]
    pub const fn realization(&self) -> &ResolvedTransientCellCenteredTransportRealization {
        &self.realization
    }

    /// Accepted endpoint state.
    #[must_use]
    pub const fn state(&self) -> &ScalarTransportCellState2d {
        &self.state
    }

    /// Acceptance evidence for this step.
    #[must_use]
    pub const fn evidence(&self) -> &ScalarTransportFvmStepEvidence2d {
        &self.evidence
    }

    /// Consume the step and continue from its accepted endpoint.
    #[must_use]
    pub fn into_state(self) -> ScalarTransportCellState2d {
        self.state
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum TransportFace2d {
    Interior {
        lower: usize,
        upper: usize,
        outward_from_lower_flux: f64,
        transmissibility: f64,
        advective_trace: AffineFaceTrace,
    },
    PrescribedTrace {
        cell: usize,
        outward_volume_flux: f64,
        transmissibility: f64,
        trace: f64,
        advective_trace: AffineFaceTrace,
    },
    PrescribedDiffusiveFlux {
        cell: usize,
        outward_volume_flux: f64,
        diffusive_flux_integral: f64,
        role: ScalarTransportBoundaryRole,
        advective_trace: AffineFaceTrace,
    },
}

/// Finalized general linear operator plus method-private reconstruction state.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedScalarTransportFvmStep2d {
    pub(super) core: FinalizedLinearCore<f64>,
    pub(super) realization: ResolvedTransientCellCenteredTransportRealization,
    pub(super) mesh: CartesianMesh,
    pub(super) field: Id<kinds::Field>,
    pub(super) previous_time: DynQuantity,
    pub(super) value_dimension: DimExponents,
    pub(super) duration: f64,
    pub(super) previous_values: Vec<f64>,
    pub(super) cell_measures: Vec<f64>,
    pub(super) faces: Vec<TransportFace2d>,
    pub(super) periodic_face_count: usize,
    pub(super) state_scale: f64,
    pub(super) weak_scale: f64,
    pub(super) maximum_operator_replay_defect: f64,
    pub(super) operator_replay_tolerance: f64,
    pub(super) assembly_report: AssemblyReport,
    pub(super) convection_scheme: CellCenteredConvectionScheme,
    pub(super) maximum_courant_number: f64,
    pub(super) limited_face_count: usize,
    pub(super) advective_bounds: [f64; 2],
}

impl FinalizedScalarTransportFvmStep2d {
    /// Exact finalized general linear problem.
    ///
    /// # Errors
    /// Preserves the sealed canonical CSR validation invariant.
    pub fn linear_problem(&self) -> Result<LinearProblem<'_>, Diagnostic> {
        self.core.linear_problem()
    }

    /// Sole solver plan retained from the resolved portable graph.
    #[must_use]
    pub fn solver_plan(&self) -> SolverPlan {
        self.core.solver_plan()
    }

    /// Finish one backend solution and independently verify its conservation.
    ///
    /// # Errors
    /// Rejects solver-policy, orientation, or execution substitution; shape or
    /// non-finite reconstruction; and CSR, physical-assembly, or global-balance
    /// replay failure without producing an accepted state.
    pub fn finish(self, solved: LinearSolution) -> Result<ScalarTransportFvmStep2d, Diagnostic> {
        super::assembly::finish_step(self, solved)
    }
}
