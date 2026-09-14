//! Canonical portable projection of a resolved Realization.
//!
//! Accepted typed plans remain authoring contracts. After their validators
//! succeed, this module lowers them into one small, typed DAG with a canonical,
//! bounded wire representation. It deliberately has no arbitrary node, edge,
//! payload, runtime handle, device ordinal, or allocation vocabulary.

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DynQuantity, Id, ScalarType};
use eqiora_solver::{LinearOperatorProperties, SolverPlan};
use std::num::NonZeroUsize;

use crate::{
    AleGeometryQualityGate, AlgebraicBlock, AlgebraicConstraint, CellCenteredConvectionScheme,
    Discretization, ExecutionSchedule, NonlinearSolvePlan, PositiveMomentumDiagonal,
    PositivePhysicalScale, RealizationLineage, Space, SymmetricCongruenceScaling, Target,
    TransientFaceFluxHistory, VectorLayoutKind, invalid_realization,
};

mod projection;
mod validation;
mod wire;

use projection::portable_placement;
use validation::{strictly_sorted_unique_by, validate_geometry_actions, validate_system};

macro_rules! graph_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(usize);

        impl $name {
            const fn new(index: usize) -> Self {
                Self(index)
            }

            /// Canonical arena position inside this graph projection.
            #[must_use]
            pub const fn index(self) -> usize {
                self.0
            }
        }
    };
}

graph_id!(
    /// Typed reference to one Domain discretization node.
    DomainDiscretizationId
);
graph_id!(
    /// Typed reference to one Field representation node.
    FieldRepresentationId
);
graph_id!(
    /// Typed reference to one sealed moving-geometry action.
    GeometryActionId
);
graph_id!(
    /// Typed reference to one numerical transformation node.
    TransformationId
);
graph_id!(
    /// Typed reference to one algebraic-system node.
    AlgebraicSystemId
);
graph_id!(
    /// Typed reference to one linear-solve node.
    LinearSolveId
);
graph_id!(
    /// Typed reference to one nonlinear-solve node.
    NonlinearSolveId
);
graph_id!(
    /// Typed reference to one portable placement requirement.
    PlacementRequirementId
);

/// How coordinates enter one portable spatial discretization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateTreatment {
    /// Coordinates are consumed in the Model's declared reference values.
    Physical,
    /// Coordinates are normalized by one positive coherent-SI length.
    Scaled(PositivePhysicalScale),
}

/// Physical configuration in which one Domain-local weak action is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainConfiguration {
    /// Existing fixed-geometry projection with no moving-configuration role.
    FixedGeometry,
    /// Immutable material/reference configuration.
    ReferenceConfiguration,
    /// A Domain is evaluated through one exact current ALE geometry action.
    CurrentAleGeometry {
        /// Sole action deriving current coordinates and mesh kinematics.
        action: GeometryActionId,
    },
}

/// One exact Semantic Domain and its portable spatial selection.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainDiscretizationNode {
    domain: Id<kinds::Domain>,
    coordinates: CoordinateTreatment,
    configuration: DomainConfiguration,
    discretization: Discretization,
}

impl DomainDiscretizationNode {
    /// Exact Semantic Domain.
    #[must_use]
    pub const fn domain(&self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Physical or explicitly scaled coordinate treatment.
    #[must_use]
    pub const fn coordinates(&self) -> CoordinateTreatment {
        self.coordinates
    }

    /// Fixed, reference, or current-ALE physical configuration.
    #[must_use]
    pub const fn configuration(&self) -> DomainConfiguration {
        self.configuration
    }

    /// Method, mesh policy, and quadrature choice.
    #[must_use]
    pub const fn discretization(&self) -> Discretization {
        self.discretization
    }
}

/// One exact Semantic Field represented in one Domain-local discrete space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldRepresentationNode {
    domain: DomainDiscretizationId,
    field: Id<kinds::Field>,
    space: Space,
}

impl FieldRepresentationNode {
    /// Domain-discretization dependency.
    #[must_use]
    pub const fn domain(self) -> DomainDiscretizationId {
        self.domain
    }

    /// Exact Semantic Field.
    #[must_use]
    pub const fn field(self) -> Id<kinds::Field> {
        self.field
    }

    /// Scalar basis applied to every semantic component.
    #[must_use]
    pub const fn space(self) -> Space {
        self.space
    }
}

/// One sealed moving-geometry action in the portable Realization DAG.
///
/// Coordinates, mesh velocity, its spatial gradient, and geometric-
/// conservation data are projections of this node.  They cannot be supplied
/// as independent graph inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeometryActionNode {
    /// Component-wise P1 harmonic extension on immutable reference topology.
    P1HarmonicExtension {
        /// Fluid Domain receiving the current ALE geometry.
        fluid_domain: DomainDiscretizationId,
        /// Solid Domain retained in its reference configuration.
        solid_domain: DomainDiscretizationId,
        /// Absolute solid-displacement Field driving the interface geometry.
        driver: FieldRepresentationId,
        /// Exact conforming FSI Connection defining the moving interface.
        interface: Id<kinds::Connection>,
        /// Shared physical duration used to derive mesh velocity.
        duration: DynQuantity,
        /// Trial and accepted geometry quality gate.
        quality_gate: AleGeometryQualityGate,
        /// Solver policy for the symmetric-positive-definite harmonic action.
        solver: SolverPlan,
    },
}

/// Portable numerical transformation selected over exact semantic identities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformationNode {
    /// Replace one exact Relation derivative by a backward difference.
    BackwardEulerDerivative {
        /// Exact Semantic Relation.
        relation: Id<kinds::Relation>,
        /// Differential Field representation.
        state: FieldRepresentationId,
        /// Positive physical step duration.
        duration: DynQuantity,
    },
    /// Use the energy-skew weak form for one conservative convective term.
    EnergySkewConvection {
        /// Exact Semantic Relation.
        relation: Id<kinds::Relation>,
        /// Velocity Field used in both convective slots.
        velocity: FieldRepresentationId,
    },
    /// Apply one exact cell-centered convection reconstruction in time and space.
    CellCenteredConvection {
        /// Exact conservative Semantic Relation.
        relation: Id<kinds::Relation>,
        /// Cell-centered transported Field representation.
        state: FieldRepresentationId,
        /// Exact endpoint or previous-state reconstruction policy.
        scheme: CellCenteredConvectionScheme,
    },
    /// Use orthogonal two-point diffusive fluxes on cell-centered faces.
    OrthogonalTwoPointDiffusion {
        /// Exact conservative Semantic Relation.
        relation: Id<kinds::Relation>,
        /// Cell-centered transported Field representation.
        state: FieldRepresentationId,
    },
    /// Use the unique collocated face flux for implicit centered momentum convection.
    ImplicitCenteredMomentumConvection {
        /// Exact conservative momentum Relation.
        relation: Id<kinds::Relation>,
        /// Cell-centered velocity representation.
        velocity: FieldRepresentationId,
    },
    /// Apply centered Cartesian Newtonian velocity--pressure face traction.
    CartesianCentralNewtonianTraction {
        /// Exact conservative momentum Relation.
        relation: Id<kinds::Relation>,
        /// Cell-centered velocity representation.
        velocity: FieldRepresentationId,
        /// Cell-centered pressure representation.
        pressure: FieldRepresentationId,
    },
    /// Share one linearly exact momentum-weighted face flux between equations.
    MomentumWeightedLinearExactCoupling {
        /// Exact conservative momentum Relation.
        momentum_relation: Id<kinds::Relation>,
        /// Exact incompressibility Relation.
        incompressibility_relation: Id<kinds::Relation>,
        /// Cell-centered velocity representation.
        velocity: FieldRepresentationId,
        /// Cell-centered pressure representation.
        pressure: FieldRepresentationId,
        /// Exact positive momentum scale used by face interpolation.
        positive_diagonal: PositiveMomentumDiagonal,
        /// Exact previous-time face-flux closure.
        transient_history: TransientFaceFluxHistory,
    },
    /// Pull back one conservative fluid action through a sealed moving geometry.
    ///
    /// Relative energy-skew convection and the endpoint differential GCL
    /// correction are one transformation and cannot be selected separately.
    GclCompatibleAlePullback {
        /// Exact conservative transient fluid Relation.
        relation: Id<kinds::Relation>,
        /// Physical fluid velocity represented on the moving Domain.
        velocity: FieldRepresentationId,
        /// Sole source of current maps and mesh kinematics.
        geometry: GeometryActionId,
    },
    /// Eliminate one state in favour of its exact rate by backward Euler.
    BackwardEulerElimination {
        /// Exact kinematic Relation being transformed.
        relation: Id<kinds::Relation>,
        /// Represented state omitted from the algebraic unknowns.
        state: FieldRepresentationId,
        /// Algebraic rate retained in the system.
        rate: FieldRepresentationId,
        /// Positive physical step duration.
        duration: DynQuantity,
        /// Characteristic physical scale for state reconstruction.
        state_scale: PositivePhysicalScale,
    },
    /// Identify two exact conforming traces through one Semantic Connection.
    ConformingTraceQuotient {
        /// Exact conserving Connection selected by the lowerer.
        connection: Id<kinds::Connection>,
        /// Canonically ordered cross-Domain Field representations.
        endpoints: [FieldRepresentationId; 2],
    },
}

/// One unknown block in a monolithic algebraic system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemBlock {
    /// Coefficients of one represented Semantic Field.
    Field(FieldRepresentationId),
    /// Multiplier introduced by one exact algebraic constraint.
    ConstraintMultiplier(AlgebraicConstraint),
}

/// Scaling treatment of one portable algebraic system.
#[derive(Debug, Clone, PartialEq)]
pub enum SystemScaling {
    /// The compatibility path acts directly in its declared scalar values.
    Dimensional,
    /// One explicit physical-to-dimensionless symmetric congruence.
    SymmetricCongruence(SymmetricCongruenceScaling),
}

/// One connected algebraic operator selected by a portable Realization.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgebraicSystemNode {
    blocks: Vec<SystemBlock>,
    transformations: Vec<TransformationId>,
    scaling: SystemScaling,
    operator_properties: LinearOperatorProperties,
    scalar_type: ScalarType,
    partition: VectorLayoutKind,
}

impl AlgebraicSystemNode {
    /// Canonical monolithic block order.
    #[must_use]
    pub fn blocks(&self) -> &[SystemBlock] {
        &self.blocks
    }

    /// Transformations contributing to this system.
    #[must_use]
    pub fn transformations(&self) -> &[TransformationId] {
        &self.transformations
    }

    /// Exact whole-system congruence scaling.
    #[must_use]
    pub const fn scaling(&self) -> &SystemScaling {
        &self.scaling
    }

    /// Explicit symmetric congruence, when selected by this system.
    #[must_use]
    pub const fn congruence_scaling(&self) -> Option<&SymmetricCongruenceScaling> {
        match &self.scaling {
            SystemScaling::Dimensional => None,
            SystemScaling::SymmetricCongruence(scaling) => Some(scaling),
        }
    }

    /// Mathematical property asserted for this realized operator.
    #[must_use]
    pub const fn operator_properties(&self) -> LinearOperatorProperties {
        self.operator_properties
    }

    /// Scalar storage and arithmetic policy.
    #[must_use]
    pub const fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }

    /// Replicated or explicitly partitioned global algebra.
    #[must_use]
    pub const fn partition(&self) -> VectorLayoutKind {
        self.partition
    }
}

/// Portable compute requirement; an observed machine is bound later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementRequirementNode {
    /// Host execution with an exact worker bound per partition.
    HostWorkers {
        /// Non-zero requested host workers.
        workers_per_partition: NonZeroUsize,
    },
    /// CUDA execution with a device count per partition, never an ordinal.
    CudaDevices {
        /// Non-zero requested devices for every partition.
        devices_per_partition: NonZeroUsize,
    },
}

impl PlacementRequirementNode {
    /// Requested host-worker count, when this is a host placement.
    #[must_use]
    pub const fn host_workers_per_partition(self) -> Option<NonZeroUsize> {
        match self {
            Self::HostWorkers {
                workers_per_partition,
            } => Some(workers_per_partition),
            Self::CudaDevices { .. } => None,
        }
    }

    /// Requested CUDA-device count, when this is a device placement.
    #[must_use]
    pub const fn cuda_devices_per_partition(self) -> Option<NonZeroUsize> {
        match self {
            Self::HostWorkers { .. } => None,
            Self::CudaDevices {
                devices_per_partition,
            } => Some(devices_per_partition),
        }
    }
}

/// One linear solver role attached to one algebraic system and placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearSolveNode {
    system: AlgebraicSystemId,
    plan: SolverPlan,
    placement: PlacementRequirementId,
    schedule: ExecutionSchedule,
}

impl LinearSolveNode {
    /// Algebraic system being solved.
    #[must_use]
    pub const fn system(self) -> AlgebraicSystemId {
        self.system
    }

    /// Sole linear-solver policy.
    #[must_use]
    pub const fn plan(self) -> SolverPlan {
        self.plan
    }

    /// Portable compute requirement.
    #[must_use]
    pub const fn placement(self) -> PlacementRequirementId {
        self.placement
    }

    /// Deployment scheduling requirement, separate from model time.
    #[must_use]
    pub const fn schedule(self) -> ExecutionSchedule {
        self.schedule
    }
}

/// One nonlinear solve whose linearization uses an explicit linear-solve node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearSolveNode {
    residual_system: AlgebraicSystemId,
    linearization: LinearSolveId,
    plan: NonlinearSolvePlan,
}

impl NonlinearSolveNode {
    /// Nonlinear residual system.
    #[must_use]
    pub const fn residual_system(self) -> AlgebraicSystemId {
        self.residual_system
    }

    /// Linear solve used for every admitted linearization.
    #[must_use]
    pub const fn linearization(self) -> LinearSolveId {
        self.linearization
    }

    /// Nonlinear convergence and globalization policy.
    #[must_use]
    pub const fn plan(self) -> NonlinearSolvePlan {
        self.plan
    }
}

/// Sole accepted solve root of one connected Phase-A graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveRoot {
    /// A linear solve is the accepted output boundary.
    Linear(LinearSolveId),
    /// A nonlinear solve is the accepted output boundary.
    Nonlinear(NonlinearSolveId),
}

/// Canonical portable DAG projected from one already resolved Realization.
///
/// Its canonical wire preserves every graph family currently projected by this
/// crate. Graph validation proves structural closure and solver compatibility.
/// An equation-aware consumer must additionally compare all claimed Semantic
/// identities and mathematical properties with its accepted lowering before
/// this graph can authorize a run or become evidence. Model, Mesh, and provider
/// payloads remain separate content-addressed dependencies.
#[derive(Debug, Clone, PartialEq)]
pub struct PortableRealizationGraph {
    lineage: RealizationLineage,
    domains: Vec<DomainDiscretizationNode>,
    fields: Vec<FieldRepresentationNode>,
    geometry_actions: Vec<GeometryActionNode>,
    transformations: Vec<TransformationNode>,
    systems: Vec<AlgebraicSystemNode>,
    linear_solves: Vec<LinearSolveNode>,
    nonlinear_solves: Vec<NonlinearSolveNode>,
    placements: Vec<PlacementRequirementNode>,
    root: SolveRoot,
}

impl PortableRealizationGraph {
    /// Resolve one graph-native dimensional linear realization for exact Field bindings.
    /// The equation-aware caller supplies the exact Semantic identities and
    /// operator class. This constructor owns graph closure and solver
    /// compatibility; provider capability admission remains a separate step.
    /// # Errors
    /// Returns `EQ0807` when the supplied choices cannot form one connected
    /// portable linear-solve graph.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_fields(
        lineage: RealizationLineage,
        domain: Id<kinds::Domain>,
        bindings: impl IntoIterator<Item = crate::FieldSpaceBinding>,
        discretization: Discretization,
        operator_properties: LinearOperatorProperties,
        scalar_type: ScalarType,
        vector_layout: VectorLayoutKind,
        solver: SolverPlan,
        target: Target,
        schedule: ExecutionSchedule,
    ) -> Result<Self, Diagnostic> {
        let mut bindings = bindings.into_iter().collect::<Vec<_>>();
        bindings.sort_by_key(|binding| binding.field().ulid());
        if bindings.is_empty()
            || bindings
                .windows(2)
                .any(|pair| pair[0].field() == pair[1].field())
        {
            return Err(invalid_realization(
                "linear graph requires nonempty unique Field bindings",
            ));
        }
        for binding in &bindings {
            discretization.validate_space(binding.space())?;
        }
        let blocks = (0..bindings.len())
            .map(|index| SystemBlock::Field(FieldRepresentationId::new(index)))
            .collect();
        crate::execution::validate_target_schedule(target, schedule)?;
        let graph = Self {
            lineage,
            domains: vec![DomainDiscretizationNode {
                domain,
                coordinates: CoordinateTreatment::Physical,
                configuration: DomainConfiguration::FixedGeometry,
                discretization,
            }],
            fields: bindings
                .into_iter()
                .map(|binding| FieldRepresentationNode {
                    domain: DomainDiscretizationId::new(0),
                    field: binding.field(),
                    space: binding.space(),
                })
                .collect(),
            geometry_actions: Vec::new(),
            transformations: Vec::new(),
            systems: vec![AlgebraicSystemNode {
                blocks,
                transformations: Vec::new(),
                scaling: SystemScaling::Dimensional,
                operator_properties,
                scalar_type,
                partition: vector_layout,
            }],
            linear_solves: vec![LinearSolveNode {
                system: AlgebraicSystemId::new(0),
                plan: solver,
                placement: PlacementRequirementId::new(0),
                schedule,
            }],
            nonlinear_solves: Vec::new(),
            placements: vec![portable_placement(target)],
            root: SolveRoot::Linear(LinearSolveId::new(0)),
        };
        graph.validate()?;
        Ok(graph)
    }

    /// Exact Model and Realization lineage.
    #[must_use]
    pub const fn lineage(&self) -> RealizationLineage {
        self.lineage
    }

    /// Canonically ordered Domain discretizations.
    #[must_use]
    pub fn domains(&self) -> &[DomainDiscretizationNode] {
        &self.domains
    }

    /// Canonically ordered Field representations.
    #[must_use]
    pub fn fields(&self) -> &[FieldRepresentationNode] {
        &self.fields
    }

    /// Canonically ordered sealed moving-geometry actions.
    #[must_use]
    pub fn geometry_actions(&self) -> &[GeometryActionNode] {
        &self.geometry_actions
    }

    /// Canonically ordered numerical transformations.
    #[must_use]
    pub fn transformations(&self) -> &[TransformationNode] {
        &self.transformations
    }

    /// Connected algebraic systems.
    #[must_use]
    pub fn systems(&self) -> &[AlgebraicSystemNode] {
        &self.systems
    }

    /// Linear-solver roles.
    #[must_use]
    pub fn linear_solves(&self) -> &[LinearSolveNode] {
        &self.linear_solves
    }

    /// Nonlinear-solver roles.
    #[must_use]
    pub fn nonlinear_solves(&self) -> &[NonlinearSolveNode] {
        &self.nonlinear_solves
    }

    /// Portable compute requirements without environment-local ordinals.
    #[must_use]
    pub fn placements(&self) -> &[PlacementRequirementNode] {
        &self.placements
    }

    /// Sole connected solve root.
    #[must_use]
    pub const fn root(&self) -> SolveRoot {
        self.root
    }

    /// Resolve a typed Domain reference.
    #[must_use]
    pub fn domain(&self, id: DomainDiscretizationId) -> Option<&DomainDiscretizationNode> {
        self.domains.get(id.index())
    }

    /// Resolve a typed Field reference.
    #[must_use]
    pub fn field(&self, id: FieldRepresentationId) -> Option<&FieldRepresentationNode> {
        self.fields.get(id.index())
    }

    /// Resolve a typed moving-geometry action reference.
    #[must_use]
    pub fn geometry_action(&self, id: GeometryActionId) -> Option<&GeometryActionNode> {
        self.geometry_actions.get(id.index())
    }

    /// Resolve a typed transformation reference.
    #[must_use]
    pub fn transformation(&self, id: TransformationId) -> Option<&TransformationNode> {
        self.transformations.get(id.index())
    }

    /// Resolve a typed algebraic-system reference.
    #[must_use]
    pub fn system(&self, id: AlgebraicSystemId) -> Option<&AlgebraicSystemNode> {
        self.systems.get(id.index())
    }

    /// Resolve a typed linear-solve reference.
    #[must_use]
    pub fn linear_solve(&self, id: LinearSolveId) -> Option<&LinearSolveNode> {
        self.linear_solves.get(id.index())
    }

    /// Resolve a typed nonlinear-solve reference.
    #[must_use]
    pub fn nonlinear_solve(&self, id: NonlinearSolveId) -> Option<&NonlinearSolveNode> {
        self.nonlinear_solves.get(id.index())
    }

    /// Resolve a typed placement reference.
    #[must_use]
    pub fn placement(&self, id: PlacementRequirementId) -> Option<PlacementRequirementNode> {
        self.placements.get(id.index()).copied()
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        if self.domains.is_empty() || self.fields.is_empty() {
            return Err(invalid_realization(
                "portable Realization graph requires a Domain and a Field representation",
            ));
        }
        if !strictly_sorted_unique_by(&self.domains, |node| node.domain.ulid()) {
            return Err(invalid_realization(
                "portable Realization Domain nodes must be unique and canonically ordered",
            ));
        }
        if !strictly_sorted_unique_by(&self.fields, |node| node.field.ulid()) {
            return Err(invalid_realization(
                "portable Realization Field nodes must be unique and canonically ordered",
            ));
        }
        for field in &self.fields {
            if self.domain(field.domain).is_none() {
                return Err(invalid_realization(
                    "portable Realization Field references an absent Domain node",
                ));
            }
        }
        validate_geometry_actions(self)?;
        if self.systems.len() != 1
            || self.linear_solves.len() != 1
            || self.placements.len() != 1
            || self.nonlinear_solves.len() > 1
        {
            return Err(invalid_realization(
                "portable Realization Phase A admits one connected system, linear solve, placement, and optional nonlinear root",
            ));
        }
        let system = &self.systems[0];
        validate_system(self, system)?;
        let linear = self.linear_solves[0];
        if linear.system != AlgebraicSystemId::new(0)
            || linear.placement != PlacementRequirementId::new(0)
        {
            return Err(invalid_realization(
                "portable linear solve must reference the connected system and placement",
            ));
        }
        if !linear.plan.algorithm().accepts(system.operator_properties) {
            return Err(invalid_realization(
                "portable linear solver algorithm is incompatible with the asserted operator properties",
            ));
        }
        match self.root {
            SolveRoot::Linear(id) => {
                if id != LinearSolveId::new(0) || !self.nonlinear_solves.is_empty() {
                    return Err(invalid_realization(
                        "portable linear root leaves an unreachable solve node",
                    ));
                }
            }
            SolveRoot::Nonlinear(id) => {
                let Some(nonlinear) = self.nonlinear_solve(id) else {
                    return Err(invalid_realization(
                        "portable nonlinear root references an absent node",
                    ));
                };
                if self.nonlinear_solves.len() != 1
                    || nonlinear.residual_system != AlgebraicSystemId::new(0)
                    || nonlinear.linearization != LinearSolveId::new(0)
                {
                    return Err(invalid_realization(
                        "portable nonlinear root must own the sole residual system and linearization",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "portable_graph/tests.rs"]
mod tests;
