use std::collections::{BTreeMap, BTreeSet};
use std::num::{NonZeroU64, NonZeroUsize};

use eqiora_core::{Diagnostic, ScalarType};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverCapabilities, SolverCapability, SolverPlan,
};

use crate::{
    CoupledFieldwiseRealizationPlan, DiscretizationMethod, ExecutionSchedule,
    FieldwiseRealizationPlan, MeshKind, RealizationPlan, Target, invalid_realization,
};

/// Execution data layout required by a realized problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VectorLayoutKind {
    /// A complete vector is resident in one process.
    Replicated,
    /// A global vector has explicit owned and ghost partitions.
    Distributed,
}

/// Model/lowering facts against which a Realization plan is admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealizationRequirements {
    spatial_dimension: NonZeroUsize,
    scalar_type: ScalarType,
    vector_layout: VectorLayoutKind,
}

impl RealizationRequirements {
    /// Construct explicit problem requirements.
    #[must_use]
    pub const fn new(
        spatial_dimension: NonZeroUsize,
        scalar_type: ScalarType,
        vector_layout: VectorLayoutKind,
    ) -> Self {
        Self {
            spatial_dimension,
            scalar_type,
            vector_layout,
        }
    }

    /// Spatial dimension of the admitted canonical lowerer/problem.
    #[must_use]
    pub const fn spatial_dimension(self) -> NonZeroUsize {
        self.spatial_dimension
    }

    /// Scalar representation.
    #[must_use]
    pub const fn scalar_type(self) -> ScalarType {
        self.scalar_type
    }

    /// Replicated or explicitly distributed vector data.
    #[must_use]
    pub const fn vector_layout(self) -> VectorLayoutKind {
        self.vector_layout
    }
}

/// Inclusive spatial-dimension capability of one complete lowering path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpatialDimensionSupport {
    minimum: NonZeroUsize,
    maximum: NonZeroUsize,
}

/// One target-family member of an exact realization-admission tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TargetCapability {
    /// Host execution up to one exact worker-capacity bound.
    HostCpu {
        /// Largest run-owned worker count admitted by this tuple.
        maximum_threads: NonZeroUsize,
    },
    /// One discovered CUDA device ordinal admitted by this tuple.
    CudaGpu {
        /// Deployment-local device ordinal.
        device: u16,
    },
}

impl TargetCapability {
    fn supports(self, target: Target) -> bool {
        match (self, target) {
            (Self::HostCpu { maximum_threads }, Target::HostCpu { threads }) => {
                threads <= maximum_threads
            }
            (Self::CudaGpu { device: admitted }, Target::CudaGpu { device }) => admitted == device,
            (Self::HostCpu { .. }, Target::CudaGpu { .. })
            | (Self::CudaGpu { .. }, Target::HostCpu { .. }) => false,
        }
    }
}

/// Deployment schedule request admitted by one exact tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScheduleCapability {
    /// Ordinary offline execution only.
    Offline,
    /// One exact real-time priority/deadline request.
    RealTime {
        /// Platform scheduling priority.
        priority: u16,
        /// Non-zero deployment deadline in nanoseconds.
        deadline_ns: NonZeroU64,
    },
}

impl ScheduleCapability {
    fn supports(self, schedule: ExecutionSchedule) -> bool {
        match (self, schedule) {
            (Self::Offline, ExecutionSchedule::Offline) => true,
            (
                Self::RealTime {
                    priority: admitted_priority,
                    deadline_ns: admitted_deadline,
                },
                ExecutionSchedule::RealTime {
                    priority,
                    deadline_ns,
                },
            ) => admitted_priority == priority && admitted_deadline == deadline_ns,
            _ => false,
        }
    }
}

impl SpatialDimensionSupport {
    /// Support exactly one dimension.
    #[must_use]
    pub const fn exact(dimension: NonZeroUsize) -> Self {
        Self {
            minimum: dimension,
            maximum: dimension,
        }
    }

    /// Construct an inclusive nonempty range.
    ///
    /// # Errors
    /// Returns `EQ0807` when the maximum is below the minimum.
    pub fn inclusive(minimum: NonZeroUsize, maximum: NonZeroUsize) -> Result<Self, Diagnostic> {
        if maximum < minimum {
            return Err(invalid_realization(
                "spatial-dimension capability maximum is below its minimum",
            ));
        }
        Ok(Self { minimum, maximum })
    }

    /// Whether a dimension is admitted.
    #[must_use]
    pub const fn supports(self, dimension: NonZeroUsize) -> bool {
        dimension.get() >= self.minimum.get() && dimension.get() <= self.maximum.get()
    }

    /// Inclusive minimum.
    #[must_use]
    pub const fn minimum(self) -> NonZeroUsize {
        self.minimum
    }

    /// Inclusive maximum.
    #[must_use]
    pub const fn maximum(self) -> NonZeroUsize {
        self.maximum
    }
}

/// Spatial portion of one exact realization-admission context.
///
/// Space family, order, quadrature, and other method-specific facts remain
/// owned by the corresponding typed plan validator. This value contains only
/// the spatial axes owned by the generic realization boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpatialCapability {
    method: DiscretizationMethod,
    mesh_kind: MeshKind,
    dimensions: SpatialDimensionSupport,
}

impl SpatialCapability {
    /// Construct the spatial portion of one admitted context.
    #[must_use]
    pub const fn new(
        method: DiscretizationMethod,
        mesh_kind: MeshKind,
        dimensions: SpatialDimensionSupport,
    ) -> Self {
        Self {
            method,
            mesh_kind,
            dimensions,
        }
    }

    /// Discretization method.
    #[must_use]
    pub const fn method(self) -> DiscretizationMethod {
        self.method
    }

    /// Mesh family.
    #[must_use]
    pub const fn mesh_kind(self) -> MeshKind {
        self.mesh_kind
    }

    /// Spatial-dimension envelope.
    #[must_use]
    pub const fn dimensions(self) -> SpatialDimensionSupport {
        self.dimensions
    }
}

/// Resolved execution targets available to one concrete backend environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetCapabilities {
    maximum_host_threads: Option<NonZeroUsize>,
    cuda_devices: BTreeSet<u16>,
}

impl TargetCapabilities {
    /// No executable target.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            maximum_host_threads: None,
            cuda_devices: BTreeSet::new(),
        }
    }

    /// Add an executable host-CPU target and its worker bound.
    #[must_use]
    pub fn with_host_cpu(mut self, maximum_threads: NonZeroUsize) -> Self {
        self.maximum_host_threads = Some(maximum_threads);
        self
    }

    /// Add one discovered executable CUDA device ordinal.
    #[must_use]
    pub fn with_cuda_device(mut self, device: u16) -> Self {
        self.cuda_devices.insert(device);
        self
    }

    fn exact_members(&self) -> BTreeSet<TargetCapability> {
        let mut members = self
            .cuda_devices
            .iter()
            .copied()
            .map(|device| TargetCapability::CudaGpu { device })
            .collect::<BTreeSet<_>>();
        if let Some(maximum_threads) = self.maximum_host_threads {
            members.insert(TargetCapability::HostCpu { maximum_threads });
        }
        members
    }
}

/// Solver-independent context of one exact realization admission.
///
/// A dimension range and host worker bound are envelopes inside one verified
/// context. Backend and device identity remain deployment/run provenance, not
/// artifact identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RealizationCapabilityContext {
    spatial: SpatialCapability,
    vector_layout: VectorLayoutKind,
    target: TargetCapability,
    schedule: ScheduleCapability,
}

impl RealizationCapabilityContext {
    /// Construct one solver-independent admission context.
    #[must_use]
    pub const fn new(
        spatial: SpatialCapability,
        vector_layout: VectorLayoutKind,
        target: TargetCapability,
        schedule: ScheduleCapability,
    ) -> Self {
        Self {
            spatial,
            vector_layout,
            target,
            schedule,
        }
    }

    /// Spatial capability.
    #[must_use]
    pub const fn spatial(self) -> SpatialCapability {
        self.spatial
    }

    /// Vector layout.
    #[must_use]
    pub const fn vector_layout(self) -> VectorLayoutKind {
        self.vector_layout
    }

    /// Target-family member.
    #[must_use]
    pub const fn target(self) -> TargetCapability {
        self.target
    }

    /// Exact deployment schedule request.
    #[must_use]
    pub const fn schedule(self) -> ScheduleCapability {
        self.schedule
    }
}

/// One exact admitted combination across the axes owned by Realization.
///
/// The generic admission boundary pairs its complete context with one exact
/// solver tuple. Method-specific validators still own spaces, order,
/// quadrature, and operator-specific structure; this is deliberately not a
/// second execution graph or a claim about those axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RealizationCapability {
    context: RealizationCapabilityContext,
    solver: SolverCapability,
}

impl RealizationCapability {
    /// Construct one exact admitted tuple.
    ///
    /// # Errors
    /// Returns `EQ0807` for a mathematically incompatible solver/property pair.
    pub fn new(
        context: RealizationCapabilityContext,
        solver: SolverCapability,
    ) -> Result<Self, Diagnostic> {
        if !solver.algorithm.accepts(solver.operator_properties) {
            return Err(invalid_realization(format!(
                "realization capability has an incompatible solver/property pair: {solver:?}",
            )));
        }
        Ok(Self { context, solver })
    }

    /// Solver-independent context.
    #[must_use]
    pub const fn context(self) -> RealizationCapabilityContext {
        self.context
    }

    /// Scalar representation, owned by the solver tuple.
    #[must_use]
    pub const fn scalar_type(self) -> ScalarType {
        self.solver.scalar_type
    }

    /// Exact solver policy in this path.
    #[must_use]
    pub const fn solver(self) -> SolverCapability {
        self.solver
    }
}

/// Capabilities of a concrete lowerer/backend pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationCapabilities {
    combinations: BTreeSet<RealizationCapability>,
}

impl RealizationCapabilities {
    /// Construct a nonempty set of exact Realization-owned admission tuples.
    ///
    /// # Errors
    /// Returns `EQ0807` when no exact tuple is supplied.
    pub fn exact(
        combinations: impl IntoIterator<Item = RealizationCapability>,
    ) -> Result<Self, Diagnostic> {
        let combinations = combinations.into_iter().collect::<BTreeSet<_>>();
        if combinations.is_empty() {
            return Err(invalid_realization(
                "realization capabilities require at least one exact admission tuple",
            ));
        }
        Ok(Self { combinations })
    }

    /// Represent a concrete backend environment with no executable target.
    ///
    /// This is an availability result, not an evidence-backed capability. It
    /// exists so target discovery can fail closed at ordinary resolution.
    #[must_use]
    pub const fn unavailable() -> Self {
        Self {
            combinations: BTreeSet::new(),
        }
    }

    /// Expand independent axes only when their complete Cartesian product is
    /// intentionally implemented.
    ///
    /// Prefer [`Self::exact`] whenever support differs by method, mesh,
    /// solver, layout, or target. The explicit name prevents a convenience
    /// constructor from silently widening an evidence-backed capability.
    ///
    /// # Errors
    /// Returns `EQ0807` for an empty semantic axis or duplicate mesh family.
    /// An empty discovered target inventory produces [`Self::unavailable`].
    #[allow(clippy::too_many_arguments)]
    pub fn cartesian_product(
        methods: impl IntoIterator<Item = DiscretizationMethod>,
        mesh_dimensions: impl IntoIterator<Item = (MeshKind, SpatialDimensionSupport)>,
        vector_layouts: impl IntoIterator<Item = VectorLayoutKind>,
        solver: SolverCapabilities,
        targets: TargetCapabilities,
    ) -> Result<Self, Diagnostic> {
        let mut mesh_dimension_map = BTreeMap::new();
        for (kind, support) in mesh_dimensions {
            if mesh_dimension_map.insert(kind, support).is_some() {
                return Err(invalid_realization(format!(
                    "realization capabilities contain duplicate {kind:?} mesh support",
                )));
            }
        }
        let methods = methods.into_iter().collect::<BTreeSet<_>>();
        let vector_layouts = vector_layouts.into_iter().collect::<BTreeSet<_>>();
        let targets = targets.exact_members();
        if methods.is_empty() || mesh_dimension_map.is_empty() || vector_layouts.is_empty() {
            return Err(invalid_realization(
                "Cartesian-product realization capabilities require a method, mesh kind, and vector layout",
            ));
        }
        let mut combinations = BTreeSet::new();
        for &method in &methods {
            for (&mesh_kind, &spatial_dimensions) in &mesh_dimension_map {
                for &solver in solver.combinations() {
                    for &vector_layout in &vector_layouts {
                        for &target in &targets {
                            let context = RealizationCapabilityContext::new(
                                SpatialCapability::new(method, mesh_kind, spatial_dimensions),
                                vector_layout,
                                target,
                                ScheduleCapability::Offline,
                            );
                            combinations.insert(RealizationCapability::new(context, solver)?);
                        }
                    }
                }
            }
        }
        if targets.is_empty() {
            return Ok(Self::unavailable());
        }
        Self::exact(combinations)
    }

    /// Exact Realization-owned admission tuples.
    #[must_use]
    pub const fn combinations(&self) -> &BTreeSet<RealizationCapability> {
        &self.combinations
    }

    fn from_spatial_profiles(
        profiles: impl IntoIterator<Item = (DiscretizationMethod, MeshKind, SpatialDimensionSupport)>,
        solver: &SolverCapabilities,
        vector_layout: VectorLayoutKind,
        target: TargetCapability,
    ) -> Result<Self, Diagnostic> {
        let mut combinations = BTreeSet::new();
        for (method, mesh_kind, spatial_dimensions) in profiles {
            for &solver in solver.combinations() {
                let context = RealizationCapabilityContext::new(
                    SpatialCapability::new(method, mesh_kind, spatial_dimensions),
                    vector_layout,
                    target,
                    ScheduleCapability::Offline,
                );
                combinations.insert(RealizationCapability::new(context, solver)?);
            }
        }
        Self::exact(combinations)
    }

    /// Capabilities of the current scalar-elliptic reference realization.
    ///
    /// This is deliberately limited to the one-, two-, and three-dimensional
    /// end-to-end verification envelope. Lower-level Cartesian topology,
    /// geometry, and discrete-space contracts remain runtime-dimensional.
    #[must_use]
    pub fn scalar_elliptic_reference() -> Self {
        let dimensions_1_to_3 = SpatialDimensionSupport::inclusive(
            NonZeroUsize::MIN,
            NonZeroUsize::new(3).expect("three is non-zero"),
        )
        .expect("one through three is a valid dimension range");
        let solver = SolverCapabilities::exact([
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Jacobi,
                reduction: ReductionPolicy::Reproducible,
                scalar_type: ScalarType::F64,
            },
        ])
        .expect("the scalar elliptic reference solver tuple is exact");
        Self::from_spatial_profiles(
            [
                (
                    DiscretizationMethod::ContinuousGalerkin,
                    MeshKind::GeneratedCartesian,
                    dimensions_1_to_3,
                ),
                (
                    DiscretizationMethod::CellCenteredFiniteVolume,
                    MeshKind::GeneratedCartesian,
                    dimensions_1_to_3,
                ),
                (
                    DiscretizationMethod::ContinuousGalerkin,
                    MeshKind::ImportedAffineSimplicial,
                    SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is non-zero")),
                ),
            ],
            &solver,
            VectorLayoutKind::Replicated,
            TargetCapability::HostCpu {
                maximum_threads: NonZeroUsize::MIN,
            },
        )
        .expect("reference capability profiles are exact and nonempty")
    }

    /// Exact execution envelope of the verified 2D cell-centered transport path.
    ///
    /// Method-specific time, convection, and diffusion capability remains in
    /// `TransientCellCenteredTransportCapabilities`; this ordinary envelope
    /// admits only generated Cartesian 2D FVM, replicated `f64`, reproducible
    /// Jacobi-preconditioned BiCGSTAB, and one host worker.
    #[must_use]
    pub fn cell_centered_transport_2d_reference() -> Self {
        let solver = SolverCapabilities::exact([SolverCapability {
            algorithm: LinearSolver::BiConjugateGradientStabilized,
            operator_properties: LinearOperatorProperties::General,
            preconditioner: PreconditionerPolicy::Jacobi,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        }])
        .expect("the transport reference solver tuple is exact and nonempty");
        Self::cartesian_product(
            [DiscretizationMethod::CellCenteredFiniteVolume],
            [(
                MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is non-zero")),
            )],
            [VectorLayoutKind::Replicated],
            solver,
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
        .expect("transport reference capability axes are exact and nonempty")
    }

    /// Capabilities of the verified 2D isotropic-elasticity reference path.
    ///
    /// The envelope is intentionally exact: continuous Q1 on a generated
    /// Cartesian mesh, replicated `f64` storage, reference CG, and one host
    /// worker. The semantic lowerer independently requires a spatial-vector
    /// displacement, constant coercive Lamé coefficients, a conservative
    /// scalar load potential, and homogeneous trace on all four sides.
    #[must_use]
    pub fn isotropic_elasticity_2d_reference() -> Self {
        let solver = SolverCapabilities::exact([SolverCapability {
            algorithm: LinearSolver::ConjugateGradient,
            operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        }])
        .expect("the elasticity reference solver tuple is exact");
        Self::cartesian_product(
            [DiscretizationMethod::ContinuousGalerkin],
            [(
                MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is non-zero")),
            )],
            [VectorLayoutKind::Replicated],
            solver,
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
        .expect("elasticity reference capability axes are nonempty")
    }

    /// Exact capability of the reference 2D symmetric mixed-simplex path.
    ///
    /// The semantic/numerical handoff remains responsible for proving the
    /// exact Field roles, spaces, constraint, and scaling dimensions. This
    /// execution envelope admits only replicated `f64`, identity-preconditioned
    /// reproducible MINRES for a symmetric-indefinite operator on one host
    /// worker.
    #[must_use]
    pub fn symmetric_mixed_simplicial_2d_reference() -> Self {
        let solver = SolverCapabilities::exact([SolverCapability {
            algorithm: LinearSolver::MinimumResidual,
            operator_properties: LinearOperatorProperties::SymmetricIndefinite,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        }])
        .expect("the mixed reference solver tuple is exact and nonempty");
        Self::cartesian_product(
            [DiscretizationMethod::ContinuousGalerkin],
            [(
                MeshKind::ImportedAffineSimplicial,
                SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is non-zero")),
            )],
            [VectorLayoutKind::Replicated],
            solver,
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
        .expect("mixed reference capability axes are nonempty")
    }

    pub(crate) fn supports(
        &self,
        requirements: RealizationRequirements,
        plan: &RealizationPlan,
    ) -> Result<BTreeSet<LinearOperatorProperties>, Diagnostic> {
        self.operator_property_candidates(
            requirements,
            plan.discretization().method(),
            plan.discretization().mesh().kind(),
            plan.solver(),
            plan.target(),
            plan.schedule(),
        )
    }

    pub(crate) fn supports_fieldwise(
        &self,
        requirements: RealizationRequirements,
        plan: &FieldwiseRealizationPlan,
    ) -> Result<(), Diagnostic> {
        self.supports_components(
            requirements,
            plan.spatial().discretization().method(),
            plan.spatial().discretization().mesh().kind(),
            plan.solver(),
            Some(plan.operator_properties()),
            plan.target(),
            plan.schedule(),
        )
    }

    pub(crate) fn supports_coupled_fieldwise(
        &self,
        requirements: RealizationRequirements,
        plan: &CoupledFieldwiseRealizationPlan,
    ) -> Result<(), Diagnostic> {
        self.supports_components(
            requirements,
            plan.spatial().discretization().method(),
            plan.spatial().discretization().mesh().kind(),
            plan.solver(),
            Some(plan.operator_properties()),
            plan.target(),
            plan.schedule(),
        )
    }

    pub(crate) fn supports_additional_linear_solver(
        &self,
        requirements: RealizationRequirements,
        base: &CoupledFieldwiseRealizationPlan,
        solver: SolverPlan,
        operator_properties: LinearOperatorProperties,
    ) -> Result<(), Diagnostic> {
        self.supports_components(
            requirements,
            base.spatial().discretization().method(),
            base.spatial().discretization().mesh().kind(),
            solver,
            Some(operator_properties),
            base.target(),
            base.schedule(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn supports_components(
        &self,
        requirements: RealizationRequirements,
        method: DiscretizationMethod,
        mesh_kind: MeshKind,
        solver: SolverPlan,
        operator_properties: Option<LinearOperatorProperties>,
        target: Target,
        schedule: ExecutionSchedule,
    ) -> Result<(), Diagnostic> {
        let candidates = self.operator_property_candidates(
            requirements,
            method,
            mesh_kind,
            solver,
            target,
            schedule,
        )?;
        if operator_properties.is_none_or(|properties| candidates.contains(&properties)) {
            return Ok(());
        }
        Err(invalid_realization(format!(
            "complete realization has no exact tuple for method={method:?}, mesh={mesh_kind:?}, dimension={}, scalar={:?}, layout={:?}, solver={:?}/{:?}/{:?}, operator={operator_properties:?}, target={target:?}, schedule={schedule:?}",
            requirements.spatial_dimension,
            requirements.scalar_type,
            requirements.vector_layout,
            solver.algorithm(),
            solver.preconditioner(),
            solver.reduction(),
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn operator_property_candidates(
        &self,
        requirements: RealizationRequirements,
        method: DiscretizationMethod,
        mesh_kind: MeshKind,
        solver: SolverPlan,
        target: Target,
        schedule: ExecutionSchedule,
    ) -> Result<BTreeSet<LinearOperatorProperties>, Diagnostic> {
        if self.combinations.is_empty() {
            return Err(match target {
                Target::HostCpu { .. } => {
                    invalid_realization("backend has no executable host-CPU target")
                }
                Target::CudaGpu { device } => {
                    invalid_realization(format!("backend has no executable CUDA device {device}"))
                }
            });
        }
        let candidates = self
            .combinations
            .iter()
            .filter(|capability| {
                capability.context.spatial.method == method
                    && capability.context.spatial.mesh_kind == mesh_kind
                    && capability
                        .context
                        .spatial
                        .dimensions
                        .supports(requirements.spatial_dimension)
                    && capability.context.vector_layout == requirements.vector_layout
                    && capability.solver.algorithm == solver.algorithm()
                    && capability.solver.preconditioner == solver.preconditioner()
                    && capability.solver.reduction == solver.reduction()
                    && capability.solver.scalar_type == requirements.scalar_type
                    && capability.context.target.supports(target)
                    && capability.context.schedule.supports(schedule)
            })
            .map(|capability| capability.solver.operator_properties)
            .collect::<BTreeSet<_>>();
        if !candidates.is_empty() {
            return Ok(candidates);
        }
        Err(invalid_realization(format!(
            "complete realization has no exact tuple for method={method:?}, mesh={mesh_kind:?}, dimension={}, scalar={:?}, layout={:?}, solver={:?}/{:?}/{:?}, target={target:?}, schedule={schedule:?}",
            requirements.spatial_dimension,
            requirements.scalar_type,
            requirements.vector_layout,
            solver.algorithm(),
            solver.preconditioner(),
            solver.reduction(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solver(algorithm: LinearSolver, properties: LinearOperatorProperties) -> SolverCapability {
        SolverCapability {
            algorithm,
            operator_properties: properties,
            preconditioner: PreconditionerPolicy::Identity,
            reduction: ReductionPolicy::Reproducible,
            scalar_type: ScalarType::F64,
        }
    }

    fn plan(algorithm: LinearSolver) -> SolverPlan {
        SolverPlan::new(algorithm, 1.0e-10, 1.0e-12, NonZeroUsize::MIN)
            .expect("test solver plan is valid")
    }

    #[test]
    fn exact_tuple_rejects_an_invalid_solver_property_pair() {
        let context = RealizationCapabilityContext::new(
            SpatialCapability::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshKind::GeneratedCartesian,
                SpatialDimensionSupport::exact(NonZeroUsize::MIN),
            ),
            VectorLayoutKind::Replicated,
            TargetCapability::HostCpu {
                maximum_threads: NonZeroUsize::MIN,
            },
            ScheduleCapability::Offline,
        );
        let error = RealizationCapability::new(
            context,
            solver(
                LinearSolver::ConjugateGradient,
                LinearOperatorProperties::General,
            ),
        )
        .unwrap_err();
        assert_eq!(
            error.code(),
            eqiora_core::diagnostic::codes::INVALID_REALIZATION
        );
    }

    #[test]
    fn real_time_schedule_admission_is_exact() {
        let admitted = ScheduleCapability::RealTime {
            priority: 3,
            deadline_ns: NonZeroU64::new(10_000).expect("deadline is non-zero"),
        };
        assert!(admitted.supports(ExecutionSchedule::RealTime {
            priority: 3,
            deadline_ns: NonZeroU64::new(10_000).expect("deadline is non-zero"),
        }));
        assert!(!admitted.supports(ExecutionSchedule::RealTime {
            priority: 4,
            deadline_ns: NonZeroU64::new(10_000).expect("deadline is non-zero"),
        }));
        assert!(!admitted.supports(ExecutionSchedule::RealTime {
            priority: 3,
            deadline_ns: NonZeroU64::new(20_000).expect("deadline is non-zero"),
        }));
        assert!(!admitted.supports(ExecutionSchedule::Offline));
    }

    #[test]
    fn exact_admission_tuples_reject_recombined_axes() {
        let dimension = NonZeroUsize::new(2).expect("two is non-zero");
        let host_fem = RealizationCapability::new(
            RealizationCapabilityContext::new(
                SpatialCapability::new(
                    DiscretizationMethod::ContinuousGalerkin,
                    MeshKind::GeneratedCartesian,
                    SpatialDimensionSupport::exact(dimension),
                ),
                VectorLayoutKind::Replicated,
                TargetCapability::HostCpu {
                    maximum_threads: NonZeroUsize::MIN,
                },
                ScheduleCapability::Offline,
            ),
            solver(
                LinearSolver::ConjugateGradient,
                LinearOperatorProperties::SymmetricPositiveDefinite,
            ),
        )
        .expect("host FEM tuple is coherent");
        let cuda_fvm = RealizationCapability::new(
            RealizationCapabilityContext::new(
                SpatialCapability::new(
                    DiscretizationMethod::CellCenteredFiniteVolume,
                    MeshKind::GeneratedCartesian,
                    SpatialDimensionSupport::exact(dimension),
                ),
                VectorLayoutKind::Distributed,
                TargetCapability::CudaGpu { device: 3 },
                ScheduleCapability::Offline,
            ),
            solver(
                LinearSolver::BiConjugateGradientStabilized,
                LinearOperatorProperties::General,
            ),
        )
        .expect("CUDA FVM tuple is coherent");
        let capabilities =
            RealizationCapabilities::exact([host_fem, cuda_fvm]).expect("two exact tuples");

        assert!(
            capabilities
                .supports_components(
                    RealizationRequirements::new(
                        dimension,
                        ScalarType::F64,
                        VectorLayoutKind::Replicated,
                    ),
                    DiscretizationMethod::ContinuousGalerkin,
                    MeshKind::GeneratedCartesian,
                    plan(LinearSolver::ConjugateGradient),
                    Some(LinearOperatorProperties::SymmetricPositiveDefinite),
                    Target::HostCpu {
                        threads: NonZeroUsize::MIN,
                    },
                    ExecutionSchedule::Offline,
                )
                .is_ok()
        );
        assert!(
            capabilities
                .supports_components(
                    RealizationRequirements::new(
                        dimension,
                        ScalarType::F64,
                        VectorLayoutKind::Distributed,
                    ),
                    DiscretizationMethod::CellCenteredFiniteVolume,
                    MeshKind::GeneratedCartesian,
                    plan(LinearSolver::BiConjugateGradientStabilized),
                    Some(LinearOperatorProperties::General),
                    Target::CudaGpu { device: 3 },
                    ExecutionSchedule::Offline,
                )
                .is_ok()
        );

        let recombined = capabilities.supports_components(
            RealizationRequirements::new(dimension, ScalarType::F64, VectorLayoutKind::Replicated),
            DiscretizationMethod::CellCenteredFiniteVolume,
            MeshKind::GeneratedCartesian,
            plan(LinearSolver::BiConjugateGradientStabilized),
            Some(LinearOperatorProperties::General),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
        );
        assert!(recombined.is_err());
    }
}
