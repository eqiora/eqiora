//! Typed in-memory reconstruction over the sole finalized FSI operator.

use eqiora_assembly::AssemblyReport;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, OntologyId};
use eqiora_distributed::DistributedLinearSystem;
use eqiora_execution::{
    AcceptedLinearExecution, AdmittedExecution, DeploymentBinding, ExecutionReceipt,
};
use eqiora_meshing::{CellId, FacetId, VertexId};
use eqiora_realization::{
    CoupledFieldwiseRealizationPlan, MeshArtifactReference, PortableRealizationGraph,
    RealizationRevision, SemanticRevision, SolveRoot, Target, VectorLayoutKind,
};
use eqiora_schema::Model;
use eqiora_solver::{CanonicalCsrSystemView, LinearSolution, LinearSolverBackend, SolverPlan};
use eqiora_spatial_distribution::{
    AssemblyBoundDistributedLinearSystem, DistributedAssemblyEvidence,
    DistributedAssemblyReceiptV1, DistributedAssemblySystemIdentityV1,
};

use crate::discrete_block::DiscreteBlockSystem;
use crate::finalized_spatial::FinalizedLinearCore;
use crate::simplicial_fsi::FixedReferenceFsiAssemblyTargetRoles;
use crate::simplicial_fsi::{
    FinalizedFixedReferenceFsiStep, FixedReferenceFsiPartition, FixedReferenceFsiSolution,
};

/// Exact semantic Field roles retained across numerical execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedReferenceFsiFieldIdentities2d {
    fluid_velocity: Id<kinds::Field>,
    fluid_pressure: Id<kinds::Field>,
    solid_velocity: Id<kinds::Field>,
    solid_displacement: Id<kinds::Field>,
}

impl FixedReferenceFsiFieldIdentities2d {
    pub(super) const fn new(
        fluid_velocity: Id<kinds::Field>,
        fluid_pressure: Id<kinds::Field>,
        solid_velocity: Id<kinds::Field>,
        solid_displacement: Id<kinds::Field>,
    ) -> Self {
        Self {
            fluid_velocity,
            fluid_pressure,
            solid_velocity,
            solid_displacement,
        }
    }

    /// Inertial-fluid velocity Field represented by MINI coefficients.
    #[must_use]
    pub const fn fluid_velocity(self) -> Id<kinds::Field> {
        self.fluid_velocity
    }

    /// Incompressibility-multiplier Field represented by fluid-vertex P1 coefficients.
    #[must_use]
    pub const fn fluid_pressure(self) -> Id<kinds::Field> {
        self.fluid_pressure
    }

    /// Dynamic-solid velocity Field represented by P1 coefficients.
    #[must_use]
    pub const fn solid_velocity(self) -> Id<kinds::Field> {
        self.solid_velocity
    }

    /// Dynamic-solid displacement reconstructed after Backward Euler elimination.
    #[must_use]
    pub const fn solid_displacement(self) -> Id<kinds::Field> {
        self.solid_displacement
    }
}

/// Exact resolved FSI operator with semantic and Realization provenance.
///
/// This wrapper does not copy or replace the pure numerical operator. It owns
/// the sole finalized core object and adds only the typed identities required
/// to reconstruct physical Fields after execution.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedResolvedFixedReferenceFsiStep2d {
    model: OntologyId<Model>,
    semantic_revision: SemanticRevision,
    realization_revision: RealizationRevision,
    mesh_artifact: MeshArtifactReference,
    fields: FixedReferenceFsiFieldIdentities2d,
    partition: FixedReferenceFsiPartition<2>,
    realization_plan: CoupledFieldwiseRealizationPlan,
    realization_graph: PortableRealizationGraph,
    core: FinalizedLinearCore,
    inner: FinalizedFixedReferenceFsiStep<2>,
}

impl FinalizedResolvedFixedReferenceFsiStep2d {
    /// Reduced coordinate of a Field/entity basis component in this owner's scope.
    /// Essential coordinates, absent supports, and eliminated state Fields return `None`.
    #[must_use]
    pub fn free_field_dof(
        &self,
        field: Id<kinds::Field>,
        entity: eqiora_meshing::MeshEntity,
        slot: usize,
        component: usize,
    ) -> Option<eqiora_assembly::DofId> {
        self.inner
            .free_field_dof(crate::region_assembly::mapping::FieldDof {
                field: field.erase(),
                entity,
                slot,
                component,
            })
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        model: OntologyId<Model>,
        semantic_revision: SemanticRevision,
        realization_revision: RealizationRevision,
        mesh_artifact: MeshArtifactReference,
        fields: FixedReferenceFsiFieldIdentities2d,
        partition: FixedReferenceFsiPartition<2>,
        realization_plan: CoupledFieldwiseRealizationPlan,
        realization_graph: PortableRealizationGraph,
        block_system: DiscreteBlockSystem,
        inner: FinalizedFixedReferenceFsiStep<2>,
    ) -> Result<Self, Diagnostic> {
        let SolveRoot::Linear(root) = realization_graph.root() else {
            return Err(eqiora_core::Diagnostic::error(
                eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                "fixed-reference FSI requires a linear portable Realization root",
            ));
        };
        let linear_solve = realization_graph.linear_solve(root).ok_or_else(|| {
            eqiora_core::Diagnostic::error(
                eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                "fixed-reference FSI portable linear root is absent",
            )
        })?;
        let system = realization_graph
            .system(linear_solve.system())
            .ok_or_else(|| {
                eqiora_core::Diagnostic::error(
                    eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                    "fixed-reference FSI portable linear root references an absent system",
                )
            })?;
        let core = FinalizedLinearCore::new(
            linear_solve.plan(),
            system.partition(),
            realization_plan.target(),
            inner.canonical_system_arc(),
        )
        .with_block_system(&block_system, inner.assembly_report())?;
        Ok(Self {
            model,
            semantic_revision,
            realization_revision,
            mesh_artifact,
            fields,
            partition,
            realization_plan,
            realization_graph,
            core,
            inner,
        })
    }

    /// Exact Semantic Model identity admitted by Realization resolution.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Exact admitted Semantic Model revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Exact independent Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.realization_revision
    }

    /// Exact content-addressed mesh selected by Realization resolution.
    #[must_use]
    pub const fn mesh_artifact(&self) -> MeshArtifactReference {
        self.mesh_artifact
    }

    /// Exact semantic roles represented by the finalized coefficients.
    #[must_use]
    pub const fn fields(&self) -> FixedReferenceFsiFieldIdentities2d {
        self.fields
    }

    /// Exact selected host-reproducible or CUDA-fast MINRES plan.
    #[must_use]
    pub const fn solver_plan(&self) -> SolverPlan {
        self.realization_plan.solver()
    }

    /// Replicated or explicitly distributed ownership selected by Realization.
    #[must_use]
    pub const fn vector_layout(&self) -> VectorLayoutKind {
        self.core.vector_layout()
    }

    /// Complete physics-neutral Realization plan used to finalize the operator.
    #[must_use]
    pub const fn realization_plan(&self) -> &CoupledFieldwiseRealizationPlan {
        &self.realization_plan
    }

    /// Common portable DAG that selected the finalized monolithic operator.
    #[must_use]
    pub const fn realization_graph(&self) -> &PortableRealizationGraph {
        &self.realization_graph
    }

    /// Sole captured dimensionless CSR system submitted to execution.
    #[must_use]
    pub fn linear_system(&self) -> &CanonicalCsrSystemView {
        self.core.canonical_csr_system_view()
    }

    /// Complete packet and placement evidence available before execution.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        self.inner.assembly_report()
    }

    /// Admit this finalized operator into its exact generic CUDA deployment.
    ///
    /// The caller supplies only the already validated deployment binding. This
    /// method owns selection of the graph and CSR subject, so a call site
    /// cannot accidentally pair the FSI provenance with a same-shaped foreign
    /// system. Runtime-specific handles remain outside this numerical layer.
    ///
    /// # Errors
    /// Rejects a non-CUDA or distributed FSI plan, device-ordinal drift, or any
    /// contradiction detected by generic graph-bound CUDA admission.
    pub fn admit_cuda(
        &self,
        binding: DeploymentBinding,
    ) -> Result<AdmittedExecution<'_>, Diagnostic> {
        if self.vector_layout() != VectorLayoutKind::Replicated {
            return Err(invalid_cuda_fsi(
                "CUDA FSI admission requires replicated finalized algebra",
            ));
        }
        let Target::CudaGpu { device } = self.realization_plan().target() else {
            return Err(invalid_cuda_fsi(
                "CUDA FSI admission requires an exact CUDA-target Realization",
            ));
        };
        let executor = binding.cuda_executor().ok_or_else(|| {
            invalid_cuda_fsi("CUDA FSI admission requires a CUDA deployment binding")
        })?;
        if executor.device().id().ordinal() != device {
            return Err(invalid_cuda_fsi(
                "CUDA FSI deployment device differs from the exact Realization target",
            ));
        }
        AdmittedExecution::admit_cuda_linear(
            self.realization_graph(),
            self.linear_system(),
            binding,
        )
    }

    /// Bind the exact two-target distributed assembly evidence to this
    /// finalized FSI operator.
    ///
    /// The reduced target becomes the sole distributed solver system. The
    /// retained full target is checked independently for later continuity,
    /// reaction, interface, and energy acceptance. Neither target is exposed
    /// as a public physics IR.
    ///
    /// # Errors
    /// Rejects replicated Realization, absent or exchanged target roles,
    /// assembly receipt shape drift, or any bit-level reduced/full CSR/RHS
    /// disagreement.
    pub fn bind_distributed_assembly(
        self,
        evidence: &DistributedAssemblyEvidence,
    ) -> Result<PreparedDistributedFixedReferenceFsiStep2d, Diagnostic> {
        if self.vector_layout() != VectorLayoutKind::Distributed {
            return Err(invalid_distributed_fsi(
                "distributed FSI assembly binding requires explicitly distributed Realization",
            ));
        }
        if evidence.receipt().mesh_revision().as_bytes() != self.mesh_artifact().sha256() {
            return Err(invalid_distributed_fsi(
                "distributed FSI assembly receipt belongs to another authenticated mesh revision",
            ));
        }
        let roles = self.assembly_target_roles();
        let reduced = evidence.bind_linear_target(roles.reduced(), self.linear_system())?;
        let full_view = self.inner.full_canonical_system_view()?;
        let full_system = evidence.validate_target_system(roles.full(), &full_view)?;
        let assembly_receipt = reduced.assembly_receipt();
        if assembly_receipt.packet_count() != self.assembly_report().packet_count()
            || assembly_receipt.target_count() != self.assembly_report().target_count()
            || assembly_receipt.target_count() != 2
        {
            return Err(invalid_distributed_fsi(
                "distributed FSI assembly receipt differs from the finalized packet/target inventory",
            ));
        }
        Ok(PreparedDistributedFixedReferenceFsiStep2d {
            finalized: self,
            reduced,
            full_system,
        })
    }

    /// Exact reduced-solve and full-reconstruction assembly target roles.
    ///
    /// This stays crate-visible until the distributed evidence bridge can
    /// bind both targets without exposing them as a public physics IR.
    pub(crate) const fn assembly_target_roles(&self) -> FixedReferenceFsiAssemblyTargetRoles {
        self.inner.assembly_target_roles()
    }

    /// Reaccept one solution to this exact system and bind physical arrays to Field IDs.
    ///
    /// # Errors
    /// Preserves every core residual, interface, kinematic, and energy acceptance diagnostic.
    pub fn finish(
        self,
        solution: LinearSolution,
    ) -> Result<ResolvedFixedReferenceFsiSolution2d, Diagnostic> {
        let Self {
            model,
            semantic_revision,
            realization_revision,
            mesh_artifact,
            realization_plan,
            realization_graph,
            core,
            fields,
            partition,
            inner,
            ..
        } = self;
        core.validate_solution(&solution)?;
        Ok(ResolvedFixedReferenceFsiSolution2d {
            model,
            semantic_revision,
            realization_revision,
            mesh_artifact,
            realization_plan,
            realization_graph,
            fields,
            partition,
            inner: inner.finish(solution)?,
        })
    }

    /// Accept one generic CUDA execution and invoke the sole physical finish.
    ///
    /// The returned receipt is the unchanged generic execution receipt; no
    /// FSI-specific execution evidence schema or second physics acceptance
    /// path is introduced.
    ///
    /// # Errors
    /// Rejects graph, operator, solver, device, output-shape, report, or CUDA
    /// trace drift before invoking the existing residual, incompressibility,
    /// kinematic, interface-action, and energy acceptance path.
    pub fn finish_cuda(
        self,
        accepted: AcceptedLinearExecution,
    ) -> Result<(ResolvedFixedReferenceFsiSolution2d, ExecutionReceipt), Diagnostic> {
        let receipt = accepted.receipt();
        let Target::CudaGpu { device } = self.realization_plan().target() else {
            return Err(invalid_cuda_fsi(
                "CUDA FSI finish requires an exact CUDA-target Realization",
            ));
        };
        let executor = receipt
            .binding()
            .cuda_executor()
            .ok_or_else(|| invalid_cuda_fsi("CUDA FSI finish requires a CUDA execution receipt"))?;
        if receipt.binding().realization() != self.realization_graph()
            || receipt.operator() != self.linear_system().agreement_fingerprint()
            || receipt.solver_plan() != self.solver_plan()
            || receipt.dimension() != self.linear_system().columns()
            || receipt.report() != accepted.solution().report()
            || receipt.cuda_trace().is_none()
            || executor.device().id().ordinal() != device
        {
            return Err(invalid_cuda_fsi(
                "CUDA FSI execution receipt differs from the finalized graph, operator, solver, device, output shape, report, or trace",
            ));
        }
        let (linear_solution, receipt) = accepted.into_parts();
        let solution = self.finish(linear_solution)?;
        Ok((solution, receipt))
    }

    /// Execute the exact captured system through a selected backend.
    ///
    /// Numerical policy remains the resolved Realization; only the execution
    /// adapter is supplied here.
    ///
    /// # Errors
    /// Preserves backend and solution-acceptance diagnostics.
    pub fn solve(
        self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<ResolvedFixedReferenceFsiSolution2d, Diagnostic> {
        let SolveRoot::Linear(root) = self.realization_graph.root() else {
            return Err(eqiora_core::Diagnostic::error(
                eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                "fixed-reference FSI portable solve root changed after finalization",
            ));
        };
        let plan = self
            .realization_graph
            .linear_solve(root)
            .ok_or_else(|| {
                eqiora_core::Diagnostic::error(
                    eqiora_core::diagnostic::codes::INVALID_REALIZATION,
                    "fixed-reference FSI portable linear solve changed after finalization",
                )
            })?
            .plan();
        let solved = backend.solve(&self.linear_system().linear_problem()?, plan)?;
        self.finish(solved)
    }
}

/// Distributed algebra admitted from the exact finalized FSI assembly.
///
/// This is an execution handoff, not another model or physical acceptance
/// surface. It owns the original finalizer so exactly one subsequent finish
/// can reconstruct and accept physical Fields.
#[derive(Debug, PartialEq)]
pub struct PreparedDistributedFixedReferenceFsiStep2d {
    finalized: FinalizedResolvedFixedReferenceFsiStep2d,
    reduced: AssemblyBoundDistributedLinearSystem,
    full_system: DistributedAssemblySystemIdentityV1,
}

impl PreparedDistributedFixedReferenceFsiStep2d {
    /// Portable graph required by generic deployment binding and execution admission.
    #[must_use]
    pub const fn realization_graph(&self) -> &PortableRealizationGraph {
        self.finalized.realization_graph()
    }

    /// Sole complete canonical verifier and physical-acceptance operator.
    #[must_use]
    pub fn complete_system(&self) -> &CanonicalCsrSystemView {
        self.finalized.linear_system()
    }

    /// Distributed system derived only from accepted reduced-target ownership and shards.
    #[must_use]
    pub const fn distributed_system(&self) -> &DistributedLinearSystem {
        self.reduced.system()
    }

    /// Exact solver policy selected by the resolved Realization.
    #[must_use]
    pub const fn solver_plan(&self) -> SolverPlan {
        self.finalized.solver_plan()
    }

    /// Accepted two-target distributed assembly receipt.
    #[must_use]
    pub const fn assembly_receipt(&self) -> DistributedAssemblyReceiptV1 {
        self.reduced.assembly_receipt()
    }

    /// Exact property-free identity of the reduced solver target.
    #[must_use]
    pub const fn reduced_system_identity(&self) -> DistributedAssemblySystemIdentityV1 {
        self.reduced.assembly_system_identity()
    }

    /// Exact property-free identity of the retained full acceptance target.
    #[must_use]
    pub const fn full_system_identity(&self) -> DistributedAssemblySystemIdentityV1 {
        self.full_system
    }

    /// Accept generic distributed execution and delegate to the sole physical FSI finish.
    ///
    /// # Errors
    /// Rejects graph, operator, solver, distributed partition/layout/admission,
    /// report, or dimension drift before invoking the existing residual,
    /// incompressibility, kinematic, interface, and energy acceptance path.
    pub fn finish(
        self,
        accepted: AcceptedLinearExecution,
    ) -> Result<AcceptedDistributedFixedReferenceFsiStep2d, Diagnostic> {
        self.validate_execution(&accepted)?;
        let assembly_receipt = self.assembly_receipt();
        let reduced_system = self.reduced_system_identity();
        let Self {
            finalized,
            full_system,
            ..
        } = self;
        let (linear_solution, execution_receipt) = accepted.into_parts();
        let solution = finalized.finish(linear_solution)?;
        Ok(AcceptedDistributedFixedReferenceFsiStep2d {
            solution,
            assembly_receipt,
            reduced_system,
            full_system,
            execution_receipt,
        })
    }

    fn validate_execution(&self, accepted: &AcceptedLinearExecution) -> Result<(), Diagnostic> {
        let receipt = accepted.receipt();
        if receipt.binding().realization() != self.realization_graph()
            || receipt.operator() != self.complete_system().agreement_fingerprint()
            || receipt.solver_plan() != self.solver_plan()
            || receipt.dimension() != self.complete_system().columns()
            || receipt.report() != accepted.solution().report()
        {
            return Err(invalid_distributed_fsi(
                "distributed FSI execution receipt differs from the finalized graph, operator, solver, output shape, or report",
            ));
        }
        let executor = receipt.binding().distributed_executor().ok_or_else(|| {
            invalid_distributed_fsi(
                "distributed FSI finish requires a distributed execution receipt",
            )
        })?;
        let trace = receipt.distributed_trace().ok_or_else(|| {
            invalid_distributed_fsi("distributed FSI execution receipt omits its distributed trace")
        })?;
        let system = self.distributed_system();
        if executor.partitions() != system.partition().count()
            || trace.system() != system.system_identity()
            || trace.partition() != system.partition_identity()
            || trace.layout() != system.layout_identity()
            || trace.admission() != system.admission_fingerprint(self.solver_plan())?
            || trace.process_group() != executor.process_group()
            || trace.partitions() != executor.partitions()
            || trace.workers_per_partition() != executor.workers_per_partition()
            || trace.owner_gather_dimension() != self.complete_system().columns()
        {
            return Err(invalid_distributed_fsi(
                "distributed FSI execution receipt differs from its assembly-derived partition, layout, admission, or process group",
            ));
        }
        Ok(())
    }
}

/// Accepted distributed execution paired with the existing physical FSI result.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedDistributedFixedReferenceFsiStep2d {
    solution: ResolvedFixedReferenceFsiSolution2d,
    assembly_receipt: DistributedAssemblyReceiptV1,
    reduced_system: DistributedAssemblySystemIdentityV1,
    full_system: DistributedAssemblySystemIdentityV1,
    execution_receipt: ExecutionReceipt,
}

impl AcceptedDistributedFixedReferenceFsiStep2d {
    /// Sole physical FSI solution accepted by the existing finish path.
    #[must_use]
    pub const fn solution(&self) -> &ResolvedFixedReferenceFsiSolution2d {
        &self.solution
    }

    /// Accepted distributed assembly receipt.
    #[must_use]
    pub const fn assembly_receipt(&self) -> DistributedAssemblyReceiptV1 {
        self.assembly_receipt
    }

    /// Exact property-free identity of the reduced solver target.
    #[must_use]
    pub const fn reduced_system_identity(&self) -> DistributedAssemblySystemIdentityV1 {
        self.reduced_system
    }

    /// Exact property-free identity of the retained full acceptance target.
    #[must_use]
    pub const fn full_system_identity(&self) -> DistributedAssemblySystemIdentityV1 {
        self.full_system
    }

    /// Immutable generic execution receipt retained without translation.
    #[must_use]
    pub const fn execution_receipt(&self) -> &ExecutionReceipt {
        &self.execution_receipt
    }
}

fn invalid_distributed_fsi(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}

fn invalid_cuda_fsi(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}

/// Accepted physical FSI Fields bound to exact in-memory semantic identities.
///
/// This is deliberately not a durable result schema; RFC 0051 owns persisted
/// scientific state and trajectory. The wrapper prevents clients from
/// inventing an untyped side table between exact Fields and numerical arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFixedReferenceFsiSolution2d {
    model: OntologyId<Model>,
    semantic_revision: SemanticRevision,
    realization_revision: RealizationRevision,
    mesh_artifact: MeshArtifactReference,
    realization_plan: CoupledFieldwiseRealizationPlan,
    realization_graph: PortableRealizationGraph,
    fields: FixedReferenceFsiFieldIdentities2d,
    partition: FixedReferenceFsiPartition<2>,
    inner: FixedReferenceFsiSolution<2>,
}

impl ResolvedFixedReferenceFsiSolution2d {
    /// Exact Semantic Model identity supplying the equations.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Exact accepted Semantic revision.
    #[must_use]
    pub const fn semantic_revision(&self) -> SemanticRevision {
        self.semantic_revision
    }

    /// Exact accepted Realization revision.
    #[must_use]
    pub const fn realization_revision(&self) -> RealizationRevision {
        self.realization_revision
    }

    /// Exact content-addressed mesh used by the finalized operator.
    #[must_use]
    pub const fn mesh_artifact(&self) -> MeshArtifactReference {
        self.mesh_artifact
    }

    /// Complete physics-neutral Realization plan used by the accepted solve.
    #[must_use]
    pub const fn realization_plan(&self) -> &CoupledFieldwiseRealizationPlan {
        &self.realization_plan
    }

    /// Common portable DAG that selected this accepted monolithic solve.
    #[must_use]
    pub const fn realization_graph(&self) -> &PortableRealizationGraph {
        &self.realization_graph
    }

    /// Exact semantic roles represented by this result.
    #[must_use]
    pub const fn fields(&self) -> FixedReferenceFsiFieldIdentities2d {
        self.fields
    }

    /// Fluid-closure vertices supporting P1 coefficients of `fluid_velocity`.
    #[must_use]
    pub fn fluid_velocity_vertices(&self) -> &[VertexId] {
        self.partition.fluid_vertices()
    }

    /// Fluid cells supporting MINI bubble coefficients, in coefficient order.
    #[must_use]
    pub fn fluid_velocity_cells(&self) -> &[CellId] {
        self.partition.fluid_cells()
    }

    /// Solid-closure vertices supporting P1 coefficients of `solid_velocity`.
    #[must_use]
    pub fn solid_velocity_vertices(&self) -> &[VertexId] {
        self.partition.solid_vertices()
    }

    /// Solid cells supporting the P1 velocity and displacement Fields.
    #[must_use]
    pub fn solid_cells(&self) -> &[CellId] {
        self.partition.solid_cells()
    }

    /// Exact interface facets whose vertex coefficients are shared by both velocities.
    #[must_use]
    pub fn interface_facets(&self) -> &[FacetId] {
        self.partition.interface_facets()
    }

    /// Shared mesh-vertex physical velocity coefficients.
    ///
    /// Project with [`Self::fluid_velocity_vertices`] or
    /// [`Self::solid_velocity_vertices`]. Both projections address the same
    /// coefficients on the interface, which is the exact trace quotient.
    #[must_use]
    pub fn vertex_velocity_coefficients(&self) -> &[[f64; 2]] {
        self.inner.vertex_velocity()
    }

    /// Fluid-velocity P1 coefficient at one vertex in the fluid closure.
    #[must_use]
    pub fn fluid_velocity_coefficient(&self, vertex: VertexId) -> Option<[f64; 2]> {
        self.partition
            .fluid_vertices()
            .binary_search(&vertex)
            .ok()
            .map(|_| self.inner.vertex_velocity()[vertex.index()])
    }

    /// Solid-velocity P1 coefficient at one vertex in the solid closure.
    #[must_use]
    pub fn solid_velocity_coefficient(&self, vertex: VertexId) -> Option<[f64; 2]> {
        self.partition
            .solid_vertices()
            .binary_search(&vertex)
            .ok()
            .map(|_| self.inner.vertex_velocity()[vertex.index()])
    }

    /// Fluid-cell MINI bubble coefficients keyed by exact [`Self::fluid_velocity_cells`] identity.
    #[must_use]
    pub fn fluid_velocity_bubble_coefficients(
        &self,
    ) -> &std::collections::BTreeMap<CellId, [f64; 2]> {
        self.inner.fluid_cell_bubble_velocity()
    }

    /// Fluid pressure P1 support vertices in coefficient order.
    #[must_use]
    pub fn fluid_pressure_vertices(&self) -> &[VertexId] {
        self.inner.fluid_pressure_vertices()
    }

    /// Physical pressure coefficients bound to the exact fluid pressure Field.
    #[must_use]
    pub fn fluid_pressure_coefficients(&self) -> &[f64] {
        self.inner.fluid_pressure()
    }

    /// Fluid-pressure P1 coefficient at one supported vertex.
    #[must_use]
    pub fn fluid_pressure_coefficient(&self, vertex: VertexId) -> Option<f64> {
        self.inner
            .fluid_pressure_vertices()
            .binary_search(&vertex)
            .ok()
            .map(|position| self.inner.fluid_pressure()[position])
    }

    /// Solid-closure vertices supporting reconstructed displacement coefficients.
    #[must_use]
    pub fn solid_displacement_vertices(&self) -> &[VertexId] {
        self.partition.solid_vertices()
    }

    /// Physical displacement coefficients in mesh order, zero outside the solid.
    #[must_use]
    pub fn solid_displacement_coefficients(&self) -> &[[f64; 2]] {
        self.inner.solid_displacement()
    }

    /// Reconstructed solid-displacement P1 coefficient at one supported vertex.
    #[must_use]
    pub fn solid_displacement_coefficient(&self, vertex: VertexId) -> Option<[f64; 2]> {
        self.partition
            .solid_vertices()
            .binary_search(&vertex)
            .ok()
            .map(|_| self.inner.solid_displacement()[vertex.index()])
    }

    /// Pure numerical Fields and all falsifying balance evidence.
    #[must_use]
    pub const fn numerical_evidence(&self) -> &FixedReferenceFsiSolution<2> {
        &self.inner
    }

    /// Consume the identity wrapper after evidence has been recorded.
    #[must_use]
    pub fn into_numerical_evidence(self) -> FixedReferenceFsiSolution<2> {
        self.inner
    }
}
