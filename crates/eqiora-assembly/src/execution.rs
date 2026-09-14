use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{ExecutionReport, PreparedLinearStructureIdentity};

use crate::sparse::{CsrTopology, CsrValueAssembler};
use crate::{
    AssemblyDelta, AssemblyMap, CooAssembler, LinearSystem, LocalContribution, LocalUnknown,
};

/// Identity of the ordered logical entity set addressed by assembly packets.
///
/// A content-bound identity lets a placement backend prove that packet index
/// `i` names the same entity as its ownership layout. `Unbound` is explicit
/// and remains valid for reference or threaded local assembly, but spatially
/// distributed backends must reject it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AssemblyPacketSetIdentityV1 {
    /// No external content identity is attached to this local operation.
    Unbound,
    /// SHA-256 identity of the authenticated ordered entity set.
    ContentSha256([u8; 32]),
}

impl AssemblyPacketSetIdentityV1 {
    /// Explicitly declare an operation with no externally comparable packet
    /// set identity.
    #[must_use]
    pub const fn unbound() -> Self {
        Self::Unbound
    }

    /// Bind an already authenticated content digest.
    #[must_use]
    pub const fn from_sha256(bytes: [u8; 32]) -> Self {
        Self::ContentSha256(bytes)
    }

    /// Content digest when this packet set is externally bound.
    #[must_use]
    pub const fn sha256(self) -> Option<[u8; 32]> {
        match self {
            Self::Unbound => None,
            Self::ContentSha256(bytes) => Some(bytes),
        }
    }
}

/// Typed ordinal of one output system within an [`AssemblyPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssemblyTargetId(usize);

impl AssemblyTargetId {
    /// Zero-based target ordinal.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// One nonempty square algebraic system produced by assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssemblyTarget {
    size: usize,
}

impl AssemblyTarget {
    /// Construct one square target with `size` equations and unknowns.
    ///
    /// # Errors
    /// Returns `EQ0806` when `size` is zero.
    pub fn new(size: usize) -> Result<Self, Diagnostic> {
        if size == 0 {
            return Err(assembly_failed(
                "an assembly target requires at least one equation",
            ));
        }
        Ok(Self { size })
    }

    /// Equation and unknown count of this square target.
    #[must_use]
    pub const fn size(self) -> usize {
        self.size
    }
}

/// Ordered output shape for one assembly operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssemblyPlan {
    targets: Vec<AssemblyTarget>,
    prepared: Option<Arc<PreparedAssemblyStructure>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedAssemblyStructure {
    packet_encodings: Vec<Arc<[u8]>>,
    topologies: Vec<CsrTopology>,
    identity: PreparedLinearStructureIdentity,
}

#[derive(Debug, Clone, PartialEq)]
struct AssemblyPacketStructure {
    rows: usize,
    columns: usize,
    mappings: Vec<TargetAssemblyMap>,
}

impl AssemblyPacketStructure {
    fn new(mut mappings: Vec<TargetAssemblyMap>) -> Result<Self, Diagnostic> {
        let Some(first) = mappings.first() else {
            return Err(assembly_failed(
                "a prepared assembly packet requires target mappings",
            ));
        };
        let rows = first.map.equations().len();
        let columns = first.map.unknowns().len();
        if rows == 0 {
            return Err(assembly_failed("a prepared assembly packet requires rows"));
        }
        mappings.sort_by_key(|mapping| mapping.target);
        if mappings
            .windows(2)
            .any(|pair| pair[0].target == pair[1].target)
        {
            return Err(assembly_failed(
                "a prepared packet maps one target more than once",
            ));
        }
        Ok(Self {
            rows,
            columns,
            mappings,
        })
    }
}

impl AssemblyPlan {
    /// Construct a nonempty ordered target plan.
    ///
    /// # Errors
    /// Returns `EQ0806` when no output target is declared.
    pub fn new(targets: Vec<AssemblyTarget>) -> Result<Self, Diagnostic> {
        if targets.is_empty() {
            return Err(assembly_failed(
                "an assembly plan requires at least one target",
            ));
        }
        Ok(Self {
            targets,
            prepared: None,
        })
    }

    /// Seal fixed packet maps and canonical CSR topology for repeated value assembly.
    pub fn prepare(mut self, packets: Vec<Vec<TargetAssemblyMap>>) -> Result<Self, Diagnostic> {
        if packets.is_empty() {
            return Err(assembly_failed(
                "prepared assembly requires at least one packet",
            ));
        }
        let packets = packets
            .into_iter()
            .map(AssemblyPacketStructure::new)
            .collect::<Result<Vec<_>, _>>()?;
        if packets
            .iter()
            .flat_map(|packet| &packet.mappings)
            .any(|mapping| mapping.target.index() >= self.targets.len())
        {
            return Err(assembly_failed(
                "prepared packet references a target outside the assembly plan",
            ));
        }
        let packet_encodings = packets
            .iter()
            .map(encode_packet_structure)
            .collect::<Result<Vec<_>, _>>()?;
        let topologies = self
            .targets
            .iter()
            .enumerate()
            .map(|(target, shape)| build_topology(target, shape.size, &packets))
            .collect::<Result<Vec<_>, _>>()?;
        let mut encoding = b"eqiora.assembly.structure/v1\0".to_vec();
        push_usize(&mut encoding, self.targets.len())?;
        for target in &self.targets {
            push_usize(&mut encoding, target.size)?;
        }
        push_usize(&mut encoding, packet_encodings.len())?;
        for packet in &packet_encodings {
            push_usize(&mut encoding, packet.len())?;
            encoding.extend_from_slice(packet);
        }
        for topology in &topologies {
            push_usize(&mut encoding, topology.row_offsets.len())?;
            for &offset in topology.row_offsets.iter() {
                push_usize(&mut encoding, offset)?;
            }
            push_usize(&mut encoding, topology.column_indices.len())?;
            for &column in topology.column_indices.iter() {
                push_usize(&mut encoding, column)?;
            }
        }
        self.prepared = Some(Arc::new(PreparedAssemblyStructure {
            packet_encodings: packet_encodings.into_iter().map(Arc::from).collect(),
            topologies,
            identity: PreparedLinearStructureIdentity::new(encoding)?,
        }));
        Ok(self)
    }

    /// Exact identity of this plan's fixed packet maps and sparse topology.
    #[must_use]
    pub fn structure_identity(&self) -> Option<&PreparedLinearStructureIdentity> {
        self.prepared.as_deref().map(|prepared| &prepared.identity)
    }

    /// Number of output systems.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Obtain the plan-scoped typed ID for one target ordinal.
    #[must_use]
    pub fn target_id(&self, index: usize) -> Option<AssemblyTargetId> {
        (index < self.targets.len()).then_some(AssemblyTargetId(index))
    }

    /// Target shape for a valid plan-scoped ID.
    #[must_use]
    pub fn target(&self, id: AssemblyTargetId) -> Option<AssemblyTarget> {
        self.targets.get(id.0).copied()
    }
}

fn build_topology(
    target: usize,
    size: usize,
    packets: &[AssemblyPacketStructure],
) -> Result<CsrTopology, Diagnostic> {
    let mut rows = vec![BTreeSet::new(); size];
    for packet in packets {
        for mapping in packet
            .mappings
            .iter()
            .filter(|mapping| mapping.target.index() == target)
        {
            if mapping.map.equations().len() != packet.rows
                || mapping.map.unknowns().len() != packet.columns
            {
                return Err(assembly_failed(
                    "prepared packet shape differs from its map",
                ));
            }
            for equation in mapping.map.equations().iter().flatten() {
                if equation.index() >= size {
                    return Err(assembly_failed("prepared equation is outside its target"));
                }
                for unknown in mapping.map.unknowns() {
                    if let LocalUnknown::Free(column) = unknown {
                        if column.index() >= size {
                            return Err(assembly_failed("prepared unknown is outside its target"));
                        }
                        rows[equation.index()].insert(column.index());
                    }
                }
            }
        }
    }
    let mut offsets = Vec::with_capacity(size + 1);
    let mut columns = Vec::new();
    offsets.push(0);
    for (row, entries) in rows.into_iter().enumerate() {
        if entries.is_empty() {
            return Err(assembly_failed(format!(
                "prepared global row {row} has no structural entry"
            )));
        }
        columns.extend(entries);
        offsets.push(columns.len());
    }
    CsrTopology::new(size, offsets, columns)
}

fn encode_packet_structure(packet: &AssemblyPacketStructure) -> Result<Vec<u8>, Diagnostic> {
    let mut out = Vec::new();
    push_usize(&mut out, packet.rows)?;
    push_usize(&mut out, packet.columns)?;
    push_usize(&mut out, packet.mappings.len())?;
    for mapping in &packet.mappings {
        push_usize(&mut out, mapping.target.index())?;
        push_usize(&mut out, mapping.map.equations().len())?;
        for equation in mapping.map.equations() {
            match equation {
                Some(dof) => {
                    out.push(1);
                    push_usize(&mut out, dof.index())?;
                }
                None => out.push(0),
            }
        }
        push_usize(&mut out, mapping.map.unknowns().len())?;
        for unknown in mapping.map.unknowns() {
            match unknown {
                LocalUnknown::Free(dof) => {
                    out.push(0);
                    push_usize(&mut out, dof.index())?;
                }
                LocalUnknown::Fixed(value) => {
                    out.push(1);
                    out.extend_from_slice(&value.to_bits().to_be_bytes());
                }
            }
        }
    }
    Ok(out)
}

fn push_usize(out: &mut Vec<u8>, value: usize) -> Result<(), Diagnostic> {
    let value = u64::try_from(value)
        .map_err(|_| assembly_failed("assembly structure exceeds portable u64"))?;
    out.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

/// One local-to-global map addressed to a specific assembly target.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetAssemblyMap {
    target: AssemblyTargetId,
    map: Arc<AssemblyMap>,
}

impl TargetAssemblyMap {
    /// Bind one map to a target obtained from [`AssemblyPlan::target_id`].
    #[must_use]
    pub fn new(target: AssemblyTargetId, map: impl Into<Arc<AssemblyMap>>) -> Self {
        Self {
            target,
            map: map.into(),
        }
    }

    /// Destination target.
    #[must_use]
    pub const fn target(&self) -> AssemblyTargetId {
        self.target
    }

    /// Local-to-global map for this target.
    #[must_use]
    pub fn map(&self) -> &AssemblyMap {
        &self.map
    }
}

/// One pure local contribution and its one-or-more algebraic projections.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyPacket {
    local: LocalContribution,
    mappings: Vec<TargetAssemblyMap>,
}

/// One plan-validated packet-local delta addressed to its target ordinal.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetAssemblyDelta {
    target: AssemblyTargetId,
    delta: AssemblyDelta,
}

impl TargetAssemblyDelta {
    /// Destination target in the plan used for projection.
    #[must_use]
    pub const fn target(&self) -> AssemblyTargetId {
        self.target
    }

    /// Canonical additive global rows for this target.
    #[must_use]
    pub const fn delta(&self) -> &AssemblyDelta {
        &self.delta
    }
}

impl AssemblyPacket {
    /// Construct a validated packet and canonicalize mappings by target ID.
    ///
    /// Target bounds are checked against the concrete plan during scatter.
    ///
    /// # Errors
    /// Returns `EQ0806` for no mappings, duplicate targets, or local/map shape
    /// mismatch.
    pub fn new(
        local: LocalContribution,
        mut mappings: Vec<TargetAssemblyMap>,
    ) -> Result<Self, Diagnostic> {
        if mappings.is_empty() {
            return Err(assembly_failed(
                "an assembly packet requires at least one target mapping",
            ));
        }
        mappings.sort_by_key(|mapping| mapping.target);
        for pair in mappings.windows(2) {
            if pair[0].target == pair[1].target {
                return Err(assembly_failed(format!(
                    "assembly packet maps target {} more than once",
                    pair[0].target.0
                )));
            }
        }
        for mapping in &mappings {
            if mapping.map.equations().len() != local.rows()
                || mapping.map.unknowns().len() != local.columns()
            {
                return Err(assembly_failed(format!(
                    "target {} map is {}x{} but local contribution is {}x{}",
                    mapping.target.0,
                    mapping.map.equations().len(),
                    mapping.map.unknowns().len(),
                    local.rows(),
                    local.columns()
                )));
            }
        }
        Ok(Self { local, mappings })
    }

    /// Anonymous local matrix and right-hand side.
    #[must_use]
    pub const fn local(&self) -> &LocalContribution {
        &self.local
    }

    /// Canonically target-ordered mappings.
    #[must_use]
    pub fn mappings(&self) -> &[TargetAssemblyMap] {
        &self.mappings
    }

    /// Project every target mapping through one concrete plan.
    ///
    /// All target ordinals, dimensions, global degrees of freedom, fixed
    /// values, and projected arithmetic are checked before any accumulator is
    /// mutated. Returned deltas retain the packet's canonical target order.
    ///
    /// # Errors
    /// Returns `EQ0806` for a target outside the plan or any mapping/projection
    /// failure.
    pub fn project(&self, plan: &AssemblyPlan) -> Result<Vec<TargetAssemblyDelta>, Diagnostic> {
        let mut projected = Vec::with_capacity(self.mappings.len());
        for mapping in &self.mappings {
            let target = plan.target(mapping.target).ok_or_else(|| {
                assembly_failed(format!(
                    "assembly packet references target {} outside plan count {}",
                    mapping.target.0,
                    plan.target_count()
                ))
            })?;
            projected.push(TargetAssemblyDelta {
                target: mapping.target,
                delta: AssemblyDelta::from_local(target.size, &mapping.map, &self.local)?,
            });
        }
        Ok(projected)
    }
}

/// Indexed pure local work evaluated by an assembly backend.
pub trait AssemblyWork: fmt::Debug + Sync {
    /// Identity of the ordered entity set addressed by packet indices.
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1;

    /// Stable logical packet count for this assembly operation.
    fn packet_count(&self) -> usize;

    /// Evaluate one stable logical packet index without global side effects.
    ///
    /// # Errors
    /// Returns a numerical diagnostic from local geometry, coefficients,
    /// quadrature, or packet validation.
    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic>;
}

/// Ergonomic [`AssemblyWork`] backed by one immutable indexed closure.
pub struct IndexedAssemblyWork<F> {
    packet_set: AssemblyPacketSetIdentityV1,
    packet_count: usize,
    evaluate: F,
}

impl<F> IndexedAssemblyWork<F> {
    /// Bind a stable packet count to an indexed evaluator.
    #[must_use]
    pub const fn new(packet_count: usize, evaluate: F) -> Self {
        Self {
            packet_set: AssemblyPacketSetIdentityV1::Unbound,
            packet_count,
            evaluate,
        }
    }

    /// Bind an authenticated ordered packet set to an indexed evaluator.
    #[must_use]
    pub const fn for_packet_set(
        packet_set: AssemblyPacketSetIdentityV1,
        packet_count: usize,
        evaluate: F,
    ) -> Self {
        Self {
            packet_set,
            packet_count,
            evaluate,
        }
    }
}

impl<F> fmt::Debug for IndexedAssemblyWork<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexedAssemblyWork")
            .field("packet_count", &self.packet_count)
            .finish_non_exhaustive()
    }
}

impl<F> AssemblyWork for IndexedAssemblyWork<F>
where
    F: Fn(usize) -> Result<AssemblyPacket, Diagnostic> + Sync,
{
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.packet_set
    }

    fn packet_count(&self) -> usize {
        self.packet_count
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic> {
        if packet_index >= self.packet_count {
            return Err(assembly_failed(format!(
                "assembly packet {packet_index} is outside work count {}",
                self.packet_count
            )));
        }
        (self.evaluate)(packet_index)
    }
}

/// Evidence for one completely accepted assembly operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssemblyReport {
    execution: ExecutionReport,
    packet_count: usize,
    target_count: usize,
    structure_identity: Option<PreparedLinearStructureIdentity>,
}

impl AssemblyReport {
    /// Placement used to evaluate local packets.
    #[must_use]
    pub const fn execution(&self) -> ExecutionReport {
        self.execution
    }

    /// Accepted logical packet count.
    #[must_use]
    pub const fn packet_count(&self) -> usize {
        self.packet_count
    }

    /// Finalized output target count.
    #[must_use]
    pub const fn target_count(&self) -> usize {
        self.target_count
    }

    /// Exact fixed assembly structure used by this operation, when prepared.
    #[must_use]
    pub const fn structure_identity(&self) -> Option<&PreparedLinearStructureIdentity> {
        self.structure_identity.as_ref()
    }
}

/// Finalized target systems and their exact assembly placement evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyResult {
    systems: Vec<LinearSystem>,
    report: AssemblyReport,
}

impl AssemblyResult {
    /// Admit complete target systems produced by an alternate assembly path.
    ///
    /// This is the construction seam for owner-routed or device assembly after
    /// it has independently proved exact packet coverage and reconstructed
    /// complete canonical systems. It validates output shape but does not
    /// itself attest how packets were evaluated or transported.
    ///
    /// # Errors
    /// Returns `EQ0806` for zero accepted packets, a target-count mismatch, or
    /// a system dimension that contradicts its ordered target.
    pub fn from_complete_systems(
        plan: &AssemblyPlan,
        systems: Vec<LinearSystem>,
        packet_count: usize,
        execution: ExecutionReport,
    ) -> Result<Self, Diagnostic> {
        if packet_count == 0 {
            return Err(assembly_failed(
                "an assembly result requires at least one accepted packet",
            ));
        }
        if systems.len() != plan.target_count() {
            return Err(assembly_failed(format!(
                "assembly result has {} systems for {} planned targets",
                systems.len(),
                plan.target_count()
            )));
        }
        for (index, (system, target)) in systems.iter().zip(&plan.targets).enumerate() {
            if system.matrix().rows() != target.size || system.matrix().columns() != target.size {
                return Err(assembly_failed(format!(
                    "assembly result target {index} is {}x{} but plan requires {}x{}",
                    system.matrix().rows(),
                    system.matrix().columns(),
                    target.size,
                    target.size
                )));
            }
        }
        Ok(Self {
            systems,
            report: AssemblyReport {
                execution,
                packet_count,
                target_count: plan.target_count(),
                structure_identity: plan.structure_identity().cloned(),
            },
        })
    }

    /// One system addressed by a target ID from the input plan.
    #[must_use]
    pub fn system(&self, target: AssemblyTargetId) -> Option<&LinearSystem> {
        self.systems.get(target.0)
    }

    /// Ordered finalized systems.
    #[must_use]
    pub fn systems(&self) -> &[LinearSystem] {
        &self.systems
    }

    /// Exact assembly placement and accepted shape.
    #[must_use]
    pub const fn report(&self) -> &AssemblyReport {
        &self.report
    }

    /// Consume the result into ordered systems and its report.
    #[must_use]
    pub fn into_parts(self) -> (Vec<LinearSystem>, AssemblyReport) {
        (self.systems, self.report)
    }
}

/// Backend-neutral indexed assembly execution.
///
/// [`AssemblyWork`] is `Sync` so a backend may evaluate independent packets
/// concurrently. The backend itself is not required to be `Sync`: a physical
/// transport adapter may own one application-serialized collective stream.
/// Concurrent operations use distinct backend instances rather than sharing
/// one mutable transport context implicitly.
pub trait AssemblyBackend: fmt::Debug {
    /// Evaluate, scatter, and finalize one complete assembly operation.
    ///
    /// # Errors
    /// Returns the lowest failing logical packet diagnostic or a structured
    /// plan/scatter/finalization diagnostic. No partial result escapes.
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic>;
}

/// Shared ordered scatter state for assembly backend implementors.
///
/// Backends may evaluate packets under any placement, but must present them
/// here exactly once in increasing logical index order. This type owns the
/// numerical accumulation tree used by both reference and parallel paths.
#[derive(Debug)]
pub struct AssemblyAccumulator {
    plan: AssemblyPlan,
    assemblers: Vec<TargetAccumulator>,
    next_packet: usize,
}

#[derive(Debug)]
enum TargetAccumulator {
    Dynamic(CooAssembler),
    Prepared(CsrValueAssembler),
}

impl AssemblyAccumulator {
    /// Allocate one deterministic accumulator per planned target.
    ///
    /// # Errors
    /// Propagates invalid target shape as `EQ0806`.
    pub fn new(plan: &AssemblyPlan) -> Result<Self, Diagnostic> {
        let assemblers = if let Some(prepared) = &plan.prepared {
            prepared
                .topologies
                .iter()
                .cloned()
                .map(CsrValueAssembler::new)
                .map(TargetAccumulator::Prepared)
                .collect()
        } else {
            plan.targets
                .iter()
                .map(|target| CooAssembler::new(target.size).map(TargetAccumulator::Dynamic))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(Self {
            plan: plan.clone(),
            assemblers,
            next_packet: 0,
        })
    }

    /// Plan used to validate packet-local projections before ordered scatter.
    #[must_use]
    pub const fn plan(&self) -> &AssemblyPlan {
        &self.plan
    }

    /// Scatter the next logical packet through the common ordered path.
    ///
    /// # Errors
    /// Returns `EQ0806` for skipped/repeated indices, a target outside the
    /// plan, invalid global DOFs, or non-finite accumulation.
    pub fn scatter_packet(
        self,
        packet_index: usize,
        packet: &AssemblyPacket,
    ) -> Result<Self, Diagnostic> {
        self.require_packet_index(packet_index)?;
        if let Some(prepared) = &self.plan.prepared {
            let expected = prepared.packet_encodings.get(packet_index).ok_or_else(|| {
                assembly_failed("packet is outside the prepared assembly structure")
            })?;
            let actual = encode_packet_structure(&AssemblyPacketStructure {
                rows: packet.local.rows(),
                columns: packet.local.columns(),
                mappings: packet.mappings.clone(),
            })?;
            if actual.as_slice() != expected.as_ref() {
                return Err(assembly_failed(
                    "packet shape or maps differ from the prepared assembly structure",
                ));
            }
        }
        let projected = packet.project(&self.plan)?;
        self.scatter_projected(packet_index, &projected)
    }

    /// Accumulate one plan-validated projection in logical packet order.
    ///
    /// This is the single stateful scatter path shared by serial packet
    /// assembly and backends that project packets independently.
    ///
    /// The projection must come from [`Self::plan`]. A delta projected against
    /// a different plan is rejected when it names a target outside this plan or
    /// a degree of freedom outside its target, which is what a backend holding
    /// the wrong plan produces; two structurally interchangeable plans are not
    /// distinguished, so a caller obtains the plan from the accumulator rather
    /// than reconstructing one.
    ///
    /// # Errors
    /// Returns `EQ0806` for skipped/repeated indices, empty or foreign-plan
    /// projections, invalid global DOFs, or non-finite accumulation.
    pub fn scatter_projected(
        mut self,
        packet_index: usize,
        projected: &[TargetAssemblyDelta],
    ) -> Result<Self, Diagnostic> {
        self.require_packet_index(packet_index)?;
        if projected.is_empty() {
            return Err(assembly_failed(
                "projected assembly packet requires at least one target delta",
            ));
        }
        let target_count = self.plan.target_count();
        for target_delta in projected {
            let assembler = self
                .assemblers
                .get_mut(target_delta.target.0)
                .ok_or_else(|| {
                    assembly_failed(format!(
                        "projected assembly packet references target {} outside plan count {}",
                        target_delta.target.0, target_count
                    ))
                })?;
            match assembler {
                TargetAccumulator::Dynamic(assembler) => {
                    assembler.scatter_delta(&target_delta.delta)?
                }
                TargetAccumulator::Prepared(assembler) => {
                    assembler.scatter_delta(&target_delta.delta)?
                }
            }
        }
        self.next_packet += 1;
        Ok(self)
    }

    fn require_packet_index(&self, packet_index: usize) -> Result<(), Diagnostic> {
        if packet_index != self.next_packet {
            return Err(assembly_failed(format!(
                "ordered assembly expected packet {}, received {packet_index}",
                self.next_packet
            )));
        }
        Ok(())
    }

    /// Finalize all targets and attach exact execution evidence.
    ///
    /// # Errors
    /// Returns `EQ0806` if any target has an empty structural row.
    pub fn finish(self, execution: ExecutionReport) -> Result<AssemblyResult, Diagnostic> {
        if self
            .plan
            .prepared
            .as_ref()
            .is_some_and(|prepared| prepared.packet_encodings.len() != self.next_packet)
        {
            return Err(assembly_failed(
                "assembly did not cover the complete prepared packet structure",
            ));
        }
        let target_count = self.assemblers.len();
        let systems = self
            .assemblers
            .into_iter()
            .map(|assembler| match assembler {
                TargetAccumulator::Dynamic(assembler) => assembler.finish(),
                TargetAccumulator::Prepared(assembler) => assembler.finish(),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AssemblyResult {
            systems,
            report: AssemblyReport {
                execution,
                packet_count: self.next_packet,
                target_count,
                structure_identity: self.plan.structure_identity().cloned(),
            },
        })
    }
}

/// Direct increasing-index assembly used as the deterministic oracle.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReferenceAssemblyBackend;

/// Shared reference assembly backend.
pub const REFERENCE_ASSEMBLY_BACKEND: ReferenceAssemblyBackend = ReferenceAssemblyBackend;

impl AssemblyBackend for ReferenceAssemblyBackend {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        if work.packet_count() == 0 {
            return Err(assembly_failed(
                "assembly work requires at least one logical packet",
            ));
        }
        let mut accumulator = AssemblyAccumulator::new(plan)?;
        for packet_index in 0..work.packet_count() {
            let packet = work.evaluate(packet_index)?;
            accumulator = accumulator.scatter_packet(packet_index, &packet)?;
        }
        accumulator.finish(ExecutionReport::host_serial())
    }
}

fn assembly_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::ASSEMBLY_FAILED, message)
}

#[cfg(test)]
mod tests;
