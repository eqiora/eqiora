use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora_assembly::{AssemblyPlan, AssemblyTargetId, CsrMatrix, DofId, LinearSystem};
use eqiora_core::{Diagnostic, ScalarType};
use eqiora_distributed::{
    DistributedLinearSystem, GlobalVectorSpace, OwnedLinearSystemShard, Partition, PartitionId,
};
use eqiora_solver::{CanonicalCsrSystemView, ExecutionReport, ExecutionTopology};
use sha2::{Digest, Sha256};

use crate::{DistributedMeshLayout, DistributedMeshLayoutIdentityV1, MeshRevisionIdentityV1};

use super::codec::{
    WireReader, hash_f64s, hash_usize, hash_usizes, invalid, push_usize, reserve, target_sizes,
};
use super::ownership::{AdmittedRowOwnership, AssemblyRowOwnership, validate_ownership};
use super::route::{
    AcceptedAssemblyInbox, AssemblyRowRouteV1, DistributedAssemblyPlanIdentityV1,
    DistributedAssemblyRoutePlanV1, LocalRouteAdmissionV1, validate_local_route_admissions,
};

const SYSTEM_IDENTITY_DOMAIN_V1: &[u8] = b"eqiora.distributed-assembly-system/v1\0";
const RECEIPT_IDENTITY_DOMAIN_V1: &[u8] = b"eqiora.distributed-assembly-receipt/v1\0";
const SHARD_WIRE_MAGIC_V1: &[u8; 8] = b"EQASHD01";

/// Property-free identity of one exact assembled sparse system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DistributedAssemblySystemIdentityV1([u8; 32]);

impl DistributedAssemblySystemIdentityV1 {
    /// Fixed-size storage identity bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Identity of one fully accepted distributed assembly receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DistributedAssemblyReceiptIdentityV1([u8; 32]);

impl DistributedAssemblyReceiptIdentityV1 {
    /// Fixed-size receipt identity bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// One target's compact canonical rows owned by one partition.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedRowAssemblyResult {
    pub(super) plan: DistributedAssemblyPlanIdentityV1,
    pub(super) target: AssemblyTargetId,
    pub(super) partition: PartitionId,
    pub(super) global_size: usize,
    pub(super) rows: Vec<DofId>,
    pub(super) row_offsets: Vec<usize>,
    pub(super) column_indices: Vec<DofId>,
    pub(super) values: Vec<f64>,
    pub(super) right_hand_side: Vec<f64>,
}

impl OwnedRowAssemblyResult {
    /// Sealed route plan from which this shard was canonically folded.
    #[must_use]
    pub const fn plan(&self) -> DistributedAssemblyPlanIdentityV1 {
        self.plan
    }

    /// Plan-local target ordinal.
    #[must_use]
    pub const fn target(&self) -> AssemblyTargetId {
        self.target
    }

    /// Unique owner of every row in this shard.
    #[must_use]
    pub const fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Complete target dimension used by global columns.
    #[must_use]
    pub const fn global_size(&self) -> usize {
        self.global_size
    }

    /// Strictly increasing owned global rows.
    #[must_use]
    pub fn rows(&self) -> &[DofId] {
        &self.rows
    }

    /// CSR offsets into this shard's compact row storage.
    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// Strictly increasing global columns inside every compact row.
    #[must_use]
    pub fn column_indices(&self) -> &[DofId] {
        &self.column_indices
    }

    /// Exact accumulated values in compact CSR order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Right-hand-side values in the same order as [`Self::rows`].
    #[must_use]
    pub fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }

    /// Encode a bounded, endian-stable owner shard for byte collectives.
    ///
    /// # Errors
    /// Returns `EQ0806` for malformed internal storage, an unrepresentable
    /// index, arithmetic overflow, or allocation failure.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        validate_shard_storage(self)?;
        let words = self
            .rows
            .len()
            .checked_add(self.row_offsets.len())
            .and_then(|value| value.checked_add(self.column_indices.len()))
            .and_then(|value| value.checked_add(self.values.len()))
            .and_then(|value| value.checked_add(self.right_hand_side.len()))
            .and_then(|value| value.checked_add(5))
            .ok_or_else(|| invalid("assembly shard wire length overflows usize"))?;
        let capacity = words
            .checked_mul(8)
            .and_then(|value| value.checked_add(SHARD_WIRE_MAGIC_V1.len() + 32))
            .ok_or_else(|| invalid("assembly shard wire length overflows usize"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| invalid("could not reserve assembly shard wire payload"))?;
        bytes.extend_from_slice(SHARD_WIRE_MAGIC_V1);
        bytes.extend_from_slice(&self.plan.0);
        push_usize(&mut bytes, self.target.index())?;
        push_usize(&mut bytes, self.partition.index())?;
        push_usize(&mut bytes, self.global_size)?;
        push_usize(&mut bytes, self.rows.len())?;
        push_usize(&mut bytes, self.values.len())?;
        for row in &self.rows {
            push_usize(&mut bytes, row.index())?;
        }
        for offset in &self.row_offsets {
            push_usize(&mut bytes, *offset)?;
        }
        for column in &self.column_indices {
            push_usize(&mut bytes, column.index())?;
        }
        for value in &self.values {
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        for value in &self.right_hand_side {
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        debug_assert_eq!(bytes.len(), capacity);
        Ok(bytes)
    }

    /// Decode and validate one owner shard against a concrete assembly plan.
    ///
    /// Exact row ownership and shard inventory are subsequently checked by
    /// [`reconstruct_distributed_assembly`].
    ///
    /// # Errors
    /// Returns `EQ0806` for malformed, truncated, trailing, noncanonical,
    /// non-finite, or oversized storage.
    pub fn from_bytes(
        assembly_plan: &AssemblyPlan,
        route_plan: DistributedAssemblyPlanIdentityV1,
        bytes: &[u8],
    ) -> Result<Self, Diagnostic> {
        let mut reader = WireReader::new(bytes, SHARD_WIRE_MAGIC_V1)?;
        let encoded_plan = DistributedAssemblyPlanIdentityV1(reader.array_32()?);
        if encoded_plan != route_plan {
            return Err(invalid(
                "assembly shard belongs to another sealed route plan",
            ));
        }
        let target_index = reader.usize()?;
        let target = assembly_plan
            .target_id(target_index)
            .ok_or_else(|| invalid("assembly shard names a target outside the plan"))?;
        let partition = PartitionId::new(reader.usize()?);
        let global_size = reader.usize()?;
        let row_count = reader.usize()?;
        let value_count = reader.usize()?;
        let words = row_count
            .checked_add(
                row_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("assembly shard row-offset count overflows usize"))?,
            )
            .and_then(|value| value.checked_add(value_count))
            .and_then(|value| value.checked_add(value_count))
            .and_then(|value| value.checked_add(row_count))
            .ok_or_else(|| invalid("assembly shard payload count overflows usize"))?;
        let required = words
            .checked_mul(8)
            .ok_or_else(|| invalid("assembly shard payload length overflows usize"))?;
        if reader.remaining() != required {
            return Err(invalid(
                "assembly shard wire length does not match its counts",
            ));
        }
        let mut rows = reserve(row_count, "decoded assembly shard rows")?;
        for _ in 0..row_count {
            rows.push(DofId::new(reader.usize()?));
        }
        let mut row_offsets = reserve(row_count + 1, "decoded assembly shard row offsets")?;
        for _ in 0..=row_count {
            row_offsets.push(reader.usize()?);
        }
        let mut column_indices = reserve(value_count, "decoded assembly shard columns")?;
        for _ in 0..value_count {
            column_indices.push(DofId::new(reader.usize()?));
        }
        let mut values = reserve(value_count, "decoded assembly shard values")?;
        for _ in 0..value_count {
            values.push(reader.f64()?);
        }
        let mut right_hand_side = reserve(row_count, "decoded assembly shard right-hand side")?;
        for _ in 0..row_count {
            right_hand_side.push(reader.f64()?);
        }
        reader.finish()?;
        let shard = Self {
            plan: route_plan,
            target,
            partition,
            global_size,
            rows,
            row_offsets,
            column_indices,
            values,
            right_hand_side,
        };
        validate_shard_storage(&shard)?;
        Ok(shard)
    }
}

impl DistributedAssemblyRoutePlanV1 {
    /// Admit one destination's complete unordered route inbox and fold it by
    /// target, row, then global packet index.
    ///
    /// Admission and folding are one operation so no adapter can retain or
    /// reuse an unchecked or partially accepted inbox.
    ///
    /// # Errors
    /// Returns `EQ0806` for a missing, duplicate, foreign, wrong-destination,
    /// or payload-mismatched route, an ownership disagreement, or a non-finite
    /// or structurally empty accumulated row.
    pub fn fold_inbox(
        &self,
        ownership: &AdmittedRowOwnership,
        destination: PartitionId,
        routes: Vec<AssemblyRowRouteV1>,
    ) -> Result<Vec<OwnedRowAssemblyResult>, Diagnostic> {
        let inbox = self.accept_inbox(destination, routes)?;
        fold_accepted_inbox(self, ownership, inbox)
    }
}

fn fold_accepted_inbox(
    route_plan: &DistributedAssemblyRoutePlanV1,
    ownership: &AdmittedRowOwnership,
    inbox: AcceptedAssemblyInbox,
) -> Result<Vec<OwnedRowAssemblyResult>, Diagnostic> {
    if inbox.plan != route_plan.identity || ownership.layout != route_plan.layout {
        return Err(invalid(
            "assembly inbox, route plan, and row ownership disagree",
        ));
    }
    let mut accumulators = (0..route_plan.target_sizes.len())
        .map(|_| RowAccumulator::default())
        .collect::<Vec<_>>();
    // `accept_inbox` canonicalizes by (target,row,packet), so every scalar
    // accumulation retains increasing global packet order.
    for route in inbox.routes {
        let accumulator = &mut accumulators[route.descriptor.target.index()];
        accumulate_route(accumulator, &route)?;
    }
    accumulators
        .into_iter()
        .enumerate()
        .map(|(target_index, accumulator)| {
            let target = route_plan
                .target_ids
                .get(target_index)
                .copied()
                .ok_or_else(|| invalid("assembly target disappeared from its route plan"))?;
            finish_shard(
                route_plan.identity,
                target,
                inbox.destination,
                route_plan.target_sizes[target_index],
                &route_plan.row_owners[target_index],
                accumulator,
            )
        })
        .collect()
}

/// Fixed-size acceptance record for one complete owner-routed assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedAssemblyReceiptV1 {
    pub(super) identity: DistributedAssemblyReceiptIdentityV1,
    pub(super) mesh: MeshRevisionIdentityV1,
    pub(super) layout: DistributedMeshLayoutIdentityV1,
    pub(super) plan: DistributedAssemblyPlanIdentityV1,
    pub(super) packet_count: usize,
    pub(super) target_count: usize,
    pub(super) partition_count: NonZeroUsize,
}

impl DistributedAssemblyReceiptV1 {
    /// Domain-separated receipt summary.
    #[must_use]
    pub const fn identity(self) -> DistributedAssemblyReceiptIdentityV1 {
        self.identity
    }

    /// Authenticated mesh revision from which cell ownership was derived.
    #[must_use]
    pub const fn mesh_revision(self) -> MeshRevisionIdentityV1 {
        self.mesh
    }

    /// Exact mesh-layout agreement identity.
    #[must_use]
    pub const fn layout(self) -> DistributedMeshLayoutIdentityV1 {
        self.layout
    }

    /// Payload-bound route-plan identity.
    #[must_use]
    pub const fn plan(self) -> DistributedAssemblyPlanIdentityV1 {
        self.plan
    }

    /// Exactly-once packet inventory.
    #[must_use]
    pub const fn packet_count(self) -> usize {
        self.packet_count
    }

    /// Ordered assembly target count.
    #[must_use]
    pub const fn target_count(self) -> usize {
        self.target_count
    }

    /// Participating logical partitions.
    #[must_use]
    pub const fn partition_count(self) -> NonZeroUsize {
        self.partition_count
    }
}

/// Inspectable shards, algebraic owner maps, and receipt from an accepted run.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributedAssemblyEvidence {
    pub(super) partitions: Vec<AssemblyRowOwnership>,
    pub(super) shards: Vec<Vec<OwnedRowAssemblyResult>>,
    pub(super) system_identities: Vec<DistributedAssemblySystemIdentityV1>,
    receipt: DistributedAssemblyReceiptV1,
}

/// One accepted assembly target bound to its sole distributed solver system.
///
/// This token preserves assembly lineage while exposing only the distributed
/// algebra derived from the accepted owner map and owner-row shards. It cannot
/// be constructed from an independently selected solver partition.
#[derive(Debug, PartialEq)]
pub struct AssemblyBoundDistributedLinearSystem {
    target: AssemblyTargetId,
    assembly_receipt: DistributedAssemblyReceiptV1,
    assembly_system: DistributedAssemblySystemIdentityV1,
    system: DistributedLinearSystem,
}

impl AssemblyBoundDistributedLinearSystem {
    /// Plan-local assembly target promoted to distributed execution.
    #[must_use]
    pub const fn target(&self) -> AssemblyTargetId {
        self.target
    }

    /// Exact accepted owner-routed assembly receipt.
    #[must_use]
    pub const fn assembly_receipt(&self) -> DistributedAssemblyReceiptV1 {
        self.assembly_receipt
    }

    /// Property-free identity of the exact assembled CSR and RHS.
    #[must_use]
    pub const fn assembly_system_identity(&self) -> DistributedAssemblySystemIdentityV1 {
        self.assembly_system
    }

    /// Transport-neutral distributed system derived from the accepted rows.
    #[must_use]
    pub const fn system(&self) -> &DistributedLinearSystem {
        &self.system
    }
}

impl DistributedAssemblyEvidence {
    /// Ordered target row partitions.
    #[must_use]
    pub fn target_partitions(&self) -> &[AssemblyRowOwnership] {
        &self.partitions
    }

    /// Ordered target shards; each inner slice is in partition order.
    #[must_use]
    pub fn shards(&self) -> &[Vec<OwnedRowAssemblyResult>] {
        &self.shards
    }

    /// Exact property-free system identities in target order.
    #[must_use]
    pub fn system_identities(&self) -> &[DistributedAssemblySystemIdentityV1] {
        &self.system_identities
    }

    /// Derived exact row ownership for one valid plan target.
    #[must_use]
    pub fn target_partition(&self, target: AssemblyTargetId) -> Option<&AssemblyRowOwnership> {
        self.partitions.get(target.index())
    }

    /// Owner-row shards for one valid plan target, ordered by partition.
    #[must_use]
    pub fn target_shards(&self, target: AssemblyTargetId) -> Option<&[OwnedRowAssemblyResult]> {
        self.shards.get(target.index()).map(Vec::as_slice)
    }

    /// Exact property-free identity of one reconstructed target system.
    #[must_use]
    pub fn target_system_identity(
        &self,
        target: AssemblyTargetId,
    ) -> Option<DistributedAssemblySystemIdentityV1> {
        self.system_identities.get(target.index()).copied()
    }

    /// Accepted transport-neutral receipt.
    #[must_use]
    pub const fn receipt(&self) -> DistributedAssemblyReceiptV1 {
        self.receipt
    }

    /// Verify one complete canonical CSR/RHS against an accepted assembly
    /// target without assigning solver meaning to that target.
    ///
    /// This is used for retained full-system or reaction targets that belong
    /// to physical acceptance but are not themselves submitted to a solver.
    ///
    /// # Errors
    /// Returns `EQ0806` for an absent target or any bit-level CSR/RHS drift.
    pub fn validate_target_system(
        &self,
        target: AssemblyTargetId,
        complete: &CanonicalCsrSystemView,
    ) -> Result<DistributedAssemblySystemIdentityV1, Diagnostic> {
        let assembly_system = self
            .target_system_identity(target)
            .ok_or_else(|| invalid("distributed assembly evidence omits target system identity"))?;
        if canonical_system_identity(complete)? != assembly_system {
            return Err(invalid(
                "complete canonical CSR/RHS differs from the accepted assembly target",
            ));
        }
        Ok(assembly_system)
    }

    /// Bind one exact accepted target to its complete canonical verifier and
    /// promote the accepted owner rows to distributed algebra.
    ///
    /// No owner map is accepted from the caller. The solver partition and
    /// halo are derived only from this evidence's accepted row ownership and
    /// owner-row CSR payloads. The complete view is used as an independent
    /// bit-exact verifier, not as a source for repartitioning.
    ///
    /// # Errors
    /// Returns `EQ0806` or `EQ0807` for an absent target, CSR/RHS identity
    /// drift, an empty solver partition, or any contradiction between the
    /// accepted ownership, shards, and complete verifier.
    pub fn bind_linear_target(
        &self,
        target: AssemblyTargetId,
        complete: &CanonicalCsrSystemView,
    ) -> Result<AssemblyBoundDistributedLinearSystem, Diagnostic> {
        let ownership = self
            .target_partition(target)
            .ok_or_else(|| invalid("distributed assembly evidence omits the requested target"))?;
        let shards = self
            .target_shards(target)
            .ok_or_else(|| invalid("distributed assembly evidence omits target owner shards"))?;
        let assembly_system = self.validate_target_system(target, complete)?;
        let mut owners = reserve(ownership.owners().len(), "solver row owners")?;
        owners.extend_from_slice(ownership.owners());
        let partition = Partition::new(
            GlobalVectorSpace::new(ownership.global_size(), ScalarType::F64),
            ownership.partition_count(),
            owners,
        )?;
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(shards.len())
            .map_err(|_| invalid("could not reserve distributed solver owner shards"))?;
        for shard in shards {
            let mut rows = reserve(shard.rows.len(), "solver shard rows")?;
            rows.extend(shard.rows.iter().map(|row| row.index()));
            let mut columns = reserve(shard.column_indices.len(), "solver shard columns")?;
            columns.extend(shard.column_indices.iter().map(|column| column.index()));
            owned.push(OwnedLinearSystemShard::new(
                shard.partition,
                ownership.global_size(),
                rows,
                copied(&shard.row_offsets, "solver shard row offsets")?,
                columns,
                copied(&shard.values, "solver shard values")?,
                copied(&shard.right_hand_side, "solver shard right-hand side")?,
            )?);
        }
        let system = DistributedLinearSystem::from_owned_shards(complete, partition, owned)?;
        Ok(AssemblyBoundDistributedLinearSystem {
            target,
            assembly_receipt: self.receipt,
            assembly_system,
            system,
        })
    }
}

/// Validate all owner shards and reconstruct complete canonical systems.
///
/// Shards may arrive in any order. Exactly one shard per target/partition is
/// required, and every shard must contain exactly the rows admitted to that
/// owner before its values can enter a complete system. Exactly one opaque
/// local-route admission per producer is also required.
///
/// # Errors
/// Returns `EQ0806` for malformed execution topology, incomplete producer
/// admission, missing/duplicate/wrong shards, row-owner disagreement, or
/// invalid sparse storage.
pub fn reconstruct_distributed_assembly(
    layout: &DistributedMeshLayout,
    plan: &AssemblyPlan,
    ownership: &AdmittedRowOwnership,
    route_plan: &DistributedAssemblyRoutePlanV1,
    admissions: Vec<LocalRouteAdmissionV1>,
    mut shards: Vec<OwnedRowAssemblyResult>,
    execution: ExecutionReport,
) -> Result<(Vec<LinearSystem>, DistributedAssemblyEvidence), Diagnostic> {
    validate_ownership(layout, plan, ownership)?;
    if route_plan.layout != layout.identity()
        || route_plan.row_ownership != ownership.identity
        || route_plan.packet_count != layout.cell_count()
        || route_plan.partition_count != layout.partition_count()
        || route_plan.target_sizes != target_sizes(plan)?
        || route_plan.row_owners
            != ownership
                .targets
                .iter()
                .map(|target| target.owners().to_vec())
                .collect::<Vec<_>>()
    {
        return Err(invalid(
            "distributed reconstruction inputs disagree with the sealed route plan",
        ));
    }
    let admissions = validate_local_route_admissions(route_plan, ownership, admissions)?;
    validate_execution(layout.partition_count(), execution)?;
    shards.sort_by_key(|shard| (shard.target.index(), shard.partition.index()));
    let expected_count = plan
        .target_count()
        .checked_mul(layout.partition_count().get())
        .ok_or_else(|| invalid("assembly shard inventory count overflows usize"))?;
    if shards.len() != expected_count {
        return Err(invalid(format!(
            "distributed assembly has {} shards, expected {expected_count}",
            shards.len()
        )));
    }
    let mut grouped = Vec::new();
    grouped
        .try_reserve_exact(plan.target_count())
        .map_err(|_| invalid("could not reserve distributed shard groups"))?;
    for target_index in 0..plan.target_count() {
        let target = plan
            .target_id(target_index)
            .ok_or_else(|| invalid("assembly target disappeared during reconstruction"))?;
        let start = target_index * layout.partition_count().get();
        let end = start + layout.partition_count().get();
        let group = &shards[start..end];
        for (partition, shard) in group.iter().enumerate() {
            if shard.target != target || shard.partition != PartitionId::new(partition) {
                return Err(invalid(
                    "distributed shard inventory is missing, duplicate, or misaddressed",
                ));
            }
            validate_shard_against_ownership(
                shard,
                route_plan.identity,
                &ownership.targets[target_index],
            )?;
        }
        grouped.push(group.to_vec());
    }
    let systems = grouped
        .iter()
        .zip(&ownership.targets)
        .map(|(target_shards, target_ownership)| {
            reconstruct_system(target_ownership, target_shards)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let system_identities = systems
        .iter()
        .map(system_identity)
        .collect::<Result<Vec<_>, _>>()?;
    let receipt = receipt(
        layout,
        route_plan,
        &admissions,
        &grouped,
        &system_identities,
        execution,
    )?;
    Ok((
        systems,
        DistributedAssemblyEvidence {
            partitions: ownership.targets.clone(),
            shards: grouped,
            system_identities,
            receipt,
        },
    ))
}

#[derive(Debug, Default)]
struct RowAccumulator {
    pub(super) entries: BTreeMap<(usize, usize), f64>,
    pub(super) right_hand_side: BTreeMap<usize, f64>,
}

fn accumulate_route(
    accumulator: &mut RowAccumulator,
    route: &AssemblyRowRouteV1,
) -> Result<(), Diagnostic> {
    let row = route.descriptor.row.index();
    for &(column, value) in &route.entries {
        let accumulated = accumulator
            .entries
            .get(&(row, column.index()))
            .copied()
            .unwrap_or(0.0)
            + value;
        if !accumulated.is_finite() {
            return Err(invalid(
                "owner-row assembly produced a non-finite accumulated value",
            ));
        }
        accumulator
            .entries
            .insert((row, column.index()), accumulated);
    }
    let rhs = accumulator
        .right_hand_side
        .get(&row)
        .copied()
        .unwrap_or(0.0)
        + route.rhs;
    if !rhs.is_finite() {
        return Err(invalid(
            "owner-row assembly produced a non-finite accumulated right-hand side",
        ));
    }
    accumulator.right_hand_side.insert(row, rhs);
    Ok(())
}

fn finish_shard(
    plan: DistributedAssemblyPlanIdentityV1,
    target: AssemblyTargetId,
    partition: PartitionId,
    global_size: usize,
    row_owners: &[PartitionId],
    accumulator: RowAccumulator,
) -> Result<OwnedRowAssemblyResult, Diagnostic> {
    let rows = row_owners
        .iter()
        .enumerate()
        .filter_map(|(row, owner)| (*owner == partition).then_some(DofId::new(row)))
        .collect::<Vec<_>>();
    let mut row_offsets = Vec::with_capacity(rows.len() + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    let mut right_hand_side = Vec::with_capacity(rows.len());
    row_offsets.push(0);
    for row in &rows {
        for (&(entry_row, column), &value) in accumulator
            .entries
            .range((row.index(), 0)..=(row.index(), usize::MAX))
        {
            debug_assert_eq!(entry_row, row.index());
            if value != 0.0 {
                column_indices.push(DofId::new(column));
                values.push(value);
            }
        }
        if row_offsets.last() == Some(&column_indices.len()) {
            return Err(invalid(format!(
                "target {} owned global row {} has no nonzero entry",
                target.index(),
                row.index()
            )));
        }
        row_offsets.push(column_indices.len());
        right_hand_side.push(
            accumulator
                .right_hand_side
                .get(&row.index())
                .copied()
                .unwrap_or(0.0),
        );
    }
    if accumulator
        .entries
        .keys()
        .any(|(row, _)| row_owners.get(*row) != Some(&partition))
        || accumulator
            .right_hand_side
            .keys()
            .any(|row| row_owners.get(*row) != Some(&partition))
    {
        return Err(invalid("owner accumulator contains a row owned elsewhere"));
    }
    let shard = OwnedRowAssemblyResult {
        plan,
        target,
        partition,
        global_size,
        rows,
        row_offsets,
        column_indices,
        values,
        right_hand_side,
    };
    validate_shard_storage(&shard)?;
    Ok(shard)
}

fn validate_shard_storage(shard: &OwnedRowAssemblyResult) -> Result<(), Diagnostic> {
    if shard.global_size == 0
        || shard.row_offsets.len() != shard.rows.len() + 1
        || shard.row_offsets.first() != Some(&0)
        || shard.row_offsets.last() != Some(&shard.values.len())
        || shard.column_indices.len() != shard.values.len()
        || shard.right_hand_side.len() != shard.rows.len()
    {
        return Err(invalid("owner shard has inconsistent sparse storage shape"));
    }
    if shard
        .rows
        .iter()
        .any(|row| row.index() >= shard.global_size)
        || shard.rows.windows(2).any(|pair| pair[0] >= pair[1])
        || shard.row_offsets.windows(2).any(|pair| pair[0] >= pair[1])
        || shard
            .column_indices
            .iter()
            .any(|column| column.index() >= shard.global_size)
        || shard
            .values
            .iter()
            .any(|value| !value.is_finite() || *value == 0.0)
        || shard.right_hand_side.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("owner shard contains noncanonical rows or values"));
    }
    for range in shard.row_offsets.windows(2) {
        if shard.column_indices[range[0]..range[1]]
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "owner shard columns must increase strictly within each row",
            ));
        }
    }
    Ok(())
}

fn validate_shard_against_ownership(
    shard: &OwnedRowAssemblyResult,
    route_plan: DistributedAssemblyPlanIdentityV1,
    ownership: &AssemblyRowOwnership,
) -> Result<(), Diagnostic> {
    validate_shard_storage(shard)?;
    if shard.plan != route_plan
        || shard.global_size != ownership.global_size().get()
        || shard.partition.index() >= ownership.partition_count().get()
    {
        return Err(invalid("owner shard shape or partition is invalid"));
    }
    let expected = ownership
        .owned_rows(shard.partition)
        .map(DofId::new)
        .collect::<Vec<_>>();
    if shard.rows != expected {
        return Err(invalid(
            "owner shard rows differ from collectively admitted ownership",
        ));
    }
    Ok(())
}

fn reconstruct_system(
    ownership: &AssemblyRowOwnership,
    shards: &[OwnedRowAssemblyResult],
) -> Result<LinearSystem, Diagnostic> {
    let size = ownership.global_size().get();
    let mut row_offsets = Vec::with_capacity(size + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    let mut right_hand_side = Vec::with_capacity(size);
    row_offsets.push(0);
    for (row, owner) in ownership.owners().iter().copied().enumerate() {
        let shard = &shards[owner.index()];
        let local = shard
            .rows
            .binary_search(&DofId::new(row))
            .map_err(|_| invalid("row-owner shard is missing a global row"))?;
        let range = shard.row_offsets[local]..shard.row_offsets[local + 1];
        column_indices.extend(
            shard.column_indices[range.clone()]
                .iter()
                .map(|column| column.index()),
        );
        values.extend_from_slice(&shard.values[range]);
        right_hand_side.push(shard.right_hand_side[local]);
        row_offsets.push(column_indices.len());
    }
    LinearSystem::new(
        CsrMatrix::from_sorted_csr(size, size, row_offsets, column_indices, values)?,
        right_hand_side,
    )
}

fn system_identity(
    system: &LinearSystem,
) -> Result<DistributedAssemblySystemIdentityV1, Diagnostic> {
    let mut hash = Sha256::new();
    hash.update(SYSTEM_IDENTITY_DOMAIN_V1);
    hash_usize(&mut hash, system.matrix().rows())?;
    hash_usize(&mut hash, system.matrix().columns())?;
    hash_usizes(&mut hash, system.matrix().row_offsets())?;
    hash_usizes(&mut hash, system.matrix().column_indices())?;
    hash_f64s(&mut hash, system.matrix().values())?;
    hash_f64s(&mut hash, system.rhs())?;
    Ok(DistributedAssemblySystemIdentityV1(hash.finalize().into()))
}

fn canonical_system_identity(
    system: &CanonicalCsrSystemView,
) -> Result<DistributedAssemblySystemIdentityV1, Diagnostic> {
    let mut hash = Sha256::new();
    hash.update(SYSTEM_IDENTITY_DOMAIN_V1);
    hash_usize(&mut hash, system.rows())?;
    hash_usize(&mut hash, system.columns())?;
    hash_usizes(&mut hash, system.row_offsets())?;
    hash_usizes(&mut hash, system.column_indices())?;
    hash_f64s(&mut hash, system.values())?;
    hash_f64s(&mut hash, system.right_hand_side())?;
    Ok(DistributedAssemblySystemIdentityV1(hash.finalize().into()))
}

fn copied<T: Copy>(values: &[T], name: &'static str) -> Result<Vec<T>, Diagnostic> {
    let mut copy = reserve(values.len(), name)?;
    copy.extend_from_slice(values);
    Ok(copy)
}

fn receipt(
    layout: &DistributedMeshLayout,
    route_plan: &DistributedAssemblyRoutePlanV1,
    admissions: &[LocalRouteAdmissionV1],
    shards: &[Vec<OwnedRowAssemblyResult>],
    system_identities: &[DistributedAssemblySystemIdentityV1],
    execution: ExecutionReport,
) -> Result<DistributedAssemblyReceiptV1, Diagnostic> {
    validate_execution(layout.partition_count(), execution)?;
    let mut hash = Sha256::new();
    hash.update(RECEIPT_IDENTITY_DOMAIN_V1);
    hash.update(layout.identity().as_bytes());
    hash.update(route_plan.identity.as_bytes());
    hash_usize(&mut hash, route_plan.packet_count)?;
    hash_usize(&mut hash, admissions.len())?;
    for admission in admissions {
        hash.update(admission.identity.as_bytes());
    }
    hash_usize(&mut hash, shards.len())?;
    hash_execution(&mut hash, execution)?;
    for target_shards in shards {
        hash_usize(&mut hash, target_shards.len())?;
        for shard in target_shards {
            hash.update(Sha256::digest(shard.to_bytes()?));
        }
    }
    hash_usize(&mut hash, system_identities.len())?;
    for identity in system_identities {
        hash.update(identity.as_bytes());
    }
    Ok(DistributedAssemblyReceiptV1 {
        identity: DistributedAssemblyReceiptIdentityV1(hash.finalize().into()),
        mesh: layout.mesh(),
        layout: layout.identity(),
        plan: route_plan.identity,
        packet_count: route_plan.packet_count,
        target_count: shards.len(),
        partition_count: layout.partition_count(),
    })
}

fn validate_execution(
    partition_count: NonZeroUsize,
    execution: ExecutionReport,
) -> Result<(), Diagnostic> {
    match execution.topology() {
        ExecutionTopology::Distributed { ranks, .. } if ranks == partition_count => Ok(()),
        ExecutionTopology::Distributed { ranks, .. } => Err(invalid(format!(
            "distributed assembly execution has {ranks} ranks for {partition_count} partitions"
        ))),
        _ => Err(invalid(
            "distributed assembly requires a distributed execution topology",
        )),
    }
}

fn hash_execution(hash: &mut Sha256, execution: ExecutionReport) -> Result<(), Diagnostic> {
    let adapter = execution.adapter().as_str().as_bytes();
    hash_usize(hash, adapter.len())?;
    hash.update(adapter);
    match execution.topology() {
        ExecutionTopology::Host { workers } => {
            hash.update([0]);
            hash_usize(hash, workers.get())?;
        }
        ExecutionTopology::Distributed {
            ranks,
            workers_per_partition,
        } => {
            hash.update([1]);
            hash_usize(hash, ranks.get())?;
            hash_usize(hash, workers_per_partition.get())?;
        }
        ExecutionTopology::Cuda { device } => {
            hash.update([2]);
            hash.update(device.to_le_bytes());
        }
    }
    Ok(())
}
