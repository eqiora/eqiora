use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use eqiora_assembly::{
    AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTargetId, AssemblyWork, DofId,
    TargetAssemblyDelta,
};
use eqiora_core::Diagnostic;
use eqiora_distributed::PartitionId;
use sha2::{Digest, Sha256};

use crate::{DistributedMeshLayout, DistributedMeshLayoutIdentityV1};

use super::codec::{hash_usize, invalid, nonzero, reserve, target_sizes};
use super::route::AssemblyRowRouteV1;

const ROW_OWNERSHIP_IDENTITY_DOMAIN_V1: &[u8] = b"eqiora.distributed-assembly-row-ownership/v1\0";

/// Fixed-size identity of collectively admitted target-row ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdmittedRowOwnershipIdentityV1(pub(super) [u8; 32]);

impl AdmittedRowOwnershipIdentityV1 {
    /// Fixed-size bytes for execution-group agreement before route exchange.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// One owner-only packet projection before any transport decision.
#[derive(Debug, Clone)]
pub struct LocalAssemblyProjection {
    pub(super) layout: DistributedMeshLayoutIdentityV1,
    pub(super) packet_set: AssemblyPacketSetIdentityV1,
    pub(super) partition_count: NonZeroUsize,
    pub(super) producer: PartitionId,
    pub(super) packet_count: usize,
    pub(super) target_sizes: Vec<usize>,
    packets: Vec<ProjectedPacket>,
    pub(super) candidates: Vec<Vec<Option<PartitionId>>>,
}

impl LocalAssemblyProjection {
    /// Evaluate exactly the cell packets owned by `producer` and project them
    /// through the common assembly mapping contract.
    ///
    /// # Errors
    /// Returns `EQ0806` for a producer outside the layout, a packet/mesh count
    /// mismatch, a projection failure, or an unrepresentable allocation.
    pub fn evaluate_owned(
        layout: &DistributedMeshLayout,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork<f64>,
        producer: PartitionId,
    ) -> Result<Self, Diagnostic> {
        if producer.index() >= layout.partition_count().get() {
            return Err(invalid(format!(
                "spatial assembly producer {} is outside 0..{}",
                producer.index(),
                layout.partition_count()
            )));
        }
        if work.packet_count() == 0 || work.packet_count() != layout.cell_count() {
            return Err(invalid(format!(
                "spatial assembly has {} packets for {} mesh cells",
                work.packet_count(),
                layout.cell_count()
            )));
        }
        let packet_set = work.packet_set_identity();
        if packet_set.sha256() != Some(layout.mesh().as_bytes()) {
            return Err(invalid(
                "spatial assembly packet set is unbound or differs from the exact mesh revision",
            ));
        }
        let target_sizes = target_sizes(plan)?;
        let mut packets = Vec::new();
        packets
            .try_reserve_exact(
                (0..layout.cell_count())
                    .filter(|index| layout.cell_owner(*index) == Some(producer))
                    .count(),
            )
            .map_err(|_| invalid("could not reserve owner-local assembly packets"))?;
        for packet_index in 0..layout.cell_count() {
            if layout.cell_owner(packet_index) == Some(producer) {
                packets.push(ProjectedPacket {
                    index: packet_index,
                    producer,
                    targets: work.evaluate(packet_index)?.project(plan)?,
                });
            }
        }
        if packets.is_empty() {
            return Err(invalid(format!(
                "spatial assembly producer {} owns no packet",
                producer.index()
            )));
        }
        let candidates = local_candidates(&target_sizes, &packets)?;
        Ok(Self {
            layout: layout.identity(),
            packet_set,
            partition_count: layout.partition_count(),
            producer,
            packet_count: work.packet_count(),
            target_sizes,
            packets,
            candidates,
        })
    }

    /// Logical execution partition that evaluated these packets.
    #[must_use]
    pub const fn producer(&self) -> PartitionId {
        self.producer
    }

    /// Complete logical packet count shared by the execution group.
    #[must_use]
    pub const fn packet_count(&self) -> usize {
        self.packet_count
    }

    /// Ordered assembly target dimensions.
    #[must_use]
    pub fn target_sizes(&self) -> &[usize] {
        &self.target_sizes
    }

    /// Flatten local row-owner candidates for an elementwise unsigned MIN.
    /// The partition count is the sole sentinel for an unsupported local row.
    ///
    /// # Errors
    /// Returns `EQ0806` if a portable collective value cannot be represented.
    pub fn collective_candidates(&self) -> Result<CollectiveRowOwnerCandidatesV1, Diagnostic> {
        let sentinel = u64::try_from(self.partition_count.get())
            .map_err(|_| invalid("partition count exceeds portable u64"))?;
        let value_count = self
            .target_sizes
            .iter()
            .try_fold(0_usize, |total, size| total.checked_add(*size))
            .ok_or_else(|| invalid("collective row-candidate count overflows usize"))?;
        let mut values = reserve(value_count, "collective row-owner candidates")?;
        for target in &self.candidates {
            for candidate in target {
                values.push(match candidate {
                    Some(candidate) => u64::try_from(candidate.index())
                        .map_err(|_| invalid("row-owner candidate exceeds portable u64"))?,
                    None => sentinel,
                });
            }
        }
        Ok(CollectiveRowOwnerCandidatesV1 {
            layout: self.layout,
            packet_set: self.packet_set,
            producer: self.producer,
            target_sizes: self.target_sizes.clone(),
            sentinel,
            values,
        })
    }

    /// Materialize the exact payload routes implied by admitted row ownership.
    ///
    /// # Errors
    /// Returns `EQ0806` if the ownership belongs to another layout/plan or a
    /// projected row cannot be routed to its admitted owner.
    pub fn routes(
        &self,
        ownership: &AdmittedRowOwnership,
    ) -> Result<Vec<AssemblyRowRouteV1>, Diagnostic> {
        ownership.validate_projection(self)?;
        let mut routes = Vec::new();
        let capacity = self
            .packets
            .iter()
            .flat_map(|packet| &packet.targets)
            .map(|target| target.delta().rows().len())
            .try_fold(0_usize, usize::checked_add)
            .ok_or_else(|| invalid("spatial assembly route count overflows usize"))?;
        routes
            .try_reserve_exact(capacity)
            .map_err(|_| invalid("could not reserve owner-local assembly routes"))?;
        for packet in &self.packets {
            for target in &packet.targets {
                let target_index = target.target().index();
                let target_size = self.target_sizes[target_index];
                for row in target.delta().rows() {
                    let destination =
                        ownership.owner(target.target(), row.row()).ok_or_else(|| {
                            invalid("projected assembly row has no admitted destination")
                        })?;
                    routes.push(AssemblyRowRouteV1::new(
                        packet.index,
                        packet.producer,
                        target.target(),
                        target_size,
                        row.row(),
                        destination,
                        row.entries().to_vec(),
                        row.rhs(),
                    )?);
                }
            }
        }
        routes.sort_by_key(AssemblyRowRouteV1::canonical_key);
        Ok(routes)
    }
}

/// Portable target-major candidates for one elementwise collective MIN.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveRowOwnerCandidatesV1 {
    pub(super) layout: DistributedMeshLayoutIdentityV1,
    pub(super) packet_set: AssemblyPacketSetIdentityV1,
    pub(super) producer: PartitionId,
    pub(super) target_sizes: Vec<usize>,
    sentinel: u64,
    values: Vec<u64>,
}

impl CollectiveRowOwnerCandidatesV1 {
    /// Producing execution partition.
    #[must_use]
    pub const fn producer(&self) -> PartitionId {
        self.producer
    }

    /// Exact unsupported-row sentinel, equal to the partition count.
    #[must_use]
    pub const fn sentinel(&self) -> u64 {
        self.sentinel
    }

    /// Target-major values supplied to the collective MIN.
    #[must_use]
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}

#[derive(Debug, Clone)]
struct ProjectedPacket {
    index: usize,
    pub(super) producer: PartitionId,
    pub(super) targets: Vec<TargetAssemblyDelta<f64>>,
}

/// Exact collective row ownership admitted before route exchange.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedRowOwnership {
    pub(super) layout: DistributedMeshLayoutIdentityV1,
    pub(super) packet_set: AssemblyPacketSetIdentityV1,
    pub(super) packet_count: usize,
    pub(super) targets: Vec<AssemblyRowOwnership>,
    pub(super) identity: AdmittedRowOwnershipIdentityV1,
}

/// Unique assembly-row owners without requiring every cell partition to own
/// a row in every target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssemblyRowOwnership {
    pub(super) global_size: NonZeroUsize,
    pub(super) partition_count: NonZeroUsize,
    pub(super) owners: Vec<PartitionId>,
}

impl AssemblyRowOwnership {
    fn new(
        global_size: NonZeroUsize,
        partition_count: NonZeroUsize,
        owners: Vec<PartitionId>,
    ) -> Result<Self, Diagnostic> {
        if owners.len() != global_size.get() {
            return Err(invalid("assembly row-owner count differs from target size"));
        }
        if let Some(owner) = owners
            .iter()
            .find(|owner| owner.index() >= partition_count.get())
        {
            return Err(invalid(format!(
                "assembly row owner {} is outside 0..{}",
                owner.index(),
                partition_count
            )));
        }
        Ok(Self {
            global_size,
            partition_count,
            owners,
        })
    }

    /// Complete target dimension.
    #[must_use]
    pub const fn global_size(&self) -> NonZeroUsize {
        self.global_size
    }

    /// Number of cell partitions participating in assembly.
    #[must_use]
    pub const fn partition_count(&self) -> NonZeroUsize {
        self.partition_count
    }

    /// Unique owner of each global row.
    #[must_use]
    pub fn owners(&self) -> &[PartitionId] {
        &self.owners
    }

    /// Owner of one valid global row.
    #[must_use]
    pub fn owner(&self, row: usize) -> Option<PartitionId> {
        self.owners.get(row).copied()
    }

    pub(super) fn owned_rows(&self, partition: PartitionId) -> impl Iterator<Item = usize> + '_ {
        self.owners
            .iter()
            .enumerate()
            .filter_map(move |(row, owner)| (*owner == partition).then_some(row))
    }
}

impl AdmittedRowOwnership {
    /// Admit one exact owner-local projection from every layout partition and
    /// derive each row owner as the minimum producer supporting that equation.
    ///
    /// # Errors
    /// Returns `EQ0806` for missing/duplicate producers, incomplete packet
    /// coverage, incompatible plans, or an unsupported row.
    pub fn admit(
        layout: &DistributedMeshLayout,
        plan: &AssemblyPlan,
        projections: &[LocalAssemblyProjection],
    ) -> Result<Self, Diagnostic> {
        validate_projection_inventory(layout, plan, projections)?;
        let target_sizes = target_sizes(plan)?;
        let sentinel = layout.partition_count().get();
        let mut owners = target_sizes
            .iter()
            .map(|size| vec![sentinel; *size])
            .collect::<Vec<_>>();
        for projection in projections {
            for (target, candidates) in projection.candidates.iter().enumerate() {
                for (row, candidate) in candidates.iter().enumerate() {
                    if let Some(candidate) = candidate {
                        owners[target][row] = owners[target][row].min(candidate.index());
                    }
                }
            }
        }
        let owners = owners
            .into_iter()
            .enumerate()
            .map(|(target, rows)| {
                rows.into_iter()
                    .enumerate()
                    .map(|(row, owner)| {
                        (owner != sentinel)
                            .then(|| PartitionId::new(owner))
                            .ok_or_else(|| {
                                invalid(format!(
                                    "spatial assembly target {target} row {row} has no supporting packet"
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_owners(layout, plan, projections[0].packet_set, owners)
    }

    /// Validate target-major row owners already produced by an elementwise
    /// collective MIN for this exact local projection.
    ///
    /// The returned ownership is not sufficient by itself to issue a receipt:
    /// every producer must later create a
    /// [`LocalRouteAdmissionV1`](crate::LocalRouteAdmissionV1), which proves
    /// that the collective result is the exact supported minimum.
    ///
    /// # Errors
    /// Returns `EQ0806` for a foreign local candidate set, target/row shape
    /// mismatch, unsupported row sentinel, or out-of-range owner.
    pub fn from_collective_min(
        layout: &DistributedMeshLayout,
        plan: &AssemblyPlan,
        local: &CollectiveRowOwnerCandidatesV1,
        owners: &[u64],
    ) -> Result<Self, Diagnostic> {
        let target_sizes = target_sizes(plan)?;
        let expected_count = target_sizes
            .iter()
            .try_fold(0_usize, |total, size| total.checked_add(*size))
            .ok_or_else(|| invalid("collective row-owner count overflows usize"))?;
        if local.layout != layout.identity()
            || local.packet_set.sha256() != Some(layout.mesh().as_bytes())
            || local.target_sizes != target_sizes
            || local.sentinel
                != u64::try_from(layout.partition_count().get())
                    .map_err(|_| invalid("partition count exceeds portable u64"))?
            || local.values.len() != expected_count
            || owners.len() != expected_count
        {
            return Err(invalid(
                "collective row ownership belongs to another layout, packet set, or plan",
            ));
        }
        let mut cursor = 0;
        let mut targets = Vec::new();
        targets
            .try_reserve_exact(target_sizes.len())
            .map_err(|_| invalid("could not reserve collective row-owner targets"))?;
        for (target, size) in target_sizes.iter().copied().enumerate() {
            let mut target_owners = reserve(size, "collective target row owners")?;
            for (row, owner) in owners[cursor..cursor + size].iter().copied().enumerate() {
                if owner >= local.sentinel {
                    return Err(invalid(format!(
                        "spatial assembly target {target} row {row} has no valid collective owner"
                    )));
                }
                let owner = usize::try_from(owner)
                    .map_err(|_| invalid("collective row owner exceeds usize"))?;
                target_owners.push(PartitionId::new(owner));
            }
            targets.push(target_owners);
            cursor += size;
        }
        Self::from_owners(layout, plan, local.packet_set, targets)
    }

    fn from_owners(
        layout: &DistributedMeshLayout,
        plan: &AssemblyPlan,
        packet_set: AssemblyPacketSetIdentityV1,
        owners: Vec<Vec<PartitionId>>,
    ) -> Result<Self, Diagnostic> {
        let target_sizes = target_sizes(plan)?;
        if packet_set.sha256() != Some(layout.mesh().as_bytes())
            || owners.len() != target_sizes.len()
        {
            return Err(invalid(format!(
                "collective row ownership has {} targets for {} planned targets",
                owners.len(),
                target_sizes.len()
            )));
        }
        let targets = owners
            .into_iter()
            .zip(target_sizes)
            .map(|(owners, size)| {
                AssemblyRowOwnership::new(
                    nonzero(size, "assembly target")?,
                    layout.partition_count(),
                    owners,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let identity = row_ownership_identity(layout, packet_set, &targets)?;
        Ok(Self {
            layout: layout.identity(),
            packet_set,
            packet_count: layout.cell_count(),
            targets,
            identity,
        })
    }

    /// Identity for collective agreement before any route inventory or
    /// variable-size payload exchange.
    #[must_use]
    pub const fn identity(&self) -> AdmittedRowOwnershipIdentityV1 {
        self.identity
    }

    /// Ordered target row partitions.
    #[must_use]
    pub fn target_partitions(&self) -> &[AssemblyRowOwnership] {
        &self.targets
    }

    /// One target's exact row partition.
    #[must_use]
    pub fn target_partition(&self, target: AssemblyTargetId) -> Option<&AssemblyRowOwnership> {
        self.targets.get(target.index())
    }

    /// Unique destination for a valid target row.
    #[must_use]
    pub fn owner(&self, target: AssemblyTargetId, row: DofId) -> Option<PartitionId> {
        self.targets.get(target.index())?.owner(row.index())
    }

    pub(super) fn validate_projection(
        &self,
        projection: &LocalAssemblyProjection,
    ) -> Result<(), Diagnostic> {
        if projection.layout != self.layout
            || projection.packet_set != self.packet_set
            || projection.packet_count != self.packet_count
        {
            return Err(invalid(
                "owner-local projection belongs to another distributed assembly",
            ));
        }
        if projection.target_sizes.len() != self.targets.len()
            || projection
                .target_sizes
                .iter()
                .zip(&self.targets)
                .any(|(size, target)| *size != target.global_size().get())
        {
            return Err(invalid(
                "owner-local projection target shape differs from row ownership",
            ));
        }
        Ok(())
    }
}

fn local_candidates(
    target_sizes: &[usize],
    packets: &[ProjectedPacket],
) -> Result<Vec<Vec<Option<PartitionId>>>, Diagnostic> {
    let mut candidates = target_sizes
        .iter()
        .map(|size| vec![None; *size])
        .collect::<Vec<_>>();
    for packet in packets {
        for target in &packet.targets {
            let rows = candidates.get_mut(target.target().index()).ok_or_else(|| {
                invalid("local assembly projection names a target outside its plan")
            })?;
            for row in target.delta().rows() {
                let candidate = rows.get_mut(row.row().index()).ok_or_else(|| {
                    invalid("local assembly projection names a row outside its target")
                })?;
                *candidate = Some(packet.producer);
            }
        }
    }
    Ok(candidates)
}

fn validate_projection_inventory(
    layout: &DistributedMeshLayout,
    plan: &AssemblyPlan,
    projections: &[LocalAssemblyProjection],
) -> Result<(), Diagnostic> {
    if projections.len() != layout.partition_count().get() {
        return Err(invalid(format!(
            "spatial assembly has {} local projections for {} partitions",
            projections.len(),
            layout.partition_count()
        )));
    }
    let target_sizes = target_sizes(plan)?;
    let mut packets = Vec::new();
    packets
        .try_reserve_exact(layout.cell_count())
        .map_err(|_| invalid("could not reserve collective packet inventory"))?;
    let mut seen_producers = BTreeSet::new();
    for projection in projections {
        if projection.layout != layout.identity()
            || projection.packet_set.sha256() != Some(layout.mesh().as_bytes())
            || projection.partition_count != layout.partition_count()
            || projection.packet_count != layout.cell_count()
            || projection.target_sizes != target_sizes
        {
            return Err(invalid(
                "owner-local projection belongs to another layout or assembly plan",
            ));
        }
        if !seen_producers.insert(projection.producer) {
            return Err(invalid(format!(
                "assembly producer {} occurs more than once",
                projection.producer.index()
            )));
        }
        for packet in &projection.packets {
            if packet.producer != projection.producer {
                return Err(invalid("owner-local projection contains a foreign packet"));
            }
            packets.push((packet.index, packet.producer));
        }
    }
    packets.sort_unstable();
    if packets.len() != layout.cell_count() {
        return Err(invalid(format!(
            "spatial assembly admits {} packets for {} cells",
            packets.len(),
            layout.cell_count()
        )));
    }
    for (expected, (packet, producer)) in packets.into_iter().enumerate() {
        let owner = layout
            .cell_owner(expected)
            .ok_or_else(|| invalid("collective packet inventory names an invalid mesh cell"))?;
        if packet != expected || producer != owner {
            return Err(invalid(format!(
                "assembly packet {expected} is missing, duplicate, or assigned to the wrong producer"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_ownership(
    layout: &DistributedMeshLayout,
    plan: &AssemblyPlan,
    ownership: &AdmittedRowOwnership,
) -> Result<(), Diagnostic> {
    if ownership.layout != layout.identity()
        || ownership.packet_set.sha256() != Some(layout.mesh().as_bytes())
        || ownership.packet_count != layout.cell_count()
        || ownership.targets.len() != plan.target_count()
    {
        return Err(invalid(
            "collective row ownership belongs to another layout or assembly plan",
        ));
    }
    for (target_index, target) in ownership.targets.iter().enumerate() {
        let expected = plan
            .target_id(target_index)
            .and_then(|target_id| plan.target(target_id))
            .ok_or_else(|| invalid("assembly target is unavailable"))?;
        if target.global_size().get() != expected.size()
            || target.partition_count() != layout.partition_count()
        {
            return Err(invalid(
                "collective row ownership target shape differs from the plan",
            ));
        }
    }
    Ok(())
}

fn row_ownership_identity(
    layout: &DistributedMeshLayout,
    packet_set: AssemblyPacketSetIdentityV1,
    targets: &[AssemblyRowOwnership],
) -> Result<AdmittedRowOwnershipIdentityV1, Diagnostic> {
    let mut hash = Sha256::new();
    hash.update(ROW_OWNERSHIP_IDENTITY_DOMAIN_V1);
    hash.update(layout.identity().as_bytes());
    hash.update(
        packet_set
            .sha256()
            .ok_or_else(|| invalid("distributed row ownership requires a bound packet set"))?,
    );
    hash_usize(&mut hash, layout.cell_count())?;
    hash_usize(&mut hash, layout.partition_count().get())?;
    hash_usize(&mut hash, targets.len())?;
    for target in targets {
        hash_usize(&mut hash, target.global_size().get())?;
        hash_usize(&mut hash, target.partition_count().get())?;
        hash_usize(&mut hash, target.owners().len())?;
        for owner in target.owners() {
            hash_usize(&mut hash, owner.index())?;
        }
    }
    Ok(AdmittedRowOwnershipIdentityV1(hash.finalize().into()))
}
