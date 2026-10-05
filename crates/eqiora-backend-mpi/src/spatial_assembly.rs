use std::cell::RefCell;
use std::fmt;
use std::ops::Range;

use eqiora_assembly::{AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_distributed::PartitionId;
use eqiora_solver::{ExecutionId, ExecutionReport};
use eqiora_spatial_distribution::{
    AdmittedRowOwnership, AssemblyRowRouteDescriptorV1, AssemblyRowRouteV1,
    DistributedAssemblyEvidence, DistributedAssemblyRoutePlanV1, DistributedMeshLayout,
    LocalAssemblyProjection, LocalRouteAdmissionV1, OwnedRowAssemblyResult,
    reconstruct_distributed_assembly,
};
use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::CommunicatorCollectives;
use sha2::{Digest, Sha256};

use crate::MpiExecutionGroup;

const ADMISSION_IDENTITY_DOMAIN_V1: &[u8] = b"eqiora.mpi-spatial-assembly-admission/v1\0";

/// Stable execution identity for MPI owner-routed spatial assembly.
pub const MPI_SPATIAL_ASSEMBLY_EXECUTION: ExecutionId =
    ExecutionId::new("eqiora.mpi.spatial-assembly");

/// MPI transport adapter for the transport-neutral distributed assembly protocol.
///
/// The backend borrows one application-owned execution group exclusively for
/// its complete lifetime. Each `assemble` call is one serialized collective
/// stream; ownership, route admission, canonical folding, shard validation,
/// and reconstruction remain owned by `eqiora-spatial-distribution`.
pub struct MpiSpatialAssemblyBackend<'group> {
    group: RefCell<&'group mut MpiExecutionGroup>,
    layout: DistributedMeshLayout,
    accepted: RefCell<Option<DistributedAssemblyEvidence>>,
}

impl<'group> MpiSpatialAssemblyBackend<'group> {
    /// Collectively bind an exact distributed mesh layout to an MPI group.
    ///
    /// Every rank must call this constructor in the same collective order.
    /// A layout mismatch, including a foreign mesh revision on one rank, is
    /// rejected on every rank before an assembly operation can begin.
    ///
    /// # Errors
    /// Returns `EQ0806` on every rank for a communicator/layout shape or
    /// identity disagreement.
    pub fn new(
        group: &'group mut MpiExecutionGroup,
        layout: DistributedMeshLayout,
    ) -> Result<Self, Diagnostic> {
        let local = if group.partitions() != layout.partition_count() {
            Err(assembly_failed(format!(
                "MPI group has {} partitions but spatial layout declares {}",
                group.partitions(),
                layout.partition_count()
            )))
        } else if group.partition().index() >= layout.partition_count().get() {
            Err(assembly_failed(
                "MPI rank lies outside the distributed spatial layout",
            ))
        } else {
            Ok(())
        };
        collectively(group, local, "layout binding")?;
        agree_identity(
            group,
            layout.identity().as_bytes(),
            "layout identity agreement",
        )?;
        Ok(Self {
            group: RefCell::new(group),
            layout,
            accepted: RefCell::new(None),
        })
    }

    /// Exact mesh layout collectively bound at construction.
    #[must_use]
    pub const fn layout(&self) -> &DistributedMeshLayout {
        &self.layout
    }

    /// Evidence from the latest successful collective call, if any.
    ///
    /// A failed call clears prior evidence. An active operation is rejected
    /// instead of exposing a partial receipt.
    ///
    /// # Errors
    /// Returns `EQ0806` if the operation/evidence cell is already borrowed.
    pub fn accepted_evidence(&self) -> Result<Option<DistributedAssemblyEvidence>, Diagnostic> {
        self.accepted
            .try_borrow()
            .map(|accepted| accepted.clone())
            .map_err(|_| assembly_failed("MPI spatial assembly evidence is currently in use"))
    }
}

impl fmt::Debug for MpiSpatialAssemblyBackend<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MpiSpatialAssemblyBackend")
            .field("layout", &self.layout.identity())
            .finish_non_exhaustive()
    }
}

impl AssemblyBackend<f64> for MpiSpatialAssemblyBackend<'_> {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork<f64>,
    ) -> Result<AssemblyResult<f64>, Diagnostic> {
        let mut accepted = self
            .accepted
            .try_borrow_mut()
            .map_err(|_| assembly_failed("MPI spatial assembly operation is already active"))?;
        *accepted = None;
        let mut group = self
            .group
            .try_borrow_mut()
            .map_err(|_| assembly_failed("MPI spatial assembly group is already in use"))?;
        let (result, evidence) = execute(&mut group, &self.layout, plan, work)?;
        *accepted = Some(evidence);
        Ok(result)
    }
}

fn execute(
    group: &mut MpiExecutionGroup,
    layout: &DistributedMeshLayout,
    plan: &AssemblyPlan,
    work: &dyn AssemblyWork<f64>,
) -> Result<(AssemblyResult<f64>, DistributedAssemblyEvidence), Diagnostic> {
    let admission = collectively(
        group,
        admission_identity(layout, plan, work),
        "fixed admission preparation",
    )?;
    agree_identity(group, admission, "fixed admission agreement")?;

    let projection = collectively(
        group,
        LocalAssemblyProjection::evaluate_owned(layout, plan, work, group.partition()),
        "owner-local projection",
    )?;

    let local_candidates = collectively(
        group,
        projection.collective_candidates(),
        "row-owner candidate preparation",
    )?;
    collectively(
        group,
        portable_mpi_count(local_candidates.values().len(), "row-owner MIN collective").map(|_| ()),
        "row-owner collective count validation",
    )?;
    let mut collective_owners = collectively(
        group,
        zeroed(
            local_candidates.values().len(),
            "collective assembly row-owner workspace",
        ),
        "row-owner workspace preparation",
    )?;
    group.communicator().all_reduce_into(
        local_candidates.values(),
        &mut collective_owners[..],
        SystemOperation::min(),
    );
    let ownership = collectively(
        group,
        AdmittedRowOwnership::from_collective_min(
            layout,
            plan,
            &local_candidates,
            &collective_owners,
        ),
        "row-owner admission",
    )?;
    agree_identity(
        group,
        ownership.identity().as_bytes(),
        "row-owner identity agreement",
    )?;

    let routes = collectively(
        group,
        projection.routes(&ownership),
        "owner route construction",
    )?;
    let descriptor_bytes = collectively(
        group,
        encode_descriptors(&routes),
        "route descriptor encoding",
    )?;
    let gathered_descriptors =
        all_gather_variable_bytes(group, &descriptor_bytes, "route descriptor all-gather")?;
    let descriptors = collectively(
        group,
        decode_descriptors(plan, &gathered_descriptors),
        "route descriptor validation",
    )?;
    let route_plan = collectively(
        group,
        DistributedAssemblyRoutePlanV1::seal(layout, plan, &ownership, descriptors),
        "route plan sealing",
    )?;
    agree_identity(
        group,
        route_plan.identity().as_bytes(),
        "route plan identity agreement",
    )?;
    let local_admission = collectively(
        group,
        route_plan.admit_local_routes(&projection, &ownership, &routes),
        "local route inventory validation",
    )?;
    let admission_bytes = collectively(
        group,
        local_admission.to_bytes(),
        "local route admission encoding",
    )?;
    let gathered_admissions =
        all_gather_fixed_bytes(group, &admission_bytes, "local route admission all-gather")?;
    let admissions = collectively(
        group,
        decode_local_route_admissions(&route_plan, &ownership, &gathered_admissions),
        "local route admission decoding",
    )?;

    let outgoing = collectively(
        group,
        encode_routes_by_destination(&routes, layout.partition_count().get()),
        "route payload framing",
    )?;
    let incoming = all_to_all_variable_bytes(group, outgoing, "route payload all-to-all")?;
    let inbox_routes = collectively(
        group,
        decode_routes(plan, &incoming),
        "route payload decoding",
    )?;
    let local_shards = collectively(
        group,
        route_plan.fold_inbox(&ownership, group.partition(), inbox_routes),
        "destination inbox admission and owner fold",
    )?;

    let local_shard_bytes =
        collectively(group, encode_shards(&local_shards), "owner shard framing")?;
    let gathered_shards =
        all_gather_variable_bytes(group, &local_shard_bytes, "owner shard all-gather")?;
    let shards = collectively(
        group,
        decode_shards(plan, route_plan.identity(), &gathered_shards),
        "owner shard decoding",
    )?;

    let execution =
        ExecutionReport::distributed(MPI_SPATIAL_ASSEMBLY_EXECUTION, layout.partition_count());
    let (systems, evidence) = collectively(
        group,
        reconstruct_distributed_assembly(
            layout,
            plan,
            &ownership,
            &route_plan,
            admissions,
            shards,
            execution,
        ),
        "distributed reconstruction",
    )?;
    agree_identity(
        group,
        evidence.receipt().identity().as_bytes(),
        "final receipt agreement",
    )?;
    let result = collectively(
        group,
        AssemblyResult::from_complete_systems(plan, systems, work.packet_count(), execution),
        "assembly result acceptance",
    )?;
    Ok((result, evidence))
}

fn admission_identity(
    layout: &DistributedMeshLayout,
    plan: &AssemblyPlan,
    work: &dyn AssemblyWork<f64>,
) -> Result<[u8; 32], Diagnostic> {
    if work.packet_count() == 0 || work.packet_count() != layout.cell_count() {
        return Err(assembly_failed(format!(
            "MPI spatial assembly has {} packets for {} mesh cells",
            work.packet_count(),
            layout.cell_count()
        )));
    }
    let packet_set = work.packet_set_identity().sha256().ok_or_else(|| {
        assembly_failed("MPI spatial assembly requires a content-bound packet set")
    })?;
    if packet_set != layout.mesh().as_bytes() {
        return Err(assembly_failed(
            "MPI spatial assembly packet set differs from the exact mesh revision",
        ));
    }
    let mut hash = Sha256::new();
    hash.update(ADMISSION_IDENTITY_DOMAIN_V1);
    hash.update(layout.identity().as_bytes());
    hash.update(packet_set);
    hash_u64(&mut hash, layout.partition_count().get())?;
    hash_u64(&mut hash, layout.cell_count())?;
    hash_u64(&mut hash, work.packet_count())?;
    hash_u64(&mut hash, plan.target_count())?;
    for target_index in 0..plan.target_count() {
        let target_id = plan
            .target_id(target_index)
            .ok_or_else(|| assembly_failed("assembly target disappeared from its plan"))?;
        let target = plan
            .target(target_id)
            .ok_or_else(|| assembly_failed("assembly target disappeared from its plan"))?;
        hash_u64(&mut hash, target.size())?;
    }
    Ok(hash.finalize().into())
}

fn decode_local_route_admissions(
    route_plan: &DistributedAssemblyRoutePlanV1,
    ownership: &AdmittedRowOwnership,
    bytes: &[u8],
) -> Result<Vec<LocalRouteAdmissionV1>, Diagnostic> {
    if !bytes
        .len()
        .is_multiple_of(LocalRouteAdmissionV1::ENCODED_LEN)
    {
        return Err(assembly_failed(
            "collective local-route admission payload is truncated",
        ));
    }
    let mut admissions = reserved(
        bytes.len() / LocalRouteAdmissionV1::ENCODED_LEN,
        "decoded local route admissions",
    )?;
    for (source, record) in bytes
        .as_chunks::<{ LocalRouteAdmissionV1::ENCODED_LEN }>()
        .0
        .iter()
        .enumerate()
    {
        let admission = LocalRouteAdmissionV1::from_bytes(route_plan, ownership, record)?;
        if admission.producer() != PartitionId::new(source) {
            return Err(assembly_failed(
                "local route admission arrived from a foreign producer rank",
            ));
        }
        admissions.push(admission);
    }
    Ok(admissions)
}

fn encode_descriptors(routes: &[AssemblyRowRouteV1]) -> Result<Vec<u8>, Diagnostic> {
    let capacity = routes
        .len()
        .checked_mul(AssemblyRowRouteDescriptorV1::ENCODED_LEN)
        .ok_or_else(|| assembly_failed("route descriptor extent overflows usize"))?;
    let mut bytes = reserved(capacity, "route descriptor payload")?;
    for route in routes {
        bytes.extend_from_slice(&route.descriptor().to_bytes()?);
    }
    Ok(bytes)
}

fn decode_descriptors(
    plan: &AssemblyPlan,
    gathered: &VariableBytes,
) -> Result<Vec<AssemblyRowRouteDescriptorV1>, Diagnostic> {
    for range in &gathered.ranges {
        if range.len() % AssemblyRowRouteDescriptorV1::ENCODED_LEN != 0 {
            return Err(assembly_failed(
                "a rank supplied a partial assembly route descriptor",
            ));
        }
    }
    let count = gathered.bytes.len() / AssemblyRowRouteDescriptorV1::ENCODED_LEN;
    let mut descriptors = reserved(count, "decoded route descriptors")?;
    for (source, range) in gathered.ranges.iter().enumerate() {
        for bytes in gathered.bytes[range.clone()]
            .as_chunks::<{ AssemblyRowRouteDescriptorV1::ENCODED_LEN }>()
            .0
        {
            let descriptor = AssemblyRowRouteDescriptorV1::from_bytes(plan, bytes)?;
            if descriptor.producer() != PartitionId::new(source) {
                return Err(assembly_failed(
                    "assembly route descriptor arrived from a foreign producer rank",
                ));
            }
            descriptors.push(descriptor);
        }
    }
    Ok(descriptors)
}

fn encode_routes_by_destination(
    routes: &[AssemblyRowRouteV1],
    partition_count: usize,
) -> Result<Vec<Vec<u8>>, Diagnostic> {
    let mut outgoing = reserved(partition_count, "route destination buffers")?;
    outgoing.resize_with(partition_count, Vec::new);
    for route in routes {
        let destination = route.descriptor().destination().index();
        let buffer = outgoing
            .get_mut(destination)
            .ok_or_else(|| assembly_failed("route destination exceeds the execution group"))?;
        append_frame(buffer, &route.to_bytes()?)?;
    }
    Ok(outgoing)
}

fn decode_routes(
    plan: &AssemblyPlan,
    incoming: &VariableBytes,
) -> Result<Vec<AssemblyRowRouteV1>, Diagnostic> {
    decode_gathered_frames(incoming, "assembly routes", |source, bytes| {
        let route = AssemblyRowRouteV1::from_bytes(plan, bytes)?;
        if route.descriptor().producer() != PartitionId::new(source) {
            return Err(assembly_failed(
                "assembly route payload arrived from a foreign producer rank",
            ));
        }
        Ok(route)
    })
}

fn encode_shards(shards: &[OwnedRowAssemblyResult]) -> Result<Vec<u8>, Diagnostic> {
    let mut bytes = Vec::new();
    for shard in shards {
        append_frame(&mut bytes, &shard.to_bytes()?)?;
    }
    Ok(bytes)
}

fn decode_shards(
    plan: &AssemblyPlan,
    route_plan: eqiora_spatial_distribution::DistributedAssemblyPlanIdentityV1,
    gathered: &VariableBytes,
) -> Result<Vec<OwnedRowAssemblyResult>, Diagnostic> {
    decode_gathered_frames(gathered, "assembly shards", |source, bytes| {
        let shard = OwnedRowAssemblyResult::from_bytes(plan, route_plan, bytes)?;
        if shard.partition() != PartitionId::new(source) {
            return Err(assembly_failed(
                "assembly owner shard arrived from a foreign producer rank",
            ));
        }
        Ok(shard)
    })
}

fn append_frame(destination: &mut Vec<u8>, payload: &[u8]) -> Result<(), Diagnostic> {
    if payload.is_empty() {
        return Err(assembly_failed("MPI assembly frames cannot be empty"));
    }
    let length = u64::try_from(payload.len())
        .map_err(|_| assembly_failed("MPI assembly frame exceeds portable u64"))?;
    let additional = 8_usize
        .checked_add(payload.len())
        .ok_or_else(|| assembly_failed("MPI assembly frame extent overflows usize"))?;
    destination
        .try_reserve(additional)
        .map_err(|_| assembly_failed("could not reserve MPI assembly frame"))?;
    destination.extend_from_slice(&length.to_be_bytes());
    destination.extend_from_slice(payload);
    Ok(())
}

fn decode_gathered_frames<T>(
    gathered: &VariableBytes,
    purpose: &'static str,
    mut decode: impl FnMut(usize, &[u8]) -> Result<T, Diagnostic>,
) -> Result<Vec<T>, Diagnostic> {
    let mut count = 0_usize;
    for range in &gathered.ranges {
        for_each_frame(&gathered.bytes[range.clone()], |_| {
            count = count
                .checked_add(1)
                .ok_or_else(|| assembly_failed(format!("{purpose} frame count overflows usize")))?;
            Ok(())
        })?;
    }
    let mut decoded = reserved(count, purpose)?;
    for (source, range) in gathered.ranges.iter().enumerate() {
        for_each_frame(&gathered.bytes[range.clone()], |frame| {
            decoded.push(decode(source, frame)?);
            Ok(())
        })?;
    }
    Ok(decoded)
}

fn for_each_frame(
    bytes: &[u8],
    mut visit: impl FnMut(&[u8]) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let header_end = offset
            .checked_add(8)
            .ok_or_else(|| assembly_failed("MPI assembly frame header overflows usize"))?;
        let header = bytes
            .get(offset..header_end)
            .ok_or_else(|| assembly_failed("MPI assembly frame has a truncated header"))?;
        let length = usize::try_from(u64::from_be_bytes(
            header
                .try_into()
                .expect("frame length header has an exact eight-byte shape"),
        ))
        .map_err(|_| assembly_failed("MPI assembly frame length exceeds usize"))?;
        if length == 0 {
            return Err(assembly_failed("MPI assembly frame cannot be empty"));
        }
        let payload_end = header_end
            .checked_add(length)
            .ok_or_else(|| assembly_failed("MPI assembly frame length overflows usize"))?;
        let payload = bytes
            .get(header_end..payload_end)
            .ok_or_else(|| assembly_failed("MPI assembly frame is truncated"))?;
        visit(payload)?;
        offset = payload_end;
    }
    Ok(())
}

#[derive(Debug)]
struct VariableBytes {
    bytes: Vec<u8>,
    ranges: Vec<Range<usize>>,
}

fn all_gather_fixed_bytes<const N: usize>(
    group: &MpiExecutionGroup,
    local: &[u8; N],
    phase: &'static str,
) -> Result<Vec<u8>, Diagnostic> {
    portable_mpi_count(N, phase)?;
    let total = N
        .checked_mul(group.partitions().get())
        .ok_or_else(|| assembly_failed(format!("{phase} extent overflows usize")))?;
    portable_mpi_count(total, phase)?;
    let mut gathered = collectively(
        group,
        zeroed(total, "fixed all-gather receive payload"),
        "fixed all-gather receive allocation",
    )?;
    group
        .communicator()
        .all_gather_into(&local[..], &mut gathered[..]);
    Ok(gathered)
}

fn all_gather_variable_bytes(
    group: &MpiExecutionGroup,
    local: &[u8],
    phase: &'static str,
) -> Result<VariableBytes, Diagnostic> {
    let local_count = collectively(
        group,
        portable_mpi_count(local.len(), phase),
        "variable all-gather local count",
    )?;
    let local_count = u64::try_from(local_count)
        .expect("a validated nonnegative MPI count always fits portable u64");
    let mut gathered_counts = collectively(
        group,
        zeroed(group.partitions().get(), "variable all-gather counts"),
        "variable all-gather count workspace",
    )?;
    group
        .communicator()
        .all_gather_into(&local_count, &mut gathered_counts[..]);
    let (counts, displacements, ranges, total) = collectively(
        group,
        partition_layout(&gathered_counts, phase),
        "variable all-gather layout preparation",
    )?;
    let mut bytes = collectively(
        group,
        zeroed(total, "variable all-gather receive payload"),
        "variable all-gather receive allocation",
    )?;
    let mut receive = PartitionMut::new(&mut bytes[..], counts, displacements);
    group
        .communicator()
        .all_gather_varcount_into(local, &mut receive);
    Ok(VariableBytes { bytes, ranges })
}

fn all_to_all_variable_bytes(
    group: &MpiExecutionGroup,
    outgoing: Vec<Vec<u8>>,
    phase: &'static str,
) -> Result<VariableBytes, Diagnostic> {
    let prepared = collectively(
        group,
        prepare_all_to_all(outgoing, group.partitions().get(), phase),
        "all-to-all send preparation",
    )?;
    let mut receive_counts = collectively(
        group,
        zeroed(group.partitions().get(), "all-to-all receive counts"),
        "all-to-all count workspace",
    )?;
    group
        .communicator()
        .all_to_all_into(&prepared.counts[..], &mut receive_counts[..]);
    let (receive_counts, receive_displacements, ranges, total) = collectively(
        group,
        partition_layout_i32(&receive_counts, phase),
        "all-to-all receive layout preparation",
    )?;
    let mut received = collectively(
        group,
        zeroed(total, "all-to-all receive payload"),
        "all-to-all receive allocation",
    )?;
    let send = Partition::new(&prepared.bytes[..], prepared.counts, prepared.displacements);
    let mut receive = PartitionMut::new(&mut received[..], receive_counts, receive_displacements);
    group
        .communicator()
        .all_to_all_varcount_into(&send, &mut receive);
    Ok(VariableBytes {
        bytes: received,
        ranges,
    })
}

struct PartitionedBytes {
    bytes: Vec<u8>,
    counts: Vec<i32>,
    displacements: Vec<i32>,
}

fn prepare_all_to_all(
    outgoing: Vec<Vec<u8>>,
    partition_count: usize,
    phase: &'static str,
) -> Result<PartitionedBytes, Diagnostic> {
    if outgoing.len() != partition_count {
        return Err(assembly_failed(format!(
            "{phase} has {} destination buffers for {partition_count} ranks",
            outgoing.len()
        )));
    }
    let mut lengths = reserved(outgoing.len(), "all-to-all destination lengths")?;
    lengths.extend(outgoing.iter().map(Vec::len));
    let (counts, displacements, _, total) = partition_layout_usize(&lengths, phase)?;
    let mut bytes = reserved(total, "all-to-all send payload")?;
    for destination in outgoing {
        bytes.extend_from_slice(&destination);
    }
    Ok(PartitionedBytes {
        bytes,
        counts,
        displacements,
    })
}

type PartitionLayout = (Vec<i32>, Vec<i32>, Vec<Range<usize>>, usize);

fn partition_layout(counts: &[u64], phase: &'static str) -> Result<PartitionLayout, Diagnostic> {
    let mut portable = reserved(counts.len(), "MPI portable counts")?;
    for count in counts {
        portable.push(
            usize::try_from(*count)
                .map_err(|_| assembly_failed(format!("{phase} count exceeds usize")))?,
        );
    }
    partition_layout_usize(&portable, phase)
}

fn partition_layout_i32(
    counts: &[i32],
    phase: &'static str,
) -> Result<PartitionLayout, Diagnostic> {
    let mut portable = reserved(counts.len(), "MPI received counts")?;
    for count in counts {
        portable.push(
            usize::try_from(*count)
                .map_err(|_| assembly_failed(format!("{phase} received a negative count")))?,
        );
    }
    partition_layout_usize(&portable, phase)
}

fn partition_layout_usize(
    lengths: &[usize],
    phase: &'static str,
) -> Result<PartitionLayout, Diagnostic> {
    let mut counts = reserved(lengths.len(), "MPI counts")?;
    let mut displacements = reserved(lengths.len(), "MPI displacements")?;
    let mut ranges = reserved(lengths.len(), "MPI source ranges")?;
    let mut total = 0_usize;
    for length in lengths {
        let count = portable_mpi_count(*length, phase)?;
        let displacement = portable_mpi_count(total, phase)?;
        let end = total
            .checked_add(*length)
            .ok_or_else(|| assembly_failed(format!("{phase} extent overflows usize")))?;
        portable_mpi_count(end, phase)?;
        counts.push(count);
        displacements.push(displacement);
        ranges.push(total..end);
        total = end;
    }
    Ok((counts, displacements, ranges, total))
}

fn portable_mpi_count(value: usize, phase: &'static str) -> Result<i32, Diagnostic> {
    i32::try_from(value)
        .map_err(|_| assembly_failed(format!("{phase} extent exceeds MPI i32 count range")))
}

fn collectively<T>(
    group: &MpiExecutionGroup,
    local: Result<T, Diagnostic>,
    phase: &'static str,
) -> Result<T, Diagnostic> {
    let candidate = if local.is_err() {
        u64::try_from(group.partition().index()).expect("an MPI i32 rank always fits portable u64")
    } else {
        u64::MAX
    };
    let mut first_failure = u64::MAX;
    group
        .communicator()
        .all_reduce_into(&candidate, &mut first_failure, SystemOperation::min());
    if first_failure != u64::MAX {
        return Err(assembly_failed(format!(
            "MPI spatial assembly phase {phase} rejected partition {first_failure}"
        )));
    }
    local.map_err(|_| {
        assembly_failed(format!(
            "MPI spatial assembly phase {phase} failed after collective readiness"
        ))
    })
}

fn agree_identity(
    group: &MpiExecutionGroup,
    identity: [u8; 32],
    phase: &'static str,
) -> Result<(), Diagnostic> {
    let words = [
        u64::from_be_bytes(identity[0..8].try_into().expect("fixed identity word")),
        u64::from_be_bytes(identity[8..16].try_into().expect("fixed identity word")),
        u64::from_be_bytes(identity[16..24].try_into().expect("fixed identity word")),
        u64::from_be_bytes(identity[24..32].try_into().expect("fixed identity word")),
    ];
    let mut minimum = [0_u64; 4];
    let mut maximum = [0_u64; 4];
    group
        .communicator()
        .all_reduce_into(&words, &mut minimum, SystemOperation::min());
    group
        .communicator()
        .all_reduce_into(&words, &mut maximum, SystemOperation::max());
    if minimum == maximum {
        Ok(())
    } else {
        Err(assembly_failed(format!(
            "MPI spatial assembly {phase} disagrees across ranks"
        )))
    }
}

fn hash_u64(hash: &mut Sha256, value: usize) -> Result<(), Diagnostic> {
    hash.update(
        u64::try_from(value)
            .map_err(|_| assembly_failed("assembly admission index exceeds portable u64"))?
            .to_be_bytes(),
    );
    Ok(())
}

fn reserved<T>(capacity: usize, purpose: &'static str) -> Result<Vec<T>, Diagnostic> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| assembly_failed(format!("could not reserve {purpose}")))?;
    Ok(values)
}

fn zeroed<T: Clone + Default>(length: usize, purpose: &'static str) -> Result<Vec<T>, Diagnostic> {
    let mut values = reserved(length, purpose)?;
    values.resize(length, T::default());
    Ok(values)
}

fn assembly_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::ASSEMBLY_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_preserves_boundaries() {
        let mut encoded = Vec::new();
        append_frame(&mut encoded, b"alpha").unwrap();
        append_frame(&mut encoded, b"beta").unwrap();
        let mut decoded = Vec::new();
        for_each_frame(&encoded, |frame| {
            decoded.push(frame.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(decoded, [b"alpha".to_vec(), b"beta".to_vec()]);
    }

    #[test]
    fn frame_decoder_rejects_truncation_and_empty_payloads() {
        assert!(for_each_frame(&[0; 7], |_| Ok(())).is_err());
        assert!(for_each_frame(&0_u64.to_be_bytes(), |_| Ok(())).is_err());

        let mut truncated = 4_u64.to_be_bytes().to_vec();
        truncated.extend_from_slice(b"abc");
        assert!(for_each_frame(&truncated, |_| Ok(())).is_err());
    }

    #[test]
    fn partition_layout_rejects_mpi_count_overflow() {
        assert!(partition_layout_usize(&[i32::MAX as usize, 1], "test").is_err());
    }
}
