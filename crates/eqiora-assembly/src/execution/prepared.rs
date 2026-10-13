//! Immutable packet projection and CSR destinations owned by AssemblyPlan.
use super::*;
use crate::projection::CompiledProjection;
use crate::sparse::CsrScatterSlots;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedTargetProjection {
    pub(super) target: AssemblyTargetId,
    pub(super) projection: CompiledProjection,
    pub(super) slots: CsrScatterSlots,
}
impl PreparedTargetProjection {
    pub(super) fn new<S: Scalar + ComplexFloat + Sync>(
        mapping: &TargetAssemblyMap<S>,
        topology: &CsrTopology,
    ) -> Result<Self, Diagnostic> {
        let projection = CompiledProjection::new(topology.size, &mapping.map)?;
        let slots = CsrScatterSlots::new(topology, &projection)?;
        Ok(Self {
            target: mapping.target,
            projection,
            slots,
        })
    }
}

pub(super) fn build_topology<S: Scalar + ComplexFloat + Sync>(
    target: usize,
    size: usize,
    packets: &[AssemblyPacketStructure<S>],
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

pub(super) fn encode_packet_structure<S: Scalar + ComplexFloat + Sync>(
    packet: &AssemblyPacketStructure<S>,
) -> Result<Vec<u8>, Diagnostic> {
    // Domain and component precision participate even when every slot is free.
    let mut out = vec![match (S::DOMAIN, S::STORAGE) {
        (ScalarDomain::Real, ScalarType::F32) => 1,
        (ScalarDomain::Real, ScalarType::F64) => 2,
        (ScalarDomain::Complex, ScalarType::F32) => 3,
        (ScalarDomain::Complex, ScalarType::F64) => 4,
        _ => return Err(assembly_failed("unsupported assembly coefficient encoding")),
    }];
    push_usize(&mut out, packet.rows)?;
    push_usize(&mut out, packet.columns)?;
    push_usize(&mut out, packet.mappings.len())?;
    for mapping in &packet.mappings {
        push_usize(&mut out, mapping.target.index())?;
        out.extend_from_slice(mapping.map.binding());
    }
    Ok(out)
}

/// Scalar-independent authority captured from immutable, exactly encoded maps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedPacketBinding {
    rows: usize,
    columns: usize,
    mappings: Vec<(AssemblyTargetId, Arc<[u8]>)>,
}

impl PreparedPacketBinding {
    pub(super) fn new<S: Scalar + ComplexFloat + Sync>(
        packet: &AssemblyPacketStructure<S>,
    ) -> Self {
        Self {
            rows: packet.rows,
            columns: packet.columns,
            mappings: packet
                .mappings
                .iter()
                .map(|mapping| (mapping.target, mapping.map.binding().clone()))
                .collect(),
        }
    }

    pub(super) fn matches<S: Scalar + ComplexFloat + Sync>(
        &self,
        packet: &AssemblyPacket<S>,
    ) -> bool {
        self.rows == packet.local.rows()
            && self.columns == packet.local.columns()
            && self.mappings.len() == packet.mappings.len()
            && self
                .mappings
                .iter()
                .zip(&packet.mappings)
                .all(|((target, binding), mapping)| {
                    *target == mapping.target && binding == mapping.map.binding()
                })
    }
}
