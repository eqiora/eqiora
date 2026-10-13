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
                    // Every admitted binary32 value is exactly representable in binary64.
                    // The preceding precision tag keeps their contracts distinct.
                    let real = value.re().to_f64().ok_or_else(|| {
                        assembly_failed("fixed real component is not representable")
                    })?;
                    out.extend_from_slice(&real.to_bits().to_be_bytes());
                    if S::DOMAIN == ScalarDomain::Complex {
                        let imaginary = value.im().to_f64().ok_or_else(|| {
                            assembly_failed("fixed imaginary component is not representable")
                        })?;
                        out.extend_from_slice(&imaginary.to_bits().to_be_bytes());
                    }
                }
            }
        }
    }
    Ok(out)
}

pub(super) fn push_usize(out: &mut Vec<u8>, value: usize) -> Result<(), Diagnostic> {
    let value = u64::try_from(value)
        .map_err(|_| assembly_failed("assembly structure exceeds portable u64"))?;
    out.extend_from_slice(&value.to_be_bytes());
    Ok(())
}
