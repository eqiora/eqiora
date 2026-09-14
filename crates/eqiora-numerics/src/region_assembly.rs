//! Prepared cell-to-equation assembly after the caller resolves global constraints.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_assembly::{
    AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyWork, LocalUnknown,
    TargetAssemblyMap,
};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule};

use crate::form_compiler::region::BoundRegionForm;

pub(crate) mod mapping;
mod reactions;
#[cfg(test)]
pub(crate) use reactions::DomainReactions;
pub(crate) use reactions::{InterfaceReactions, RecoveredInterfaceReactions};

/// One cell's geometry, resolved algebraic maps and physical previous coefficients.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegionAssemblyCell {
    pub(crate) index: usize,
    pub(crate) geometry: AffineGeometryMap,
    pub(crate) mappings: Vec<TargetAssemblyMap>,
    pub(crate) previous: BTreeMap<RawId, Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedRegionAssembly {
    packet_set: AssemblyPacketSetIdentityV1,
    forms: BTreeMap<RawId, (BoundRegionForm, QuadratureRule)>,
    cell_domains: Vec<RawId>,
    cells: Vec<RegionAssemblyCell>,
    boundary_packets: Vec<AssemblyPacket>,
}

impl PreparedRegionAssembly {
    /// The Domain sequence is in the authenticated mesh's logical packet order.
    /// Maps already include Plan DOF binding and any trace/essential constraints.
    pub(crate) fn new(
        packet_set: AssemblyPacketSetIdentityV1,
        plan: &AssemblyPlan,
        forms: Vec<(BoundRegionForm, QuadratureRule)>,
        cell_domains: &[RawId],
        mut cells: Vec<RegionAssemblyCell>,
        boundary_packets: Vec<AssemblyPacket>,
    ) -> Result<Self, Diagnostic> {
        if cell_domains.is_empty() || cells.len() != cell_domains.len() {
            return Err(invalid(
                "region assembly needs exact nonempty mesh cell coverage",
            ));
        }
        let mut by_domain = BTreeMap::new();
        for (form, quadrature) in forms {
            if form.reference_cell() != quadrature.reference_cell() {
                return Err(invalid("region form and quadrature references differ"));
            }
            if by_domain
                .insert(form.domain(), (form, quadrature))
                .is_some()
            {
                return Err(invalid("duplicate assembly form for one exact Domain"));
            }
        }
        if by_domain.keys().copied().collect::<BTreeSet<_>>()
            != cell_domains.iter().copied().collect::<BTreeSet<_>>()
        {
            return Err(invalid(
                "assembly forms must exactly cover selected cell Domains",
            ));
        }
        cells.sort_by_key(|cell| cell.index);
        for (index, cell) in cells.iter().enumerate() {
            if cell.index != index {
                return Err(invalid(
                    "region assembly cell indices must cover mesh packets exactly once",
                ));
            }
            let (form, quadrature) = &by_domain[&cell_domains[index]];
            if cell.geometry.reference_cell() != form.reference_cell() {
                return Err(invalid(
                    "cell reference differs from its selected Domain form",
                ));
            }
            form.validate_cell(&cell.geometry, quadrature, &cell.previous)?;
            let local_count = form.fields().last().expect("bound nonempty form").range.end;
            validate_maps(plan, local_count, &cell.mappings)?;
        }
        cells
            .len()
            .checked_add(boundary_packets.len())
            .ok_or_else(|| invalid("region and boundary packet count overflows usize"))?;
        for packet in &boundary_packets {
            if packet.local().rows() != packet.local().columns() {
                return Err(invalid(
                    "boundary packet requires matching test and trial ranges",
                ));
            }
            validate_maps(plan, packet.local().rows(), packet.mappings())?;
        }
        Ok(Self {
            packet_set,
            forms: by_domain,
            cell_domains: cell_domains.to_vec(),
            cells,
            boundary_packets,
        })
    }
}

impl AssemblyWork for PreparedRegionAssembly {
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.packet_set
    }

    fn packet_count(&self) -> usize {
        self.cells.len() + self.boundary_packets.len()
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic> {
        if packet_index >= self.cells.len() {
            return self
                .boundary_packets
                .get(packet_index - self.cells.len())
                .cloned()
                .ok_or_else(|| invalid("region assembly packet is outside the prepared mesh"));
        }
        let cell = self
            .cells
            .get(packet_index)
            .ok_or_else(|| invalid("region assembly packet is outside the prepared mesh"))?;
        let (form, quadrature) = &self.forms[&self.cell_domains[packet_index]];
        AssemblyPacket::new(
            form.evaluate(&cell.geometry, quadrature, &cell.previous)?,
            cell.mappings.clone(),
        )
    }
}

fn validate_maps(
    plan: &AssemblyPlan,
    count: usize,
    mappings: &[TargetAssemblyMap],
) -> Result<(), Diagnostic> {
    if mappings.is_empty() {
        return Err(invalid("prepared cell needs an assembly target map"));
    }
    let mut targets = BTreeSet::new();
    for mapping in mappings {
        if !targets.insert(mapping.target()) {
            return Err(invalid("prepared cell has duplicate assembly targets"));
        }
        let target = plan
            .target(mapping.target())
            .ok_or_else(|| invalid("prepared cell target is outside the assembly Plan"))?;
        let map = mapping.map();
        if map.equations().len() != count || map.unknowns().len() != count {
            return Err(invalid(
                "prepared cell map differs from exact local Field ranges",
            ));
        }
        if map.equations().iter().flatten().any(|dof| dof.index() >= target.size())
            || map.unknowns().iter().any(|unknown| matches!(unknown, LocalUnknown::Free(dof) if dof.index() >= target.size()))
        {
            return Err(invalid("prepared cell map has a global DOF outside its target"));
        }
    }
    Ok(())
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        message,
    )
}

#[cfg(test)]
mod tests;
