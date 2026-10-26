//! Prepared cell-to-equation assembly after the caller resolves global constraints.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_assembly::{
    AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyWork, LocalUnknown,
    TargetAssemblyMap,
};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule};

use crate::form_compiler::region::coordinates::{Native, Projection};
use crate::form_compiler::region::{BoundRegionForm, PreparedRegionCell, PreparedRegionFacet};
use crate::spatial_expression::Coefficient;

mod conservative;
pub(crate) mod mapping;
pub(crate) use conservative::ConservativeFacets;
mod newton;
pub(crate) use newton::RegionNewtonSolution;
pub(crate) mod nonlinear;
mod point;
mod q1_diffusion;
pub(crate) use point::RegionPoint;
mod reactions;
pub(crate) mod residual;
#[cfg(test)]
pub(crate) use reactions::DomainReactions;
pub(crate) use reactions::{
    InterfaceReactionProjection, InterfaceReactions, RecoveredDomainReactions,
    RecoveredInterfaceReactions,
};

/// One cell's geometry, resolved algebraic maps and physical previous coefficients.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegionAssemblyCell<S: Coefficient> {
    pub(crate) index: usize,
    /// Local functional orientation relative to each shared global coefficient.
    pub(crate) orientation: Vec<i8>,
    pub(crate) geometry: AffineGeometryMap,
    pub(crate) previous_geometry: Option<AffineGeometryMap>,
    pub(crate) mappings: std::sync::Arc<[TargetAssemblyMap<S>]>,
    pub(crate) previous: BTreeMap<RawId, Vec<S>>,
}

/// A local action owns its equation/unknown maps; it need not belong to one cell.
#[derive(Debug, Clone, PartialEq)]
struct PreparedRegionPacket<S: Coefficient, P: Projection<S>> {
    domain: RawId,
    orientation: Vec<i8>,
    mappings: std::sync::Arc<[TargetAssemblyMap<P::Scalar>]>,
    action: RegionPacketAction<S, P>,
}

#[derive(Debug, Clone, PartialEq)]
enum RegionPacketAction<S: Coefficient, P: Projection<S>> {
    Cell {
        prepared: PreparedRegionCell<S, P>,
        previous: BTreeMap<RawId, Vec<P::Scalar>>,
    },
    Boundary(PreparedRegionFacet<S, P>),
    Affine(eqiora_assembly::LocalContribution<P::Scalar>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedRegionAssembly<S: Coefficient, P: Projection<S> = Native> {
    packet_set: AssemblyPacketSetIdentityV1,
    packets: Vec<PreparedRegionPacket<S, P>>,
}

impl<S: Coefficient + Send + Sync, P: Projection<S>> PreparedRegionAssembly<S, P>
where
    P::Scalar: Send + Sync,
{
    /// The Domain sequence is in the authenticated mesh's logical packet order.
    /// Maps already include Plan DOF binding and any trace/essential constraints.
    pub(crate) fn new(
        packet_set: AssemblyPacketSetIdentityV1,
        plan: &AssemblyPlan,
        forms: Vec<(BoundRegionForm<S>, QuadratureRule)>,
        cell_domains: &[RawId],
        mut cells: Vec<RegionAssemblyCell<P::Scalar>>,
        boundary_packets: Vec<(usize, PreparedRegionFacet<S, P>)>,
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
        let packet_count = cells
            .len()
            .checked_add(boundary_packets.len())
            .ok_or_else(|| invalid("region and boundary packet count overflows usize"))?;
        let mut packets = Vec::with_capacity(packet_count);
        cells.sort_by_key(|cell| cell.index);
        for (index, cell) in cells.iter_mut().enumerate() {
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
            form.validate_cell::<P>(&cell.geometry, quadrature, &cell.previous)?;
            let local_count = P::ranges(form.fields())?
                .last()
                .expect("bound nonempty form")
                .end;
            if cell.orientation.len() != local_count
                || cell.orientation.iter().any(|sign| !matches!(sign, -1 | 1))
            {
                return Err(invalid(
                    "prepared cell requires exact local coefficient orientations",
                ));
            }
            validate_maps(plan, local_count, &cell.mappings)?;
            let prepared = form
                .prepare_projected_cell_actions::<P>(
                    &cell.geometry,
                    cell.previous_geometry.as_ref().unwrap_or(&cell.geometry),
                    None,
                    quadrature,
                )?
                .0;
            packets.push(PreparedRegionPacket {
                domain: cell_domains[index],
                orientation: cell.orientation.clone(),
                mappings: cell.mappings.clone(),
                action: RegionPacketAction::Cell {
                    prepared,
                    previous: std::mem::take(&mut cell.previous),
                },
            });
        }
        for (index, facet) in boundary_packets {
            let cell = cells
                .get(index)
                .ok_or_else(|| invalid("natural facet has a foreign parent cell"))?;
            facet.validate_parent(&by_domain[&cell_domains[index]].0, &cell.geometry)?;
            validate_maps(plan, facet.affine().rows(), &cell.mappings)?;
            packets.push(PreparedRegionPacket {
                domain: cell_domains[index],
                orientation: cell.orientation.clone(),
                mappings: cell.mappings.clone(),
                action: RegionPacketAction::Boundary(facet),
            });
        }
        Ok(Self {
            packet_set,
            packets,
        })
    }
    pub(crate) fn packet_domains(&self) -> Vec<RawId> {
        self.packets.iter().map(|packet| packet.domain).collect()
    }

    fn packet(&self, index: usize) -> Result<&PreparedRegionPacket<S, P>, Diagnostic> {
        self.packets
            .get(index)
            .ok_or_else(|| invalid("region assembly packet is outside the prepared mesh"))
    }
}

impl<S: Coefficient + Send + Sync, P: Projection<S>> AssemblyWork<P::Scalar>
    for PreparedRegionAssembly<S, P>
where
    P::Scalar: Send + Sync,
{
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.packet_set
    }

    fn packet_count(&self) -> usize {
        self.packets.len()
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket<P::Scalar>, Diagnostic> {
        let packet = self.packet(packet_index)?;
        let local = match &packet.action {
            RegionPacketAction::Cell { prepared, previous } => prepared.evaluate(previous)?,
            RegionPacketAction::Boundary(facet) => facet.affine().clone(),
            RegionPacketAction::Affine(local) => local.clone(),
        };
        AssemblyPacket::new(
            local.reoriented(&packet.orientation, &packet.orientation)?,
            packet.mappings.clone(),
        )
    }
}

fn validate_maps<S: Coefficient + Sync>(
    plan: &AssemblyPlan,
    count: usize,
    mappings: &[TargetAssemblyMap<S>],
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
