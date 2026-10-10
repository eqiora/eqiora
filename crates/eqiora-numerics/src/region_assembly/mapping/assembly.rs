//! Shared preparation for solving and checking a supplied algebraic State.
use super::*;
use crate::form_compiler::region::BoundRegionForm;
use crate::region_assembly::PreparedRegionAssembly;
use eqiora_assembly::{
    AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTarget, LocalContribution,
};
use eqiora_meshing::{
    AffineGeometryMap, FixedTopologyGeometryAction, MeshGeometry, QuadratureRule,
};

pub(super) struct MappedRegionAssembly<S: Coefficient> {
    pub(super) plan: AssemblyPlan,
    pub(super) work: PreparedRegionAssembly<S>,
    pub(super) packet_domains: Vec<RawId>,
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    pub(super) fn prepare_assembly<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        forms: Vec<(BoundRegionForm<S>, QuadratureRule)>,
        natural: Vec<(usize, LocalContribution<S>)>,
        previous: Option<&BTreeMap<RawId, RecoveredRegionField<S>>>,
        geometry_action: Option<&FixedTopologyGeometryAction<2>>,
    ) -> Result<MappedRegionAssembly<S>, Diagnostic> {
        let domains = &self.cell_domains;
        let plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(self.free_count())?,
            AssemblyTarget::new(self.full_count())?,
        ])?;
        let maps = |index| self.assembly_maps(index, &plan);
        let cells = self.assembly_cells(mesh, &forms, previous, geometry_action, &plan)?;
        let mut packet_domains = domains.to_vec();
        let packets = natural
            .into_iter()
            .map(|(index, local)| {
                packet_domains.push(
                    *domains
                        .get(index)
                        .ok_or_else(|| invalid("natural packet has a foreign cell"))?,
                );
                let signs = self.cell_signs(index)?;
                AssemblyPacket::new(local.reoriented(signs, signs)?, maps(index)?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let work = PreparedRegionAssembly::new(
            AssemblyPacketSetIdentityV1::Unbound,
            &plan,
            forms,
            domains,
            cells,
            packets,
        )?;
        Ok(MappedRegionAssembly {
            plan,
            work,
            packet_domains,
        })
    }
}
