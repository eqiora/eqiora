//! Shared preparation for assembly, solving and checking a supplied algebraic State.
use super::*;
use crate::form_compiler::region::BoundRegionForm;
use crate::region_assembly::PreparedRegionAssembly;
use eqiora_assembly::{AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTarget};
use eqiora_meshing::{
    AffineGeometryMap, FixedTopologyGeometryAction, MeshGeometry, QuadratureRule,
};

pub(crate) struct MappedRegionAssembly<S: Coefficient> {
    pub(crate) plan: AssemblyPlan,
    pub(crate) work: PreparedRegionAssembly<S>,
    pub(crate) packet_domains: Vec<RawId>,
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    /// Reduced and full targets share the exact constrained Field layout.
    pub(crate) fn assembly_plan(&self) -> Result<AssemblyPlan, Diagnostic> {
        AssemblyPlan::new(vec![
            AssemblyTarget::new(self.free_count())?,
            AssemblyTarget::new(self.full_count())?,
        ])
    }

    pub(crate) fn prepare_assembly<'mesh>(
        &self,
        packet_set: AssemblyPacketSetIdentityV1,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        forms: Vec<(BoundRegionForm<S>, QuadratureRule)>,
        natural: Vec<(usize, crate::form_compiler::region::PreparedRegionFacet<S>)>,
        previous: Option<&BTreeMap<RawId, RecoveredRegionField<S>>>,
        geometry_action: Option<&FixedTopologyGeometryAction<2>>,
    ) -> Result<MappedRegionAssembly<S>, Diagnostic> {
        let domains = &self.cell_domains;
        let plan = self.assembly_plan()?;
        let cells = self.assembly_cells(mesh, &forms, previous, geometry_action, &plan)?;
        let mut packet_domains = domains.to_vec();
        for (index, _) in &natural {
            packet_domains.push(
                *domains
                    .get(*index)
                    .ok_or_else(|| invalid("natural packet has a foreign cell"))?,
            );
        }
        let work = PreparedRegionAssembly::new(packet_set, &plan, forms, domains, cells, natural)?;
        Ok(MappedRegionAssembly {
            plan,
            work,
            packet_domains,
        })
    }
}
