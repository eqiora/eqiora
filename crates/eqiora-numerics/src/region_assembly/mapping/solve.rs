use super::*;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::form_compiler::region::BoundRegionForm;
use crate::region_assembly::{
    InterfaceReactions, PreparedRegionAssembly, RecoveredInterfaceReactions, RegionAssemblyCell,
};
use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyReport,
    AssemblyTarget, LocalContribution, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora_meshing::{AffineGeometryMap, MeshGeometry, QuadratureRule};
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{LinearSolveRequest, SolveReport};

pub(crate) struct RegionSolveOutput<S: Coefficient> {
    pub(crate) fields: BTreeMap<RawId, RecoveredRegionField<S>>,
    pub(crate) solve_report: SolveReport,
    pub(crate) assembly_report: AssemblyReport,
}

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send> RegionDofMap<S> {
    /// Execute an exact mapped static region system. The caller authenticates
    /// Mesh/Geometry lineage; the common owner checks bound Field layouts and
    /// retains physical coefficient interpretation in the recovered inventory.
    pub(crate) fn solve<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        forms: Vec<(BoundRegionForm<S>, QuadratureRule)>,
        natural: Vec<(usize, LocalContribution<S>)>,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, S>,
        complete: impl FnOnce(
            &InterfaceReactions<S>,
            &[S],
        ) -> Result<RecoveredInterfaceReactions<S>, Diagnostic>,
    ) -> Result<RegionSolveOutput<S>, Diagnostic> {
        let dimension = mesh.topological_dimension();
        let domains = &self.cell_domains;
        let expected = forms
            .iter()
            .flat_map(|(form, _)| {
                form.fields()
                    .iter()
                    .map(move |field| (field.field, (form.domain(), field.clone())))
            })
            .collect::<BTreeMap<_, _>>();
        if mesh.entity_count(dimension) != Some(domains.len())
            || expected.len()
                != forms
                    .iter()
                    .map(|(form, _)| form.fields().len())
                    .sum::<usize>()
            || expected != self.fields
        {
            return Err(invalid(
                "region solve differs from the mapped mesh coverage or exact Field layouts",
            ));
        }
        let plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(self.free_count())?,
            AssemblyTarget::new(self.full_count())?,
        ])?;
        let maps = |index| {
            Ok::<_, Diagnostic>(vec![
                TargetAssemblyMap::new(
                    plan.target_id(0).expect("target"),
                    self.cell_map(index, true)?,
                ),
                TargetAssemblyMap::new(
                    plan.target_id(1).expect("full target"),
                    self.cell_map(index, false)?,
                ),
            ])
        };
        let cells = domains
            .iter()
            .enumerate()
            .map(|(index, _)| {
                Ok(RegionAssemblyCell {
                    orientation: self.cell_signs(index)?.to_vec(),
                    index,
                    geometry: mesh
                        .geometry_map(MeshEntity::new(dimension, index))
                        .ok_or_else(|| invalid("region solve has a missing cell geometry"))?,
                    mappings: maps(index)?,
                    previous: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
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
        let reactions = crate::region_assembly::InterfaceReactions::prepare(
            &work,
            plan.target_id(1).expect("full target"),
            self,
            &packet_domains,
        )?;
        let (systems, assembly_report) = REFERENCE_ASSEMBLY_BACKEND
            .assemble(&plan, &work)?
            .into_parts();
        let canonical = Arc::new(eqiora_solver::CanonicalCsrSystemView::new(
            &systems[0],
            eqiora_solver::LinearOperatorProperties::General,
        )?);
        let core = crate::finalized_spatial::FinalizedLinearCore::new(
            request.plan(),
            VectorLayoutKind::Replicated,
            Target::HostCpu { threads: workers },
            canonical,
        );
        let solution = request.solve(&core.linear_problem()?)?;
        core.validate_solution(&solution)?;
        let (values, solve_report) = solution.into_parts();
        complete(&reactions, &self.lift(&values, false)?)?;
        let fields = self.recover(&values, &expected.keys().copied().collect::<Vec<_>>())?;
        Ok(RegionSolveOutput {
            fields,
            solve_report,
            assembly_report,
        })
    }
}
