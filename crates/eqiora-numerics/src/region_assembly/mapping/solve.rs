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

/// Exact local forms, exterior loads, and physical history consumed by one solve.
pub(crate) struct RegionSolveInput<S: Coefficient> {
    pub(crate) forms: Vec<(BoundRegionForm<S>, QuadratureRule)>,
    pub(crate) natural: Vec<(usize, LocalContribution<S>)>,
    pub(crate) previous: Option<BTreeMap<RawId, RecoveredRegionField<S>>>,
}

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send> RegionDofMap<S> {
    /// Execute an exact mapped static region system. The caller authenticates
    /// Mesh/Geometry lineage; the common owner checks bound Field layouts and
    /// retains physical coefficient interpretation in the recovered inventory.
    pub(crate) fn solve<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: RegionSolveInput<S>,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, S>,
        complete: impl FnOnce(
            &InterfaceReactions<S>,
            &[S],
        ) -> Result<RecoveredInterfaceReactions<S>, Diagnostic>,
    ) -> Result<RegionSolveOutput<S>, Diagnostic> {
        let RegionSolveInput {
            forms,
            natural,
            previous,
        } = input;
        if previous.is_some()
            != forms
                .iter()
                .any(|(form, _)| !form.previous_fields().is_empty())
        {
            return Err(invalid(
                "region solve requires history exactly when its forms consume previous Fields",
            ));
        }
        if let Some(previous) = &previous {
            if previous.keys().copied().collect::<BTreeSet<_>>()
                != self.fields.keys().copied().collect()
            {
                return Err(invalid(
                    "region history differs from the complete physical Field inventory",
                ));
            }
            self.validate_physical(previous)?;
        }
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
        let by_domain = forms
            .iter()
            .map(|(form, _)| (form.domain(), form))
            .collect::<BTreeMap<_, _>>();
        if by_domain.len() != forms.len()
            || domains.iter().any(|domain| !by_domain.contains_key(domain))
        {
            return Err(invalid(
                "region solve requires one exact form per owned Domain",
            ));
        }
        let cells = domains
            .iter()
            .enumerate()
            .map(|(index, domain)| {
                let form = by_domain[domain];
                let mut local_history = BTreeMap::new();
                for field in form.previous_fields().keys() {
                    let mut coefficients = Vec::new();
                    for (key, sign) in self.cell_keys[index].iter().zip(self.cell_signs(index)?) {
                        if key.field != *field {
                            continue;
                        }
                        let value = previous
                            .as_ref()
                            .and_then(|fields| fields.get(field))
                            .and_then(|field| field.coefficients.get(key))
                            .ok_or_else(|| {
                                invalid("region history omits an exact consumed Field coefficient")
                            })?;
                        coefficients.push(*value * <S as From<f64>>::from(f64::from(*sign)));
                    }
                    local_history.insert(*field, coefficients);
                }
                Ok(RegionAssemblyCell {
                    orientation: self.cell_signs(index)?.to_vec(),
                    index,
                    geometry: mesh
                        .geometry_map(MeshEntity::new(dimension, index))
                        .ok_or_else(|| invalid("region solve has a missing cell geometry"))?,
                    mappings: maps(index)?,
                    previous: local_history,
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
