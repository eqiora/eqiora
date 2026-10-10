//! One exact history and orientation projection for ordinary and FSI Region packets.
use super::*;
use crate::form_compiler::region::BoundRegionForm;
use crate::region_assembly::RegionAssemblyCell;
use eqiora_assembly::{AssemblyPlan, TargetAssemblyMap};
use eqiora_meshing::{
    AffineGeometryMap, FixedTopologyGeometryAction, MeshGeometry, QuadratureRule,
};

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    pub(crate) fn assembly_maps(
        &self,
        index: usize,
        plan: &AssemblyPlan,
    ) -> Result<Vec<TargetAssemblyMap<S>>, Diagnostic> {
        Ok(vec![
            TargetAssemblyMap::new(
                plan.target_id(0)
                    .ok_or_else(|| invalid("Region plan lacks reduced target"))?,
                self.cell_map(index, true)?,
            ),
            TargetAssemblyMap::new(
                plan.target_id(1)
                    .ok_or_else(|| invalid("Region plan lacks full target"))?,
                self.cell_map(index, false)?,
            ),
        ])
    }

    pub(crate) fn assembly_cells<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        forms: &[(BoundRegionForm<S>, QuadratureRule)],
        previous: Option<&BTreeMap<RawId, RecoveredRegionField<S>>>,
        geometry_action: Option<&FixedTopologyGeometryAction<2>>,
        plan: &AssemblyPlan,
    ) -> Result<Vec<RegionAssemblyCell<S>>, Diagnostic> {
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
        let step = super::solve::kinematic_step(forms)?;
        let maps = |index| self.assembly_maps(index, plan);
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
                    let rate = step
                        .as_ref()
                        .and_then(|step| {
                            step.eliminated_states()
                                .iter()
                                .find(|state| state.pair().state().erase() == *field)
                                .map(|state| state.pair().rate().erase())
                        })
                        .unwrap_or(*field);
                    for (key, sign) in self.cell_keys[index].iter().zip(self.cell_signs(index)?) {
                        if key.field != rate {
                            continue;
                        }
                        let value = previous
                            .as_ref()
                            .and_then(|fields| fields.get(field))
                            .and_then(|history| {
                                history.coefficients.get(&FieldDof {
                                    field: *field,
                                    ..*key
                                })
                            })
                            .ok_or_else(|| {
                                invalid("region history omits an exact consumed Field coefficient")
                            })?;
                        coefficients.push(*value * <S as From<f64>>::from(f64::from(*sign)));
                    }
                    local_history.insert(*field, coefficients);
                }
                Ok(RegionAssemblyCell {
                    previous_geometry: geometry_action.as_ref().map(|action| {
                        action
                            .cell(index)
                            .expect("exact cell coverage")
                            .previous_map()
                            .clone()
                    }),
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
        Ok(cells)
    }
}
