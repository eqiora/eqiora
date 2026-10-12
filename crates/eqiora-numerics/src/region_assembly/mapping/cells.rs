//! One exact history and orientation projection for ordinary and FSI Region packets.
use super::*;
use crate::form_compiler::region::BoundRegionForm;
use crate::region_assembly::RegionAssemblyCell;
use eqiora_assembly::{AssemblyPlan, TargetAssemblyMap};
use eqiora_meshing::{
    AffineGeometryMap, FixedTopologyGeometryAction, MeshGeometry, QuadratureRule,
};

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    /// Gather physical coefficients in the same oriented local basis order as assembly.
    /// An eliminated state uses its rate's coordinate layout, without its scale.
    pub(crate) fn physical_cell_coefficients(
        &self,
        index: usize,
        field: RawId,
        layout_field: RawId,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
    ) -> Result<Vec<S>, Diagnostic> {
        let keys = self
            .cell_keys
            .get(index)
            .ok_or_else(|| invalid("physical coefficient gather has a foreign cell"))?;
        let coefficients = fields
            .get(&field)
            .ok_or_else(|| invalid("region history omits an exact consumed Field"))?;
        let mut values = Vec::new();
        for (key, sign) in keys.iter().zip(self.cell_signs(index)?) {
            if key.field != layout_field {
                continue;
            }
            let value = coefficients
                .coefficients
                .get(&FieldDof { field, ..*key })
                .ok_or_else(|| {
                    invalid("region history omits an exact consumed Field coefficient")
                })?;
            values.push(*value * <S as From<f64>>::from(f64::from(*sign)));
        }
        if values.is_empty() {
            return Err(invalid(
                "physical coefficient gather has a foreign Field layout",
            ));
        }
        Ok(values)
    }

    /// Gather the exact consumed Fields through their own or eliminated-rate layouts.
    pub(crate) fn physical_cell_history(
        &self,
        index: usize,
        form: &BoundRegionForm<S>,
        previous: Option<&BTreeMap<RawId, RecoveredRegionField<S>>>,
    ) -> Result<BTreeMap<RawId, Vec<S>>, Diagnostic> {
        form.previous_fields()
            .keys()
            .map(|&field| {
                let rate = form
                    .time_binding()
                    .and_then(|time| {
                        time.states
                            .iter()
                            .find(|state| state.pair().state().erase() == field)
                    })
                    .map_or(field, |state| state.pair().rate().erase());
                let fields = previous.ok_or_else(|| {
                    invalid("region history omits an exact consumed Field coefficient")
                })?;
                self.physical_cell_coefficients(index, field, rate, fields)
                    .map(|values| (field, values))
            })
            .collect()
    }

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
        let _ = super::solve::kinematic_step(forms)?;
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
                let local_history = self.physical_cell_history(index, form, previous)?;
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
