//! The complete physical State inventory includes eliminated kinematic Fields.
use super::*;
use crate::region_assembly::mapping::{FieldDof, RecoveredRegionField, RegionDofMap};

impl CommonLinearPlan {
    pub(super) fn history_inventory(
        &self,
        mapping: &RegionDofMap<f64>,
    ) -> Result<BTreeMap<eqiora_core::RawId, RecoveredRegionField<f64>>, Diagnostic> {
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("linear State lacks its exact equations"));
        };
        let mut fields = BTreeMap::new();
        for region in &equations.regions {
            for (field, value_type) in region.form.represented_fields() {
                let rate = region
                    .form
                    .kinematics()
                    .iter()
                    .find(|(pair, _)| pair.state().erase() == field)
                    .map(|(pair, _)| pair.rate().erase())
                    .unwrap_or(field);
                let (domain, layout) = mapping.field_layout(rate).ok_or_else(|| {
                    invalid("represented State Field lacks its exact algebraic rate")
                })?;
                let coefficients = mapping
                    .keys()
                    .filter(|key| key.field == rate)
                    .map(|key| (FieldDof { field, ..key }, 0.))
                    .collect();
                fields.insert(
                    field,
                    RecoveredRegionField {
                        domain,
                        value_type,
                        space: layout.space,
                        coefficients,
                    },
                );
            }
        }
        Ok(fields)
    }

    pub(super) fn history_fields(
        &self,
        mapping: &RegionDofMap<f64>,
        values: &[f64],
    ) -> Result<BTreeMap<eqiora_core::RawId, RecoveredRegionField<f64>>, Diagnostic> {
        let mut fields = self.history_inventory(mapping)?;
        if values.len()
            != fields
                .values()
                .map(|field| field.coefficients.len())
                .sum::<usize>()
        {
            return Err(invalid(
                "linear history requires complete finite nodal coefficients in its exact mapped coefficient inventory",
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "history is nonfinite; State requires complete finite nodal coefficients",
            ));
        }
        for (target, value) in fields
            .values_mut()
            .flat_map(|field| field.coefficients.values_mut())
            .zip(values)
        {
            *target = *value;
        }
        mapping.validate_physical(&fields)?;
        Ok(fields)
    }
}
