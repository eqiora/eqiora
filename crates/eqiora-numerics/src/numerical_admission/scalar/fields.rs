//! Complete physical field coordinates shared by State, Result and observations.
use super::*;
use crate::region_assembly::mapping::{FieldDof, RecoveredRegionField, RegionDofMap};

impl CommonSpatialPlan {
    /// Borrow each accepted State Field in its exact Mesh-entity/component order.
    ///
    /// The Plan owns Field identity, physical type and coefficient layout. Values
    /// borrow the immutable State; they are coefficients, not physical samples.
    /// No reconstruction or derived observation is evaluated here.
    /// # Errors
    /// Rejects a foreign State, unsupported state storage, or incomplete layout.
    #[allow(clippy::type_complexity)]
    pub fn state_fields<'a>(
        &'a self,
        state: &'a CommonState,
    ) -> Result<
        impl ExactSizeIterator<
            Item = (
                eqiora_core::Id<eqiora_core::entity::kinds::Field>,
                &'a eqiora_core::ValueType,
                Vec<eqiora_meshing::MeshEntity>,
                &'a [f64],
            ),
        >,
        Diagnostic,
    > {
        if state.state_space_identity() != self.identity() {
            return Err(invalid(
                "State Field projection requires its exact owning Plan",
            ));
        }
        let mut remaining = state
            .spatial_values()
            .ok_or_else(|| invalid("State Field projection requires spatial coefficients"))?;
        let mut fields = Vec::with_capacity(self.fields().len());
        for (field, value_type) in self.fields() {
            let entities = self.field_coefficient_entities(field)?;
            let count = value_type
                .shape()
                .component_count()
                .and_then(|components| entities.len().checked_mul(components))
                .ok_or_else(|| invalid("State Field coefficient count overflows"))?;
            let (values, rest) = remaining.split_at_checked(count).ok_or_else(|| {
                invalid("State coefficients differ from their exact Field layout")
            })?;
            fields.push((field, value_type, entities, values));
            remaining = rest;
        }
        if !remaining.is_empty() {
            return Err(invalid("State contains unowned spatial coefficients"));
        }
        Ok(fields.into_iter())
    }

    pub(super) fn physical_field_inventory(
        &self,
        mapping: &RegionDofMap<f64>,
    ) -> Result<BTreeMap<eqiora_core::RawId, RecoveredRegionField<f64>>, Diagnostic> {
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid("spatial State lacks its exact equations"));
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

    pub(super) fn physical_fields(
        &self,
        mapping: &RegionDofMap<f64>,
        values: &[f64],
    ) -> Result<BTreeMap<eqiora_core::RawId, RecoveredRegionField<f64>>, Diagnostic> {
        let mut fields = self.physical_field_inventory(mapping)?;
        if values.len()
            != fields
                .values()
                .map(|field| field.coefficients.len())
                .sum::<usize>()
        {
            return Err(invalid(
                "physical fields require complete coefficients in their exact mapped inventory",
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "physical fields require complete finite coefficients",
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
