//! Complete typed Field recovery from the single exact global coordinate map.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecoveredRegionField {
    pub(crate) domain: RawId,
    pub(crate) value_type: eqiora_core::ValueType,
    pub(crate) coefficients: BTreeMap<FieldDof, f64>,
}

impl RegionDofMap {
    /// Recover every requested exact Field, including essential and quotient values.
    /// The request must equal the complete admitted inventory; order is immaterial.
    pub(crate) fn recover(
        &self,
        reduced: &[f64],
        requested: &[RawId],
    ) -> Result<BTreeMap<RawId, RecoveredRegionField>, Diagnostic> {
        let inventory = requested.iter().copied().collect::<BTreeSet<_>>();
        if inventory.len() != requested.len() || inventory != self.fields.keys().copied().collect()
        {
            return Err(invalid(
                "Field recovery requires the complete exact mapped inventory",
            ));
        }
        let full = self.constraints.lift(reduced)?;
        let mut recovered = self
            .fields
            .iter()
            .map(|(&field, (domain, layout))| {
                (
                    field,
                    RecoveredRegionField {
                        domain: *domain,
                        value_type: layout.value_type.clone(),
                        coefficients: BTreeMap::new(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (key, index) in &self.globals {
            let physical = full[*index] * self.fields[&key.field].1.scale;
            if !physical.is_finite() {
                return Err(invalid(
                    "Field recovery produced a nonfinite physical value",
                ));
            }
            recovered
                .get_mut(&key.field)
                .expect("authenticated Field coordinate")
                .coefficients
                .insert(*key, physical);
        }
        if recovered
            .values()
            .any(|field| field.coefficients.is_empty())
        {
            return Err(invalid(
                "Field recovery omitted exact supporting coordinates",
            ));
        }
        Ok(recovered)
    }
}
