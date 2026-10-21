//! Complete typed values bound to independent scalar coordinates in a local scope.
use super::*;
use eqiora_core::ValueType;

#[derive(Clone)]
pub(super) struct ScopedValue {
    value_type: ValueType,
    coordinates: Vec<ScalarInputValueId>,
}

impl ScopedValue {
    pub(super) fn new(
        value_type: ValueType,
        coordinates: Vec<ScalarInputValueId>,
    ) -> Result<Self, Diagnostic> {
        let parts = ScalarPart::for_domain(value_type.scalar_domain())?;
        let count = value_type
            .shape()
            .component_count()
            .and_then(|count| count.checked_mul(parts.len()))
            .ok_or_else(|| invalid_component_ir("scoped input coordinate count overflow"))?;
        if coordinates.len() != count {
            return Err(invalid_component_ir(
                "scoped input coordinates differ from their complete value type",
            ));
        }
        Ok(Self {
            value_type,
            coordinates,
        })
    }

    pub(super) fn value_type(&self) -> &ValueType {
        &self.value_type
    }

    pub(super) fn coordinates(&self) -> &[ScalarInputValueId] {
        &self.coordinates
    }

    pub(super) fn component(
        &self,
        component: &[u32],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        let parts = ScalarPart::for_domain(self.value_type.scalar_domain())?;
        let flat = self.value_type.shape().flat_index(component);
        let part = parts.iter().position(|candidate| *candidate == part);
        flat.zip(part)
            .and_then(|(flat, part)| self.coordinates.get(flat * parts.len() + part))
            .copied()
            .ok_or_else(|| {
                invalid_component_ir("scoped input component differs from its exact type")
            })
    }

    pub(super) fn remap(
        &self,
        map: impl FnMut(ScalarInputValueId) -> Result<ScalarInputValueId, Diagnostic>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            value_type: self.value_type.clone(),
            coordinates: self
                .coordinates
                .iter()
                .copied()
                .map(map)
                .collect::<Result<_, _>>()?,
        })
    }
}
