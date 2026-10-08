use super::CompiledLinearBlockForm;
use crate::spatial_expression::Coefficient;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};

impl<S: Coefficient> CompiledLinearBlockForm<S> {
    pub(crate) fn bind_parameter_point(
        &self,
        fields: &[Id<kinds::Parameter>],
        values: &[S],
    ) -> Result<Self, Diagnostic> {
        let mut bound = self.clone();
        bound.volume = self.volume.bind_parameter_point(fields, values)?;
        for law in bound
            .boundary_laws
            .values_mut()
            .flat_map(|laws| laws.values_mut())
        {
            law.bind_parameter_point(fields, values)?;
        }
        for value in bound.storage.values_mut().chain(bound.initial.values_mut()) {
            *value = value.bind_parameter_point(fields, values)?;
        }
        bound.validate_storage()?;
        Ok(bound)
    }
}
