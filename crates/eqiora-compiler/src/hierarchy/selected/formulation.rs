//! Attach authored forms after ordinary selected-component specialization.
use super::*;

pub(super) fn attach(
    elaborator: &Elaborator<'_>,
    component: &preflight::ComponentDefinition<'_>,
    bindings: &[(&str, StaticBindingValue<'_>)],
    prepared: &ExternalComponentBinding,
    compiled: CompiledModel,
) -> Result<CompiledModel, Vec<Diagnostic>> {
    if component.formulations().len() != 0 {
        let geometry = bindings.iter().find_map(|(_, value)| match value {
            StaticBindingValue::GeometrySupport { geometry, .. } => Some(*geometry),
            _ => None,
        });
        let coefficients = parameters::resolve_formulation_coefficients(
            component.file,
            component.declaration,
            &parameters::RecordContext::component(elaborator, component),
            &compiled,
        )?;
        let formulations = crate::formulation::compile_component_formulations(
            component.file,
            component.declaration,
            compiled.symbols(),
            compiled.transaction(),
            geometry,
            prepared.supports(),
            coefficients,
        )?;
        Ok(compiled.with_authored_formulations(formulations))
    } else {
        Ok(compiled)
    }
}
