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
        let formulations = crate::formulation::compile_formulations(
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

/// Model forms attach after the ordinary original Model has been compiled.
pub(super) fn attach_model(
    elaborator: &Elaborator<'_>,
    model: &preflight::ModelDefinition<'_>,
    bindings: &[(&str, StaticBindingValue<'_>)],
    prepared: &ExternalComponentBinding,
    compiled: CompiledModel,
) -> Result<CompiledModel, Vec<Diagnostic>> {
    if model.declaration.formulations().len() == 0 {
        return Ok(compiled);
    }
    let geometry = bindings.iter().find_map(|(_, value)| match value {
        StaticBindingValue::GeometrySupport { geometry, .. } => Some(*geometry),
        _ => None,
    });
    let coefficients = parameters::resolve_model_formulation_coefficients(
        model.file,
        model.declaration,
        &parameters::RecordContext::model(elaborator, model),
        &compiled,
    )?;
    let forms = crate::formulation::compile_formulations(
        model.file,
        model.declaration,
        compiled.symbols(),
        compiled.transaction(),
        geometry,
        prepared.supports(),
        coefficients,
    )?;
    Ok(compiled.with_authored_formulations(forms))
}
