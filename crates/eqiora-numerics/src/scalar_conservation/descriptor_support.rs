use super::*;

pub(super) fn collect_region_parameters(
    region: &ScalarConservationRegion,
    parameters: &mut Vec<Id<kinds::Parameter>>,
) {
    let mut collect = |expression: &ScalarSpatialExpression<f64>| {
        for parameter in expression.parameter_fields() {
            if !parameters.contains(parameter) {
                parameters.push(*parameter);
            }
        }
    };
    collect(&region.flux.coefficient);
    if let Some(storage) = &region.storage {
        collect(&storage.coefficient);
    }
    if let Some(source) = &region.source {
        collect(&source.expression);
    }
    for boundary in region.exterior.values() {
        match &boundary.law {
            ScalarExteriorLaw::PrescribedTrace { value, .. }
            | ScalarExteriorLaw::PrescribedOutwardFlux { value, .. } => collect(value),
            ScalarExteriorLaw::Robin {
                trace_coefficient,
                value,
                ..
            } => {
                collect(trace_coefficient);
                collect(value);
            }
            ScalarExteriorLaw::ZeroOutwardFlux { .. } => {}
        }
    }
    parameters.sort_by_key(|parameter| parameter.erase());
}

pub(super) fn model_error(program: &KernelProgram, message: impl Into<String>) -> Diagnostic {
    model_lowering_error(program, message)
}
