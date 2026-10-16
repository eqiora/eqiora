//! Structural identity of derived values, probability variables and their measures.
use super::*;
use eqiora_schema::kernel::{
    ObservableDef, ObservableMeasure, ObservableReduction, ProbabilityMoment,
};

pub(super) fn encode(
    definition: &ObservableDef,
    encoder: &mut Encoder,
    ids: &BTreeMap<RawId, usize>,
    references: &mut Vec<Reference>,
    budget: &mut ConstructionBudget,
) -> Result<(), Diagnostic> {
    encoder.u8(15)?;
    encode_value_type(encoder, definition.value_type())?;
    type_reference(
        definition.value_type(),
        nominal_label(15),
        ids,
        references,
        budget,
    )?;
    match definition.reduction() {
        eqiora_schema::kernel::ObservableReduction::Value => encoder.u8(0)?,
        ObservableReduction::RandomVariable { distribution } => {
            encoder.u8(2)?;
            push_reference(
                references,
                nominal_label(18),
                lookup(ids, distribution.erase(), "probability law")?,
                budget,
            )?;
        }
        ObservableReduction::ProbabilityMoment {
            distribution,
            moment,
        } => {
            encoder.u8(3)?;
            encoder.u8(match moment {
                ProbabilityMoment::Expectation => 0,
                ProbabilityMoment::Variance => 1,
                ProbabilityMoment::Covariance => 2,
            })?;
            push_reference(
                references,
                nominal_label(18),
                lookup(ids, distribution.erase(), "probability law")?,
                budget,
            )?;
        }
        eqiora_schema::kernel::ObservableReduction::SpatialIntegral {
            input,
            domain,
            measure,
            ..
        } => {
            encoder.u8(1)?;
            encoder.u8(measure_tag(measure))?;
            push_reference(
                references,
                nominal_label(17),
                lookup(ids, input.erase(), "Observable input Domain")?,
                budget,
            )?;
            push_reference(
                references,
                nominal_label(16),
                lookup(ids, domain.erase(), "Observable integration Domain")?,
                budget,
            )?;
        }
    }
    let limits = definition.reduction().limits();
    let canonical_index = encode_expression(
        encoder,
        definition.expression(),
        limits.as_ref().map_or(&[], |limits| limits.as_slice()),
        4,
        ids,
        references,
        budget,
    )?;
    encoder.u8(u8::from(limits.is_some()))?;
    if let Some(limits) = limits {
        for limit in limits {
            encoder.u32(canonical_expr_id(limit, &canonical_index)?)?;
        }
    }

    Ok(())
}

pub(super) const fn measure_tag(measure: ObservableMeasure) -> u8 {
    match measure {
        ObservableMeasure::Volume => 0,
        ObservableMeasure::Boundary => 1,
        ObservableMeasure::SphericalVolume => 2,
    }
}
