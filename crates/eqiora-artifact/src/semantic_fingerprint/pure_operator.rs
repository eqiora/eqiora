//! Pure contracts use the same nominal-reference projection as other value types.
use super::*;
use eqiora_schema::kernel::pure_operator::{PureOperatorDefinition, PureValueClass};

pub(super) fn encode(
    encoder: &mut Encoder,
    definition: &PureOperatorDefinition,
    scope: u8,
    slot: usize,
    ids: &BTreeMap<RawId, usize>,
    references: &mut Vec<Reference>,
    budget: &mut ConstructionBudget,
) -> Result<(), Diagnostic> {
    encoder.len(definition.formals().len())?;
    for (role, class) in definition
        .formals()
        .iter()
        .chain([definition.result_rule()])
        .enumerate()
    {
        if let Some(value) = class.exact_type() {
            encoder.u8(1)?;
            encode_value_type(encoder, value)?;
            let mut label = Encoder::new(32);
            label.u8(3)?;
            label.u8(scope)?;
            label.u32(
                u32::try_from(slot)
                    .map_err(|_| fingerprint_error("pure definition slot exceeds u32"))?,
            )?;
            label.u8(17)?;
            label.u16(
                u16::try_from(role)
                    .map_err(|_| fingerprint_error("pure formal slot exceeds u16"))?,
            )?;
            type_reference(value, label.finish()?, ids, references, budget)?;
        } else {
            encoder.u8(0)?;
            encode_template(encoder, class)?;
        }
    }
    let body = definition.canonical_body_bytes();
    budget.account_bytes(body.len())?;
    encoder.bytes(&body)
}

fn encode_template(encoder: &mut Encoder, class: &PureValueClass) -> Result<(), Diagnostic> {
    encoder.u16(class.spatial_rank().unwrap_or(0))?;
    encoder.u32(class.spatial_extent().unwrap_or(0))?;
    encoder.u8(match class.scalar_domain() {
        None => 0,
        Some(eqiora_core::ScalarDomain::Real) => 1,
        Some(eqiora_core::ScalarDomain::Complex) => 2,
        Some(_) => {
            return Err(fingerprint_error(
                "pure contract has a discrete scalar domain",
            ));
        }
    })?;
    encoder.u8(u8::from(class.dimension().is_some()))?;
    if let Some(dimension) = class.dimension() {
        values::encode_dimension(encoder, dimension)?;
    }
    Ok(())
}
