//! Canonical records for top-level operator, connector and component declarations.

use super::*;

pub(super) fn encode_pure_operators(
    document: &Document,
    budget: &mut Budget,
) -> Result<Vec<Vec<u8>>, Diagnostic> {
    let definitions = crate::pure_operator::compile_definitions("<source-identity>", document)?;
    encode_sorted_records(document.pure_operators(), budget, |declaration, budget| {
        encode_pure_operator(declaration, &definitions[declaration.name()], budget)
    })
}

fn encode_pure_operator(
    declaration: &PureOperatorDecl,
    definition: &eqiora_schema::kernel::pure_operator::PureOperatorDefinition,
    budget: &mut Budget,
) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| encoder.raw(&definition.digest().bytes()))?;
    if declaration.visibility() == VisibilitySyntax::Public {
        encoder.field(3, |encoder| {
            encode_visibility(encoder, declaration.visibility())
        })?;
    }
    encoder.finish()
}

pub(super) fn encode_connector(
    declaration: &ConnectorDecl,
    budget: &mut Budget,
) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| match declaration.syntax() {
        ConnectorSyntax::ScalarPhysical {
            across_name,
            across_type,
            through_name,
            through_type,
        } => {
            encoder.u16(1)?;
            encoder.field(1, |encoder| {
                value_type::encode_value_type(encoder, across_type, budget, 1)
            })?;
            encoder.field(2, |encoder| {
                value_type::encode_value_type(encoder, through_type, budget, 1)
            })?;
            encoder.field(3, |encoder| encode_name(encoder, across_name, budget))?;
            encoder.field(4, |encoder| encode_name(encoder, through_name, budget))
        }
        ConnectorSyntax::FieldPhysical {
            trace,
            flux,
            shape,
            frame,
            pairing,
        } => {
            encoder.u16(2)?;
            encoder.field(1, |encoder| {
                encode_connector_quantity(encoder, trace, budget)
            })?;
            encoder.field(2, |encoder| {
                encode_connector_quantity(encoder, flux, budget)
            })?;
            encoder.field(3, |encoder| encode_value_shape(encoder, shape))?;
            encoder.field(4, |encoder| encode_frame(encoder, *frame))?;
            encoder.field(5, |encoder| encode_boundary_pairing(encoder, *pairing))
        }
        _ => Err(source_identity_error(
            "Connector syntax is newer than source identity v1",
        )),
    })?;
    if declaration.visibility() == VisibilitySyntax::Public {
        encoder.field(3, |encoder| {
            encode_visibility(encoder, declaration.visibility())
        })?;
    }
    encoder.finish()
}

pub(super) fn encode_component(
    declaration: &ComponentDecl,
    budget: &mut Budget,
) -> Result<Vec<u8>, Diagnostic> {
    let member_count = declaration
        .items()
        .len()
        .checked_add(declaration.signature().len())
        .ok_or_else(|| source_identity_error("component member count overflows usize"))?;
    budget.account_members(member_count, "component")?;
    let members = encode_container_records(
        declaration.items(),
        budget,
        component_connection,
        encode_component_item,
        COMPONENT_CONNECTION_ITEM_TAG,
    )?;
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| encoder.records(&members))?;
    if declaration.visibility() == VisibilitySyntax::Public {
        encoder.field(3, |encoder| {
            encode_visibility(encoder, declaration.visibility())
        })?;
    }
    let signature = signature::encode_signature(declaration.signature(), budget)?;
    encoder.field(4, |encoder| encoder.records(&signature))?;
    encoder.finish()
}

pub(super) fn encode_event(
    encoder: &mut Encoder,
    declaration: &eqiora_lang::EventDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| {
        encode_expression(encoder, declaration.guard(), budget, 0)
    })?;
    encoder.field(3, |encoder| {
        encoder.u16(match declaration.direction() {
            eqiora_schema::kernel::EventDirection::Any => 0,
            eqiora_schema::kernel::EventDirection::Rising => 1,
            eqiora_schema::kernel::EventDirection::Falling => 2,
        })
    })?;
    encoder.field(4, |encoder| {
        encoder.raw(&declaration.priority().to_be_bytes())
    })
}

pub(super) fn encode_clock(
    encoder: &mut Encoder,
    declaration: &ClockDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    encoder.field(1, |encoder| {
        encode_name(encoder, declaration.name(), budget)
    })?;
    encoder.field(2, |encoder| {
        encode_expression(encoder, declaration.period(), budget, 0)
    })?;
    encoder.field(3, |encoder| {
        encode_expression(encoder, declaration.phase(), budget, 0)
    })
}

pub(super) fn encode_enumeration(
    declaration: &eqiora_lang::EnumDecl,
    budget: &mut Budget,
) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    encode_visibility(&mut encoder, declaration.visibility())?;
    encode_name(&mut encoder, declaration.name(), budget)?;
    encoder.u32(as_u32(declaration.tags().len(), "enum members")?)?;
    for tag in declaration.tags() {
        encode_name(&mut encoder, tag.as_str(), budget)?;
    }
    encoder.finish()
}
