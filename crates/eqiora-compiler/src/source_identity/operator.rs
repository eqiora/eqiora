//! Lexical operator identity before nominal binding. Kernel definitions own
//! mathematical normalization after binding; source identity seeds that binding.
use super::*;
use eqiora_lang::PureValueClassSyntax;

pub(super) fn encode_pure_operators(
    document: &Document,
    budget: &mut Budget,
) -> Result<Vec<Vec<u8>>, Diagnostic> {
    encode_sorted_records(document.pure_operators(), budget, |declaration, budget| {
        budget.account_members(declaration.formals().len(), "operator")?;
        let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
        encode_name(&mut encoder, declaration.name(), budget)?;
        encode_visibility(&mut encoder, declaration.visibility())?;
        encoder.u32(as_u32(declaration.formals().len(), "operator formals")?)?;
        for formal in declaration.formals() {
            encode_class(&mut encoder, formal.value_class(), budget)?;
        }
        encode_class(&mut encoder, declaration.result(), budget)?;
        for (slot, formal) in declaration.formals().iter().enumerate() {
            budget.account_name(formal.name())?;
            if budget
                .lexical_formals
                .insert(formal.name().to_owned(), slot)
                .is_some()
            {
                return Err(source_identity_error("duplicate operator formal"));
            }
        }
        let body = encode_expression(&mut encoder, declaration.body(), budget, 0);
        budget.lexical_formals.clear();
        body?;
        encoder.finish()
    })
}

fn encode_class(
    encoder: &mut Encoder,
    class: &PureValueClassSyntax,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    match class {
        PureValueClassSyntax::Scalar => encoder.u8(0),
        PureValueClassSyntax::Spatial { rank } => {
            encoder.u8(1)?;
            encoder.raw(&rank.value().to_be_bytes())
        }
        PureValueClassSyntax::Typed(syntax) => {
            encoder.u8(2)?;
            value_type::encode_value_type(encoder, syntax, budget, 1)
        }
        _ => Err(source_identity_error("unsupported operator value class")),
    }
}
