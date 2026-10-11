//! Claim-local encoding of nominal product declarations and ordered typed members.
use super::*;

pub(super) fn encode_record(
    declaration: &eqiora_lang::RecordDecl,
    budget: &mut Budget,
) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    encode_visibility(&mut encoder, declaration.visibility())?;
    encode_name(&mut encoder, declaration.name(), budget)?;
    encoder.u32(as_u32(declaration.members().len(), "record members")?)?;
    for member in declaration.members() {
        encode_name(&mut encoder, member.name(), budget)?;
        value_type::encode_value_type(&mut encoder, member.value_type(), budget, 1)?;
    }
    encoder.finish()
}
