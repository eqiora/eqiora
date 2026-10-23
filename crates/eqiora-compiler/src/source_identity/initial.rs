//! Canonical initial-equation records and their generated declaration names.

use super::*;

pub(crate) fn initial_declaration_name(
    declaration: &eqiora_lang::InitialDecl,
) -> Result<String, Diagnostic> {
    let limits = LocalSourceIdentityLimits::default();
    let mut budget = Budget::new(limits);
    let mut encoder = Encoder::new(limits.max_canonical_bytes);
    encode_initial(&mut encoder, declaration, &mut budget)?;
    let digest = Sha256::digest(encoder.finish()?);
    Ok(format!("$initial{}", LocalSourceIdentity(digest.into())))
}

pub(super) fn encode_initial(
    encoder: &mut Encoder,
    declaration: &eqiora_lang::InitialDecl,
    budget: &mut Budget,
) -> Result<(), Diagnostic> {
    if declaration.equations().len() > budget.limits.max_residuals_per_relation {
        return Err(source_identity_error(
            "initial equation count exceeds limit",
        ));
    }
    encoder.u32(as_u32(
        declaration.equations().len(),
        "initial equation count",
    )?)?;
    for equation in declaration.equations() {
        encoder.field(1, |encoder| {
            encoder.field(1, |encoder| {
                encode_expression(encoder, equation.left(), budget, 1)
            })?;
            encoder.field(2, |encoder| {
                encode_expression(encoder, equation.right(), budget, 1)
            })
        })?;
    }
    if let Some(domain) = declaration.domain() {
        encoder.field(2, |encoder| encode_name(encoder, domain, budget))?;
    }
    Ok(())
}
