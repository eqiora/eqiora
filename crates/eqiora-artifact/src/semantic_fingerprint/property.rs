//! Property calculus is already structural; provenance belongs to exact artifacts.
use super::*;
use eqiora_schema::kernel::PropertyMeaning;

pub(super) fn encode(
    encoder: &mut Encoder,
    expression: &ExprDag,
    canonical_index: &[u32],
) -> Result<(), Diagnostic> {
    let mut derivative_products = Vec::new();
    for (root, release) in expression.properties() {
        // Exhaustive handling keeps future meaning variants from disappearing.
        // Constant payloads and analytic definitions already bind exact types,
        // ordered operands and validity guards at their expression occurrence.
        // Formal names and branch labels are authored identifiers. Contract and
        // release paths, attribution and grouping are exact-artifact provenance.
        match release.meaning() {
            PropertyMeaning::Constant(_) => {}
            PropertyMeaning::Analytic(_) => {
                if release.first_partials() {
                    derivative_products.push((
                        canonical_expr_id(*root, canonical_index)?,
                        release.derivatives(),
                    ));
                }
            }
            PropertyMeaning::Table(table) => {
                // The complete table, including the open derivative policy, is
                // generated at the exact definition digest. The profile remains
                // explicit here so a future alternative cannot erase semantics.
                match table.profile() { eqiora_schema::property_table::RealTableProfile::PiecewiseAffineOpenIntervalsV1 => {} }
                if release.first_partials() {
                    derivative_products.push((
                        canonical_expr_id(*root, canonical_index)?,
                        release.derivatives(),
                    ));
                }
            }
        }
    }
    derivative_products.sort_unstable_by_key(|(root, _)| *root);
    encoder.len(derivative_products.len())?;
    for (root, profile) in derivative_products {
        encoder.u32(root)?;
        encoder.u8(match profile {
            eqiora_schema::kernel::PropertyDerivatives::ValueOnly => 0,
            eqiora_schema::kernel::PropertyDerivatives::FirstPartials => 1,
            eqiora_schema::kernel::PropertyDerivatives::FirstOpenIntervals => 2,
        })?;
    }
    Ok(())
}
