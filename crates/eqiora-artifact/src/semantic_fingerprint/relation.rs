//! Exact exclusive mathematical relation meaning in the shared structural projection.
use super::*;
use eqiora_schema::kernel::RelationDef;

pub(super) fn encode(
    relation: &RelationDef,
    encoder: &mut Encoder,
    ids: &BTreeMap<RawId, usize>,
    references: &mut Vec<Reference>,
    budget: &mut ConstructionBudget,
) -> Result<(), Diagnostic> {
    encoder.u8(6)?;
    encoder.u8(u8::from(relation.is_initial()))?;
    let extra_roots = match relation.meaning() {
        RelationMeaning::Conditions(_) => Vec::new(),
        RelationMeaning::Conservation(terms) => vec![terms.flux(), terms.source()],
    };
    let canonical_index = encode_expression(
        encoder,
        relation.expression(),
        &extra_roots,
        1,
        ids,
        references,
        budget,
    )?;
    match relation.meaning() {
        RelationMeaning::Conditions(conditions) => {
            encoder.u8(0)?;
            encoder.len(conditions.len())?;
            for condition in conditions {
                encoder.u8(match condition {
                    RelationConditionKind::Equality => 0,
                    RelationConditionKind::Inequality => 1,
                    RelationConditionKind::Complementarity => 2,
                })?;
            }
        }
        RelationMeaning::Conservation(_) => {
            encoder.u8(1)?;
            for term in extra_roots {
                encoder.u32(canonical_expr_id(term, &canonical_index)?)?;
            }
        }
    }
    Ok(())
}
