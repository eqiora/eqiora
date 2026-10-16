//! Exact mathematical Relation meaning in the structural projection.
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
        RelationMeaning::Conditions(_)
        | RelationMeaning::ProbabilityMass { .. }
        | RelationMeaning::ProbabilityDensity { .. }
        | RelationMeaning::Noise { .. }
        | RelationMeaning::Stochastic { .. } => Vec::new(),
        RelationMeaning::Conservation(terms) => terms
            .storage()
            .into_iter()
            .flat_map(|(value, accumulation)| [value, accumulation])
            .chain([terms.flux(), terms.source()])
            .collect(),
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
        RelationMeaning::ProbabilityMass { outcomes, weights } => {
            encoder.u8(4)?;
            push_reference(
                references,
                vec![4],
                lookup(ids, outcomes.erase(), "probability outcomes")?,
                budget,
            )?;
            encoder.u32(canonical_expr_id(*weights, &canonical_index)?)?;
        }
        RelationMeaning::ProbabilityDensity {
            domain,
            measure,
            density,
        } => {
            encoder.u8(5)?;
            encoder.u8(super::observable::measure_tag(*measure))?;
            push_reference(
                references,
                vec![5],
                lookup(ids, domain.erase(), "probability support")?,
                budget,
            )?;
            encoder.u32(canonical_expr_id(*density, &canonical_index)?)?;
        }
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
        RelationMeaning::Conservation(terms) => {
            encoder.u8(1)?;
            encoder.u8(u8::from(terms.storage().is_some()))?;
            for term in extra_roots {
                encoder.u32(canonical_expr_id(term, &canonical_index)?)?;
            }
        }
        RelationMeaning::Noise {
            process,
            parameter,
            independent,
        } => {
            use eqiora_schema::kernel::NoiseProcess;
            encoder.u8(2)?;
            encoder.u8(match process {
                NoiseProcess::Wiener => 0,
                NoiseProcess::Poisson => 1,
            })?;
            encoder.u8(u8::from(*independent))?;
            encoder.u32(canonical_expr_id(*parameter, &canonical_index)?)?;
        }
        RelationMeaning::Stochastic {
            target,
            calculus,
            drift,
            diffusions,
            jumps,
        } => {
            use eqiora_schema::kernel::StochasticCalculus;
            encoder.u8(3)?;
            encoder.u8(match calculus {
                None => 0,
                Some(StochasticCalculus::Ito) => 1,
                Some(StochasticCalculus::Stratonovich) => 2,
            })?;
            push_reference(
                references,
                vec![2],
                lookup(ids, target.erase(), "stochastic State")?,
                budget,
            )?;
            encoder.u8(u8::from(drift.is_some()))?;
            if let Some(drift) = *drift {
                encoder.u32(canonical_expr_id(drift, &canonical_index)?)?;
            }
            for (kind, bindings) in [(0u8, diffusions), (1, jumps)] {
                encoder.len(bindings.len())?;
                for (position, (group, coefficient)) in bindings.iter().enumerate() {
                    let mut label = Encoder::new(16);
                    label.u8(3)?;
                    label.u8(kind)?;
                    label.len(position)?;
                    push_reference(
                        references,
                        label.finish()?,
                        lookup(ids, group.erase(), "noise group")?,
                        budget,
                    )?;
                    encoder.u32(canonical_expr_id(*coefficient, &canonical_index)?)?;
                }
            }
        }
    }
    Ok(())
}
