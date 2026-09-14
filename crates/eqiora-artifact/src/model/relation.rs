//! Exact mathematical meaning of Relation operands on the Model wire.
use crate::invalid_artifact;
use eqiora_core::{Diagnostic, Id, entity::kinds};
use eqiora_schema::kernel::{
    ConservationTerms, ExprDag, ExprId, RelationConditionKind, RelationDef, RelationMeaning,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum WireRelationMeaning {
    Conditions {
        conditions: Vec<WireCondition>,
    },
    Conservation {
        storage: Option<WireStorage>,
        flux: u32,
        source: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireCondition {
    Equality,
    Inequality,
    Complementarity,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireStorage {
    value: u32,
    accumulation: u32,
}

impl WireRelationMeaning {
    pub(crate) fn encode(meaning: &RelationMeaning) -> Self {
        match meaning {
            RelationMeaning::Conditions(conditions) => Self::Conditions {
                conditions: conditions
                    .iter()
                    .map(|kind| match kind {
                        RelationConditionKind::Equality => WireCondition::Equality,
                        RelationConditionKind::Inequality => WireCondition::Inequality,
                        RelationConditionKind::Complementarity => WireCondition::Complementarity,
                    })
                    .collect(),
            },
            RelationMeaning::Conservation(terms) => Self::Conservation {
                storage: terms.storage().map(|(value, accumulation)| WireStorage {
                    value: value.index(),
                    accumulation: accumulation.index(),
                }),
                flux: terms.flux().index(),
                source: terms.source().index(),
            },
        }
    }

    pub(crate) fn decode(
        &self,
        id: Id<kinds::Relation>,
        expression: ExprDag,
        initial: bool,
    ) -> Result<RelationDef, Diagnostic> {
        let result = match self {
            Self::Conditions { conditions } => {
                let conditions: Vec<_> = conditions
                    .iter()
                    .map(|kind| match kind {
                        WireCondition::Equality => RelationConditionKind::Equality,
                        WireCondition::Inequality => RelationConditionKind::Inequality,
                        WireCondition::Complementarity => RelationConditionKind::Complementarity,
                    })
                    .collect();
                if initial {
                    if conditions.len() != expression.roots().len() / 2
                        || conditions
                            .iter()
                            .any(|kind| *kind != RelationConditionKind::Equality)
                    {
                        return Err(invalid_artifact(
                            "initial Relation requires only paired equality conditions",
                        ));
                    }
                    RelationDef::initial(id, expression)
                } else {
                    RelationDef::with_conditions(id, expression, conditions)
                }
            }
            Self::Conservation {
                storage,
                flux,
                source,
            } => {
                if initial {
                    return Err(invalid_artifact(
                        "conservation Law cannot be an initialization-only Relation",
                    ));
                }
                let lookup = |index: u32| -> Result<ExprId, Diagnostic> {
                    expression.node_id(index).ok_or_else(|| {
                        invalid_artifact("Law term index is outside its owning Relation DAG")
                    })
                };
                let storage = storage
                    .map(|stored| {
                        Ok::<_, Diagnostic>((lookup(stored.value)?, lookup(stored.accumulation)?))
                    })
                    .transpose()?;
                let terms = ConservationTerms::new(storage, lookup(*flux)?, lookup(*source)?);
                RelationDef::conservation(id, expression, terms)
            }
        };
        result.map_err(|error| invalid_artifact(error.message()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, DynQuantity};
    use eqiora_schema::kernel::{ExprDagBuilder, ExprNode};

    #[test]
    fn exact_condition_kind_and_operand_order_survive_wire() {
        let mut builder = ExprDagBuilder::new();
        let gap = builder
            .constant(DynQuantity::new(
                3.0,
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
            ))
            .unwrap();
        let force = builder
            .constant(DynQuantity::new(
                0.0,
                DimExponents::from_integers([1, 1, -2, 0, 0, 0, 0]).unwrap(),
            ))
            .unwrap();
        let expression = builder.finish([gap, force]).unwrap();
        for condition in [
            RelationConditionKind::Equality,
            RelationConditionKind::Inequality,
            RelationConditionKind::Complementarity,
        ] {
            let relation =
                RelationDef::with_conditions(Id::new(), expression.clone(), vec![condition])
                    .unwrap();
            let wire = WireRelationMeaning::encode(relation.meaning());
            assert_eq!(
                wire.decode(relation.id(), expression.clone(), false)
                    .unwrap(),
                relation
            );
            if condition != RelationConditionKind::Equality {
                assert!(
                    wire.decode(relation.id(), expression.clone(), true)
                        .is_err()
                );
            }
        }
        assert!(
            WireRelationMeaning::Conditions { conditions: vec![] }
                .decode(Id::new(), expression, false)
                .is_err()
        );
    }

    #[test]
    fn conservation_wire_binds_exact_term_nodes_and_rejects_foreign_indices() {
        let mut builder = ExprDagBuilder::new();
        let flux = builder
            .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        let divergence = builder.push(ExprNode::Divergence(flux)).unwrap();
        let source = builder
            .constant(DynQuantity::new(0.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        let expression = builder.finish([divergence, source]).unwrap();
        let relation = RelationDef::conservation(
            Id::new(),
            expression.clone(),
            ConservationTerms::new(None, flux, source),
        )
        .unwrap();
        let wire = WireRelationMeaning::encode(relation.meaning());
        assert_eq!(
            wire.decode(relation.id(), expression.clone(), false)
                .unwrap(),
            relation
        );
        let bad = WireRelationMeaning::Conservation {
            storage: None,
            flux: u32::MAX,
            source: source.index(),
        };
        assert!(bad.decode(relation.id(), expression, false).is_err());
    }
}
