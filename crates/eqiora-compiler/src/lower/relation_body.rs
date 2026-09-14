//! Exclusive lowered mathematical Relation bodies.

use super::{LoweringEquation, LoweringExpression};

#[derive(Debug, Clone)]
pub(crate) enum LoweringRelationBody {
    Equations(Vec<LoweringEquation>),
    Conservation {
        storage: Option<LoweringExpression>,
        flux: LoweringExpression,
        source: LoweringExpression,
    },
}

impl From<Vec<LoweringEquation>> for LoweringRelationBody {
    fn from(value: Vec<LoweringEquation>) -> Self {
        Self::Equations(value)
    }
}

impl LoweringRelationBody {
    pub(crate) fn expressions(&self) -> impl Iterator<Item = &LoweringExpression> {
        let (conditions, storage, flux, source) = match self {
            Self::Equations(values) => (values.as_slice(), None, None, None),
            Self::Conservation {
                storage,
                flux,
                source,
            } => (&[][..], storage.as_ref(), Some(flux), Some(source)),
        };
        conditions
            .iter()
            .flat_map(|condition| [&condition.left, &condition.right])
            .chain(storage)
            .chain(flux)
            .chain(source)
    }
}

impl LoweringRelationBody {
    pub(super) fn lower(
        &self,
        file: &str,
        range: super::TextRange,
        activation: &super::ActivationSyntax,
        domain: Option<&str>,
        initial: bool,
        bindings: &std::collections::BTreeMap<String, super::Binding>,
    ) -> Result<
        (
            super::expression::LoweredRelation,
            Option<eqiora_schema::kernel::ConservationTerms>,
        ),
        eqiora_core::Diagnostic,
    > {
        use super::expression::lower_relation;
        use super::{codes, expression, source_error};
        match self {
            LoweringRelationBody::Equations(equations) => lower_relation(
                file, range, activation, domain, equations, initial, bindings,
            )
            .map(|lowered| (lowered, None)),
            LoweringRelationBody::Conservation {
                storage,
                flux,
                source,
            } => {
                let domain = domain.ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        range,
                        "Law requires volume support",
                    )
                });
                domain
                    .and_then(|domain| {
                        expression::lower_law(
                            file,
                            range,
                            domain,
                            storage.as_ref(),
                            flux,
                            source,
                            bindings,
                        )
                    })
                    .map(|(lowered, terms)| (lowered, Some(terms)))
            }
        }
    }
}
