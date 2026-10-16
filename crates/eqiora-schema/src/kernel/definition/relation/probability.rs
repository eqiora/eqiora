//! Probability laws share the retained Relation expression owner, never a sampled State.

use super::*;
use crate::kernel::ObservableMeasure;

impl RelationDef {
    /// Whether this Relation defines an immutable probability measure rather than an
    /// activated equation or evolution law.
    #[must_use]
    pub const fn is_probability_law(&self) -> bool {
        matches!(
            self.meaning,
            RelationMeaning::ProbabilityMass { .. } | RelationMeaning::ProbabilityDensity { .. }
        )
    }

    /// Define normalized mass on an exact ordered declaration of named outcomes.
    ///
    /// The retained weights are proportional masses: normalization is part of this
    /// law's meaning, not a clipping or repair step. Whole-Model admission requires
    /// one dimensionless nonnegative real weight per outcome and positive total mass.
    pub fn probability_mass(
        id: Id<kinds::Relation>,
        expression: ExprDag,
        outcomes: Id<kinds::Enum>,
        weights: ExprId,
    ) -> Result<Self, Diagnostic> {
        probability_root(&expression, weights)?;
        Ok(Self {
            id,
            expression,
            meaning: RelationMeaning::ProbabilityMass { outcomes, weights },
            initial: false,
        })
    }

    /// Define normalized probability relative to one explicit continuous measure.
    ///
    /// The retained density has inverse-measure units. Its finite positive
    /// normalizing integral belongs to the law. Admission must prove this integral
    /// and nonnegativity; a quadrature sample alone is not that proof.
    pub fn probability_density(
        id: Id<kinds::Relation>,
        expression: ExprDag,
        domain: Id<kinds::Domain>,
        measure: ObservableMeasure,
        density: ExprId,
    ) -> Result<Self, Diagnostic> {
        probability_root(&expression, density)?;
        Ok(Self {
            id,
            expression,
            meaning: RelationMeaning::ProbabilityDensity {
                domain,
                measure,
                density,
            },
            initial: false,
        })
    }
}

fn probability_root(expression: &ExprDag, root: ExprId) -> Result<(), Diagnostic> {
    if expression.roots() != [root] || expression.node(root).is_none() {
        return Err(Diagnostic::error(
            codes::INVALID_KERNEL_DEFINITION,
            "probability law requires exactly its declared mass or density root",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{ExprDagBuilder, SymbolRef};

    #[test]
    fn probability_weights_are_not_equations_or_sampled_values() {
        let mut builder = ExprDagBuilder::new();
        let root = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
        let foreign = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
        let expression = builder.finish([root]).unwrap();
        for law in [
            RelationDef::probability_mass(Id::new(), expression.clone(), Id::new(), root),
            RelationDef::probability_density(
                Id::new(),
                expression.clone(),
                Id::new(),
                ObservableMeasure::Volume,
                root,
            ),
        ] {
            let law = law.unwrap();
            assert!(law.conditions().is_none());
            assert!(law.equation_sides().is_err());
            assert!(!law.is_initial());
        }
        assert!(RelationDef::probability_mass(Id::new(), expression, Id::new(), foreign).is_err());
    }
}
