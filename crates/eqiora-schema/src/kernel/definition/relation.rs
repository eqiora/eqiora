//! Typed mathematical relations, separate from numerical enforcement.

use crate::kernel::{ConservationTerms, ExprDag, ExprId, NoiseProcess, StochasticCalculus};
use eqiora_core::{Diagnostic, Id, diagnostic::codes, entity::kinds};

/// Mathematical meaning of one ordered pair of expression roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationConditionKind {
    /// Both operands are equal and have the same type.
    Equality,
    /// The real left operand is greater than or equal to the real right operand.
    Inequality,
    /// Both real operands are nonnegative and their product is zero.
    /// Operands may have different dimensions; they must share exact support.
    Complementarity,
}

impl RelationConditionKind {
    /// Check the original operand types without constructing a Boolean or numerical residual.
    ///
    /// # Errors
    /// Rejects unordered values, incompatible inequality units or distinct complementarity supports.
    pub fn check_operands<I: Clone + Eq>(
        self,
        left: &crate::kernel::typing::ExpressionType<I>,
        right: &crate::kernel::typing::ExpressionType<I>,
    ) -> Result<crate::kernel::typing::ExpressionType<I>, crate::kernel::typing::TypeViolation<I>>
    {
        use crate::kernel::typing::TypeViolation;
        use eqiora_core::{ScalarDomain, ValueFrame};
        if self == Self::Equality {
            return left.clone().equation(right.clone());
        }
        for value in [left, right] {
            if value.value_type.scalar_domain() != ScalarDomain::Real
                || !value.shape().is_scalar()
                || value.frame() != ValueFrame::Invariant
            {
                return Err(TypeViolation::RootRequiresRealScalar);
            }
        }
        if self == Self::Inequality {
            return left.clone().equation(right.clone());
        }
        if left.support != right.support {
            return Err(TypeViolation::ResidualSupportMismatch {
                residual: Box::new(left.support.clone()),
                relation: Box::new(right.support.clone()),
            });
        }
        Ok(left.clone())
    }
}

/// Mathematical meaning of a Relation's expression roots.
#[derive(Debug, Clone, PartialEq)]
pub enum RelationMeaning {
    /// Ordered equality, inequality, and complementarity conditions.
    Conditions(Vec<RelationConditionKind>),
    /// A physical conservation balance with its authored terms.
    Conservation(ConservationTerms),
    /// Nominal increment distribution, with no ambient generator or sampled value.
    Noise {
        /// Mathematical increment law.
        process: NoiseProcess,
        /// Covariance or intensity expression root.
        parameter: ExprId,
        /// Explicit independence from other groups.
        independent: bool,
    },
    /// Owned stochastic evolution with pure drift and increment coefficients.
    Stochastic {
        /// Exact owned State.
        target: Id<kinds::Field>,
        /// Written continuous stochastic calculus.
        calculus: Option<StochasticCalculus>,
        /// Pure drift expression, absent for pure jumps.
        drift: Option<ExprId>,
        /// Ordered Wiener group and coefficient bindings.
        diffusions: Vec<(Id<kinds::Relation>, ExprId)>,
        /// Ordered counting group and State-increment bindings.
        jumps: Vec<(Id<kinds::Relation>, ExprId)>,
    },
}

/// Ordered mathematical conditions together with their authored operands.
/// How the conditions are enforced numerically is not part of the Model definition.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationDef {
    id: Id<kinds::Relation>,
    expression: ExprDag,
    meaning: RelationMeaning,
    initial: bool,
}

impl RelationDef {
    /// Define equalities from consecutive left/right expression root pairs.
    ///
    /// # Errors
    /// Rejects an odd number of roots.
    pub fn new(id: Id<kinds::Relation>, expression: ExprDag) -> Result<Self, Diagnostic> {
        let conditions = vec![RelationConditionKind::Equality; expression.roots().len() / 2];
        Self::with_conditions(id, expression, conditions)
    }

    /// Define typed mathematical conditions with one descriptor per root pair.
    /// Complementarity implies that both operands are nonnegative.
    /// Units, real order, and support are checked by semantic admission.
    ///
    /// # Errors
    /// Rejects unpaired roots or a mismatched descriptor count.
    pub fn with_conditions(
        id: Id<kinds::Relation>,
        expression: ExprDag,
        conditions: Vec<RelationConditionKind>,
    ) -> Result<Self, Diagnostic> {
        if !expression.roots().len().is_multiple_of(2)
            || conditions.len() != expression.roots().len() / 2
        {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "Relation conditions require one descriptor per consecutive left/right root pair",
            ));
        }
        Ok(Self {
            id,
            expression,
            meaning: RelationMeaning::Conditions(conditions),
            initial: false,
        })
    }

    /// Define simultaneous equalities used only for fresh initialization.
    ///
    /// # Errors
    /// Rejects an odd number of roots.
    pub fn initial(id: Id<kinds::Relation>, expression: ExprDag) -> Result<Self, Diagnostic> {
        let mut result = Self::new(id, expression)?;
        result.initial = true;
        Ok(result)
    }

    /// Whether these conditions apply only to fresh initialization.
    #[must_use]
    pub const fn is_initial(&self) -> bool {
        self.initial
    }

    /// Relation identity.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Relation> {
        self.id
    }

    /// Expression arena holding all condition operands.
    #[must_use]
    pub const fn expression(&self) -> &ExprDag {
        &self.expression
    }

    /// Mathematical condition descriptors in authored order, when this is a condition Relation.
    #[must_use]
    pub fn conditions(&self) -> Option<&[RelationConditionKind]> {
        match &self.meaning {
            RelationMeaning::Conditions(conditions) => Some(conditions),
            RelationMeaning::Conservation(_)
            | RelationMeaning::Noise { .. }
            | RelationMeaning::Stochastic { .. } => None,
        }
    }

    /// Mathematical meaning, including the terms of a physical Law.
    #[must_use]
    pub const fn meaning(&self) -> &RelationMeaning {
        &self.meaning
    }

    /// Define a physical conservation Law from its balance operands.
    ///
    /// # Errors
    /// Rejects a term descriptor that does not name exactly the balance roots.
    pub fn conservation(
        id: Id<kinds::Relation>,
        expression: ExprDag,
        terms: ConservationTerms,
    ) -> Result<Self, Diagnostic> {
        terms.validate_balance(&expression)?;
        Ok(Self {
            id,
            expression,
            meaning: RelationMeaning::Conservation(terms),
            initial: false,
        })
    }

    /// Define a nominal process Law from its exact parameter root.
    pub fn noise(
        id: Id<kinds::Relation>,
        expression: ExprDag,
        process: NoiseProcess,
        parameter: ExprId,
        independent: bool,
    ) -> Result<Self, Diagnostic> {
        let meaning = RelationMeaning::Noise {
            process,
            parameter,
            independent,
        };
        meaning.validate_process_roots(&expression)?;
        Ok(Self {
            id,
            expression,
            meaning,
            initial: false,
        })
    }

    /// Define stochastic evolution; coefficients are values, never equation sides.
    pub fn stochastic(
        id: Id<kinds::Relation>,
        expression: ExprDag,
        target: Id<kinds::Field>,
        calculus: Option<StochasticCalculus>,
        drift: Option<ExprId>,
        diffusions: Vec<(Id<kinds::Relation>, ExprId)>,
        jumps: Vec<(Id<kinds::Relation>, ExprId)>,
    ) -> Result<Self, Diagnostic> {
        let meaning = RelationMeaning::Stochastic {
            target,
            calculus,
            drift,
            diffusions,
            jumps,
        };
        meaning.validate_process_roots(&expression)?;
        Ok(Self {
            id,
            expression,
            meaning,
            initial: false,
        })
    }

    /// Whether a numerical realization must explicitly enforce constraints.
    #[must_use]
    pub fn has_constraints(&self) -> bool {
        matches!(&self.meaning, RelationMeaning::Conditions(conditions) if conditions
            .iter()
            .any(|kind| *kind != RelationConditionKind::Equality))
    }

    /// Operand pairs in order. Check `conditions()` before forming residuals:
    /// an inequality or complementarity pair is not an equality to subtract.
    /// # Errors
    /// Process and stochastic evolution Laws have no equality residual projection.
    pub fn equation_sides(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = (ExprId, ExprId)> + '_, Diagnostic> {
        if matches!(
            self.meaning,
            RelationMeaning::Noise { .. } | RelationMeaning::Stochastic { .. }
        ) {
            return Err(Diagnostic::error(
                codes::NOT_IMPLEMENTED,
                "stochastic Laws require explicit stochastic execution; their coefficients are not equation sides",
            ));
        }
        Ok(self
            .expression
            .roots()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0], pair[1])))
    }
}

impl RelationDef {
    /// Check the conditions against independently inferred operand types.
    /// The caller must infer the complete DAG first.
    ///
    /// # Errors
    /// Rejects incompatible equality/inequality types, non-real ordered values,
    /// unsupported frames, and distinct complementarity supports.
    pub fn validate_conditions<I: Clone + Eq>(
        &self,
        typed: &crate::kernel::typing::TypedResidual<I>,
        support: Option<&crate::kernel::typing::SpatialSupport<I>>,
    ) -> Result<(), crate::kernel::typing::TypeViolation<I>> {
        use crate::kernel::typing::{TypeViolation, residual};
        if typed.expression() != self.expression() {
            return Err(TypeViolation::ScalarDomainMismatch);
        }
        let equality = [RelationConditionKind::Equality];
        let conditions = match &self.meaning {
            RelationMeaning::Conditions(conditions) => conditions.as_slice(),
            RelationMeaning::Conservation(_) => &equality,
            RelationMeaning::Noise { .. } | RelationMeaning::Stochastic { .. } => {
                return Err(TypeViolation::NotEquationConditions);
            }
        };
        for (kind, (left, right)) in conditions.iter().zip(
            self.equation_sides()
                .map_err(|_| TypeViolation::NotEquationConditions)?,
        ) {
            let left = typed
                .node_type(left)
                .ok_or(TypeViolation::ScalarDomainMismatch)?;
            let right = typed
                .node_type(right)
                .ok_or(TypeViolation::ScalarDomainMismatch)?;
            let value = kind.check_operands(left, right)?;
            if !self.initial {
                residual(&value, support)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
