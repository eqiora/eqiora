//! Checked construction of ordered equalities and Relation declarations.

use super::*;

impl SourceAstFactory {
    /// Construct simultaneous mathematical initialization conditions.
    ///
    /// # Errors
    /// Rejects empty conditions, malformed expressions, and invalid byte ranges.
    pub fn initial(
        equations: Vec<RelationCondition>,
        range: TextRange,
    ) -> Result<crate::ast::InitialDecl, AstConstructionError> {
        if equations.is_empty() {
            return Err(AstConstructionError::new(
                "initial requires at least one equation",
            ));
        }
        for equation in &equations {
            if equation.kind() != eqiora_schema::kernel::RelationConditionKind::Equality {
                return Err(AstConstructionError::new(
                    "initial blocks admit only equality conditions",
                ));
            }
            validate_expression(equation.left())?;
            validate_expression(equation.right())?;
            checked_range(equation.range())?;
        }
        Ok(crate::ast::InitialDecl {
            comments: Default::default(),
            equations,
            range: checked_range(range)?,
        })
    }

    /// Construct an ordered equality without interpreting either side.
    ///
    /// # Errors
    /// Returns an error for a malformed expression or range.
    pub fn equation(
        left: Expr,
        right: Expr,
        range: TextRange,
    ) -> Result<RelationCondition, AstConstructionError> {
        validate_expression(&left)?;
        validate_expression(&right)?;
        Ok(RelationCondition {
            kind: eqiora_schema::kernel::RelationConditionKind::Equality,
            left,
            right,
            range: checked_range(range)?,
        })
    }

    /// Construct an implicit Relation with at least one equality.
    ///
    /// # Errors
    /// Returns an error for an empty equation set or malformed source shape.
    pub fn relation(
        name: impl Into<String>,
        activation: ActivationSyntax,
        domain: Option<String>,
        equations: Vec<RelationCondition>,
        range: TextRange,
    ) -> Result<RelationDecl, AstConstructionError> {
        if equations.is_empty() {
            return Err(AstConstructionError::new(
                "a Relation requires at least one equation",
            ));
        }
        if let ActivationSyntax::Named(clock) = &activation {
            validate_identifier(clock, "periodic Clock")?;
        }
        if let Some(domain) = &domain {
            validate_identifier(domain, "Relation Domain")?;
        }
        for equation in &equations {
            validate_expression(equation.left())?;
            validate_expression(equation.right())?;
            checked_range(equation.range())?;
        }
        Ok(RelationDecl {
            comments: Default::default(),
            name: checked_identifier(name, "Relation")?,
            activation,
            domain,
            body: crate::ast::RelationBody::Conditions(equations),
            range: checked_range(range)?,
        })
    }

    /// Construct one Relation family over an exact finite set.
    ///
    /// # Errors
    /// Rejects malformed binders or equations. The compiler determines the
    /// set kind and checks support and activation restrictions.
    pub fn relation_family(
        relation: RelationDecl,
        binder: FamilyBinderSyntax,
    ) -> Result<RelationFamilyDecl, AstConstructionError> {
        validate_boundary_family_binder(&binder)?;
        checked_range(relation.range())?;
        for equation in relation.conditions().ok_or_else(|| {
            AstConstructionError::new("Law families require explicit supported elaboration")
        })? {
            validate_expression(equation.left())?;
            validate_expression(equation.right())?;
            checked_range(equation.range())?;
        }
        Ok(RelationFamilyDecl { relation, binder })
    }
}
