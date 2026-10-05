//! Bounded structural validation for all shared expression syntax.

use super::{
    AstConstructionError, Expr, ExprKind, checked_range, validate_boundary_port_selector,
    validate_identifier, validate_name_path,
};

pub(crate) fn validate_expression(expression: &Expr) -> Result<(), AstConstructionError> {
    validate_expression_depth(expression, 1)
}

fn validate_expression_depth(expression: &Expr, depth: usize) -> Result<(), AstConstructionError> {
    if depth > super::SourceAstFactory::MAX_EXPRESSION_DEPTH {
        return Err(AstConstructionError::new(
            "expression tree exceeds the 256-level limit",
        ));
    }
    checked_range(expression.range())?;
    match expression.kind() {
        ExprKind::Evaluate { at, .. }
        | ExprKind::Pullback { at, .. }
        | ExprKind::CoordinateMapFactor { at, .. } => {
            if let ExprKind::Pullback { source, .. }
            | ExprKind::CoordinateMapFactor { source, .. } = expression.kind()
            {
                if source.is_empty() || source.len() > super::SourceAstFactory::MAX_EXPRESSION_NODES
                {
                    return Err(AstConstructionError::new(
                        "pullback source inventory exceeds expression bounds",
                    ));
                }
                let mut names = std::collections::BTreeSet::new();
                for name in source {
                    validate_name_path(name)?;
                    if !names.insert(name.as_str()) {
                        return Err(AstConstructionError::new(
                            "pullback repeats a source coordinate",
                        ));
                    }
                }
            }
            if at.is_empty() || at.len() > super::SourceAstFactory::MAX_EXPRESSION_NODES {
                return Err(AstConstructionError::new(
                    "evaluation coordinate count exceeds expression bounds",
                ));
            }
            if let ExprKind::Evaluate { value, .. } | ExprKind::Pullback { value, .. } =
                expression.kind()
            {
                validate_expression_depth(value, depth + 1)?;
            }
            let mut names = std::collections::BTreeSet::new();
            for (coordinate, point) in at {
                validate_name_path(coordinate)?;
                if !names.insert(coordinate.as_str()) {
                    return Err(AstConstructionError::new(
                        "point evaluation repeats a coordinate binding",
                    ));
                }
                validate_expression_depth(point, depth + 1)?;
            }
            Ok(())
        }
        ExprKind::Partial {
            value,
            wrt,
            holding,
        } => {
            validate_name_path(wrt)?;
            if holding.len() > super::SourceAstFactory::MAX_EXPRESSION_NODES {
                return Err(AstConstructionError::new(
                    "partial holding set exceeds expression resource bound",
                ));
            }
            for name in holding {
                validate_name_path(name)?;
            }
            validate_expression_depth(value, depth + 1)
        }
        ExprKind::Number(_) | ExprKind::Boolean(_) => Ok(()),
        ExprKind::Member { value, member } => {
            if !matches!(
                value.kind(),
                ExprKind::Index { .. }
                    | ExprKind::Member { .. }
                    | ExprKind::BoundaryPortSelection { .. }
            ) {
                return Err(AstConstructionError::new(
                    "member access requires an indexed occurrence or selected boundary port",
                ));
            }
            validate_identifier(member, "indexed occurrence member")?;
            validate_expression_depth(value, depth + 1)
        }
        ExprKind::Quantity { unit, .. } => validate_expression_depth(unit, depth + 1),
        ExprKind::Name(name) => validate_identifier(name, "expression name"),
        ExprKind::Path(path) => validate_name_path(path),
        ExprKind::BoundaryPortSelection { port, selector } => {
            validate_name_path(port)?;
            validate_boundary_port_selector(selector)
        }
        ExprKind::Unary { value, .. } => validate_expression_depth(value, depth + 1),
        ExprKind::Binary { left, right, .. } => {
            validate_expression_depth(left, depth + 1)?;
            validate_expression_depth(right, depth + 1)
        }
        ExprKind::Index { value, index } => {
            validate_expression_depth(value, depth + 1)?;
            validate_expression_depth(index, depth + 1)
        }
        ExprKind::Slice {
            value,
            lower,
            upper,
        } => {
            validate_expression_depth(value, depth + 1)?;
            validate_expression_depth(lower, depth + 1)?;
            validate_expression_depth(upper, depth + 1)
        }
        ExprKind::Tuple(elements) => {
            for element in elements {
                validate_expression_depth(element, depth + 1)?;
            }
            Ok(())
        }
        ExprKind::Array(elements) => {
            if elements.is_empty() {
                return Err(AstConstructionError::new(
                    "array literal requires at least one element",
                ));
            }
            for element in elements {
                validate_expression_depth(element, depth + 1)?;
            }
            Ok(())
        }
        ExprKind::Case { value, arms } => {
            validate_expression_depth(value, depth + 1)?;
            super::enumeration::validate_arms(arms)?;
            for arm in arms {
                validate_expression_depth(arm.value(), depth + 1)?;
            }
            Ok(())
        }
        ExprKind::Select {
            condition,
            then_value,
            else_value,
        } => {
            validate_expression_depth(condition, depth + 1)?;
            validate_expression_depth(then_value, depth + 1)?;
            validate_expression_depth(else_value, depth + 1)
        }
        ExprKind::Reduction { binder, value, .. } => {
            super::validate_identifier(binder.member(), "reduction member")?;
            validate_name_path(binder.set())?;
            super::checked_range(binder.range())?;
            validate_expression_depth(value, depth + 1)
        }
        ExprKind::Call { callee, arguments } => {
            validate_name_path(callee)?;
            if matches!(callee.as_str(), "sum" | "product" | "partial" | "evaluate") {
                return Err(AstConstructionError::new(
                    "sum/product/partial/evaluate require structured bindings",
                ));
            }
            if let crate::CallArguments::Mixed { positional, named } = arguments
                && (positional.is_empty() || named.is_empty())
            {
                return Err(AstConstructionError::new(
                    "mixed arguments require both a positional prefix and named suffix",
                ));
            }
            {
                let bindings = arguments.parts().1;
                let mut names = std::collections::HashSet::new();
                for binding in bindings {
                    validate_identifier(binding.name(), "argument name")?;
                    checked_range(binding.range())?;
                    if !names.insert(binding.name()) {
                        return Err(AstConstructionError::new("duplicate named argument"));
                    }
                }
            }
            if callee.as_str() == "tensor_value"
                && !arguments.named().is_some_and(|bindings| {
                    bindings.len() == 2
                        && bindings
                            .iter()
                            .any(|binding| binding.name() == "components")
                        && bindings.iter().any(|binding| {
                            binding.name() == "frame"
                                && matches!(
                                    binding.value().kind(),
                                    ExprKind::Name(_) | ExprKind::Path(_)
                                )
                        })
                })
            {
                return Err(AstConstructionError::new(
                    "tensor_value requires a frame name and components",
                ));
            }
            for argument in arguments.expressions() {
                validate_expression_depth(argument, depth + 1)?;
            }
            Ok(())
        }
    }
}

pub(super) fn validate_endpoint(expression: &Expr) -> Result<(), AstConstructionError> {
    validate_expression(expression)?;
    match expression.kind() {
        ExprKind::Name(_) | ExprKind::Path(_) | ExprKind::Member { .. } => Ok(()),
        _ => Err(AstConstructionError::new(
            "Connection endpoint requires an exact declared Port selection",
        )),
    }
}

impl super::SourceAstFactory {
    /// Construct a bounded scalar reduction with an explicit lexical binder.
    ///
    /// # Errors
    /// Rejects malformed binder names/ranges or excessive expression depth.
    /// The compiler checks index-set identity, cardinality, and scalar domain.
    pub fn reduction(
        operation: crate::ReductionOp,
        binder: crate::FamilyBinderSyntax,
        value: Expr,
        range: crate::TextRange,
    ) -> Result<Expr, AstConstructionError> {
        Self::expression(
            ExprKind::Reduction {
                operation,
                binder,
                value: Box::new(value),
            },
            range,
        )
    }
}
