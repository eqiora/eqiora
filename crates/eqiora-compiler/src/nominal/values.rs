//! Finite constructors retain nominal identity and reuse the static value evaluator.
use super::*;
use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral};
use eqiora_lang::{Expr, FiniteBasisSyntax};

pub(crate) fn bind_finite_expressions(
    file: &str,
    document: &mut Document,
    spaces: &BTreeMap<String, BoundFiniteSpace>,
) -> Result<(), Vec<Diagnostic>> {
    let mut errors = vec![];
    SourceAstFactory::visit_typed_initializers(document, |syntax, expression| {
        if let Some(expected) = syntax.resolved_nominal()
            && let Err(error) = bind(file, expression, spaces, Some(expected))
        {
            errors.push(error);
        }
    });
    if !errors.is_empty() {
        return Err(errors);
    }
    SourceAstFactory::visit_expressions(document, |_, expression| {
        let expected = expression.resolved_nominal().cloned();
        if let Err(error) = bind(file, expression, spaces, expected.as_ref()) {
            errors.push(error);
        }
    });
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn bind(
    file: &str,
    expression: &mut Expr,
    spaces: &BTreeMap<String, BoundFiniteSpace>,
    expected: Option<&ValueType>,
) -> Result<(), Diagnostic> {
    let ExprKind::Call { callee, arguments } = expression.kind() else {
        return Ok(());
    };
    if !matches!(callee.as_str(), "counts" | "coordinates" | "linear_map") {
        return Ok(());
    }
    let invalid = |message: &str| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    };
    let arguments = arguments
        .positional()
        .ok_or_else(|| invalid("finite constructor requires positional arguments"))?;
    let arity = if callee.as_str() == "linear_map" {
        2
    } else {
        1
    };
    if arguments.len() != arity + 1 {
        return Err(invalid(
            "finite constructor requires its basis arguments and one component array",
        ));
    }
    let mut bases = vec![];
    let mut names = vec![];
    for argument in &arguments[..arity] {
        let syntax = FiniteBasisSyntax::from_expression(argument)
            .ok_or_else(|| invalid("finite constructor requires exact declared basis arguments"))?;
        let declaration = spaces
            .get(syntax.name.as_str())
            .ok_or_else(|| invalid("unresolved finite space in constructor"))?;
        let basis = declaration.definition.basis();
        bases.push(if syntax.dual { basis.dual() } else { basis });
        names.push(syntax.name);
    }
    let value_type = if callee.as_str() == "counts" {
        if bases[0].is_dual() { return Err(invalid("counts require a primal finite basis")); }
        ValueType::counts(bases[0].space().ok_or_else(|| invalid("counts require an atomic finite basis"))?,bases[0].extent())
    } else {
        let scalar = if let Some(expected) = expected {
            (expected.scalar_domain(),expected.dimension())
        } else if arity == 1 && !bases[0].is_dual()
            && matches!(arguments[arity].kind(), ExprKind::Array(values) if values.iter().all(|value| crate::hierarchy::exact_signed_literal(value).is_some())) {
            (ScalarDomain::Integer,DimExponents::DIMENSIONLESS)
        } else {
            let value = crate::hierarchy::infer_closed_value(file,&arguments[arity])?;
            (value.value_type().scalar_domain(),value.value_type().dimension())
        };
        if arity == 1 { ValueType::coordinates(bases[0],scalar.0,scalar.1) }
        else { ValueType::linear_map(bases[0],bases[1],scalar.0,scalar.1) }
    }.map_err(|error| invalid(&error.to_string()))?;
    SourceAstFactory::bind_nominal_expression(expression, &names, value_type).map_err(|error| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            error.message(),
        )
    })?;
    literal(file, expression).map(|_| ())
}

pub(crate) fn literal(file: &str, expression: &Expr) -> Result<ValueLiteral, Diagnostic> {
    if let Some(value) = expression.resolved_enum() {
        return Ok(value.clone());
    }
    let invalid = |message: &str| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    };
    let value_type = expression
        .resolved_nominal()
        .ok_or_else(|| invalid("nominal constructor requires exact lexical resolution"))?;
    literal_with_type(file, expression, value_type)
}

/// A constructor's scalar literals may take their Parameter signature's domain;
/// its exact nominal endpoints are never converted or rebound.
pub(crate) fn contextual_literal(
    file: &str,
    expression: &Expr,
    target: &ValueType,
) -> Result<ValueLiteral, Diagnostic> {
    let declared = expression.resolved_nominal().ok_or_else(|| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            "finite constructor requires exact lexical resolution",
        )
    })?;
    if declared.coordinate_basis() != target.coordinate_basis()
        || declared.map_bases() != target.map_bases()
        || declared.finite_bases().next().is_none()
        || !matches!(
            target.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        )
    {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            "finite constructor does not match the exact Parameter basis",
        ));
    }
    literal_with_type(file, expression, target)
}

fn literal_with_type(
    file: &str,
    expression: &Expr,
    value_type: &ValueType,
) -> Result<ValueLiteral, Diagnostic> {
    let invalid = |message: &str| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            message,
        )
    };
    let ExprKind::Call { arguments, .. } = expression.kind() else {
        return Err(invalid("nominal literal requires a constructor"));
    };
    let argument = arguments
        .positional()
        .and_then(|values| values.last())
        .ok_or_else(|| invalid("nominal constructor requires a component value"))?;
    let mut leaves = vec![];
    collect(argument, value_type.shape().extents(), &mut leaves).map_err(invalid)?;
    if value_type.scalar_domain() == ScalarDomain::Integer {
        let components = leaves
            .into_iter()
            .map(|value| {
                crate::hierarchy::exact_signed_literal(value)
                    .ok_or_else(|| {
                        invalid("nominal integer components require closed exact signed literals")
                    })?
                    .map_err(|error| invalid(error.message()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ValueLiteral::integer(value_type.clone(), components)
            .map_err(|error| invalid(&error.to_string()))
    } else {
        let scalar = ValueType::scalar(value_type.scalar_domain(), value_type.dimension())
            .map_err(|error| invalid(&error.to_string()))?;
        let components = leaves
            .into_iter()
            .map(|value| {
                crate::hierarchy::closed_value(file, value, scalar.clone()).and_then(|value| {
                    value
                        .component(0)
                        .ok_or_else(|| invalid("finite component must be a continuous scalar"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        ValueLiteral::new(value_type.clone(), components)
            .map_err(|error| invalid(&error.to_string()))
    }
}

fn collect<'a>(
    expression: &'a Expr,
    extents: &[std::num::NonZeroU32],
    leaves: &mut Vec<&'a Expr>,
) -> Result<(), &'static str> {
    let Some((extent, rest)) = extents.split_first() else {
        leaves.push(expression);
        return Ok(());
    };
    let ExprKind::Array(values) = expression.kind() else {
        return Err("finite components require explicit arrays matching their exact basis axes");
    };
    if values.len() != extent.get() as usize {
        return Err("finite component array extent differs from its declared basis");
    }
    for value in values {
        collect(value, rest, leaves)?;
    }
    Ok(())
}
