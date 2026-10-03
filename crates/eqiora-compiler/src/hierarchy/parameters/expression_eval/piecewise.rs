use super::*;
use crate::math::piecewise::{Primitive, cost, emit, result_type};
use eqiora_schema::kernel::typing::ExpressionType;

fn kind(value: &EvaluatedParameter) -> ExpressionType<()> {
    ExpressionType::new(value.value_type.value_type().clone(), None)
}

pub(super) fn select(
    file: &str,
    range: TextRange,
    condition: EvaluatedParameter,
    then_value: EvaluatedParameter,
    else_value: EvaluatedParameter,
) -> Result<EvaluatedParameter, Diagnostic> {
    kind(&condition)
        .select(kind(&then_value), kind(&else_value))
        .map_err(|error| {
            source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string())
        })?;
    let value = condition
        .value
        .as_ref()
        .and_then(ValueLiteral::as_bool)
        .and_then(|condition| {
            if condition {
                then_value.value.clone()
            } else {
                else_value.value.clone()
            }
        });
    Ok(EvaluatedParameter {
        value,
        value_type: then_value.value_type,
        bare_literal: false,
        lineage: combine_lineages(
            condition.lineage,
            combine_lineages(then_value.lineage, else_value.lineage),
        ),
        expression: condition
            .expression
            .zip(then_value.expression)
            .zip(else_value.expression)
            .map(|((condition, then_value), else_value)| {
                LoweringExpression::select(condition, then_value, else_value, range)
            }),
    })
}

pub(super) fn sugar(
    file: &str,
    range: TextRange,
    name: &str,
    count: usize,
    evaluate_values: bool,
    mut operand: impl FnMut(usize, bool) -> Result<EvaluatedParameter, Diagnostic>,
) -> Result<EvaluatedParameter, Diagnostic> {
    let mut cached: Vec<[Option<EvaluatedParameter>; 2]> = vec![[None, None]; count];
    let mut operand = |index: usize, demand: bool| {
        let slot = usize::from(demand);
        if let Some(value) = &cached[index][slot] {
            return Ok(value.clone());
        }
        let value = operand(index, demand)?;
        cached[index][slot] = Some(value.clone());
        Ok::<_, Diagnostic>(value)
    };
    let types = (0..count)
        .map(|index| operand(index, false).map(|value| kind(&value)))
        .collect::<Result<Vec<_>, _>>()?;
    result_type(name, &types).map_err(|error| {
        source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string())
    })?;
    let mut nodes = Vec::with_capacity(cost(name).expect("checked sugar name").0);
    let root = emit(
        name,
        &(0..count).collect::<Vec<_>>(),
        types[0].dimension(),
        |node| {
            nodes.push(node);
            Ok::<_, Diagnostic>(count + nodes.len() - 1)
        },
    )
    .expect("checked sugar arity")?;
    evaluate_node(
        file,
        range,
        root,
        count,
        &nodes,
        evaluate_values,
        &mut operand,
    )
}

fn evaluate_node(
    file: &str,
    range: TextRange,
    index: usize,
    count: usize,
    nodes: &[Primitive<usize>],
    evaluate_values: bool,
    operand: &mut impl FnMut(usize, bool) -> Result<EvaluatedParameter, Diagnostic>,
) -> Result<EvaluatedParameter, Diagnostic> {
    if index < count {
        return operand(index, evaluate_values);
    }
    let error = |message: String| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, message);
    let mut evaluate =
        |index, demand| evaluate_node(file, range, index, count, nodes, demand, operand);
    Ok(match &nodes[index - count] {
        Primitive::Constant(number, dimension) => {
            let value = ValueLiteral::from_real(
                ValueType::scalar(ScalarDomain::Real, *dimension)
                    .expect("valid numeric scalar type"),
                f64::from(*number),
            )
            .unwrap();
            EvaluatedParameter {
                value_type: EvaluatedType::Known(value.value_type().clone()),
                expression: Some(LoweringExpression::literal(value.clone(), range)),
                value: evaluate_values.then_some(value),
                bare_literal: false,
                lineage: Some(ParameterLineage::Constant),
            }
        }
        Primitive::Compare(op, left, right) => {
            use eqiora_schema::kernel::ComparisonOp;
            let op = match op {
                ComparisonOp::Equal => BinaryOp::Equal,
                ComparisonOp::NotEqual => BinaryOp::NotEqual,
                ComparisonOp::Less => BinaryOp::Less,
                ComparisonOp::LessEqual => BinaryOp::LessEqual,
                ComparisonOp::Greater => BinaryOp::Greater,
                ComparisonOp::GreaterEqual => BinaryOp::GreaterEqual,
            };
            super::super::predicates::combine(
                file,
                range,
                op,
                evaluate(*left, evaluate_values)?,
                evaluate(*right, evaluate_values)?,
            )?
        }
        Primitive::Select {
            condition,
            then_value,
            else_value,
        } => {
            let condition = evaluate(*condition, evaluate_values)?;
            let demand = condition.value.as_ref().and_then(ValueLiteral::as_bool);
            let then_value = evaluate(*then_value, evaluate_values && demand == Some(true))?;
            let else_value = evaluate(*else_value, evaluate_values && demand == Some(false))?;
            select(file, range, condition, then_value, else_value)?
        }
        Primitive::Require { condition, value } => {
            let condition = evaluate(*condition, evaluate_values)?;
            let demand = condition.value.as_ref().and_then(ValueLiteral::as_bool);
            let mut value = evaluate(*value, evaluate_values && demand == Some(true))?;
            kind(&condition)
                .require(kind(&value))
                .map_err(|violation| error(violation.to_string()))?;
            if evaluate_values && demand == Some(false) {
                return Err(error("math.clamp requires lower <= upper".into()));
            }
            value.expression = condition
                .expression
                .zip(value.expression)
                .map(|(condition, value)| LoweringExpression::require(condition, value, range));
            value.lineage = combine_lineages(condition.lineage, value.lineage);
            value
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn expression(source: &str) -> Expr {
        let source = format!("model M() {{ let value = {source}; }}");
        let document = eqiora_lang::parse("piecewise.eqi", &source)
            .into_document()
            .unwrap();
        let eqiora_lang::Item::Let(value) = &document.models()[0].items()[0] else {
            panic!("let")
        };
        value.value().clone()
    }
    fn evaluate(source: &str) -> Result<EvaluatedParameter, Diagnostic> {
        evaluate_mode(
            "piecewise.eqi",
            &expression(source),
            ExpressionContext::Let,
            &mut |_, range| {
                Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    "piecewise.eqi",
                    range,
                    "unknown Parameter",
                ))
            },
            (&mut |_| None, &mut |_| None),
            None,
            true,
        )
    }
    fn real(source: &str) -> f64 {
        evaluate(source)
            .unwrap()
            .value
            .unwrap()
            .real_scalar_value()
            .unwrap()
            .value()
    }
    #[test]
    fn conditional_values_are_lazy_but_both_types_are_checked() {
        assert_eq!(real("if true then 2 else math.sqrt(-1)"), 2.0);
        assert_eq!(real("if false then 1/0 else 3"), 3.0);
        assert_eq!(real("if 2[m] < 3[m] then 4[s] else 5[s]"), 4.0);
        for source in [
            "if true then 1[m] else 2[s]",
            "if true then 1 else false",
            "if 1 then 2 else 3",
            "if true then 1 else unknown",
            "if false then 1 else math.sqrt(-1)",
        ] {
            assert!(evaluate(source).is_err(), "{source}");
        }
    }
    #[test]
    fn sugar_uses_exact_endpoints_and_lazy_domain_checks() {
        for (source, expected) in [
            ("math.abs(-2[m])", 2.0),
            ("math.min(2[m],3[m])", 2.0),
            ("math.max(2[m],3[m])", 3.0),
            ("math.clamp(-1,0,2)", 0.0),
            ("math.clamp(0,0,2)", 0.0),
            ("math.clamp(2,0,2)", 2.0),
            ("math.clamp(3,0,2)", 2.0),
            ("math.sign(-1[m])", -1.0),
            ("math.sign(0[m])", 0.0),
            ("math.sign(1[m])", 1.0),
            ("math.step(-1[m])", 0.0),
            ("math.step(0[m])", 1.0),
            ("math.step(1[m])", 1.0),
            ("if true then 3 else math.clamp(1,2,0)", 3.0),
        ] {
            assert_eq!(real(source), expected, "{source}");
        }
        for source in ["math.clamp(1,2,0)", "math.min(1[m],2[s])", "math.abs(true)"] {
            assert!(evaluate(source).is_err(), "{source}");
        }
    }
    #[test]
    fn inactive_typed_array_checks_types_without_evaluating_components() {
        let evaluated = super::super::super::value_expressions::evaluate_mode(
            "piecewise.eqi",
            &expression("[1/0,2]"),
            ExpressionContext::Let,
            &mut |_, _| unreachable!(),
            (&mut |_| None, &mut |_| None),
            Some(
                &ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                    .expect("valid numeric scalar type")
                    .array(2)
                    .unwrap(),
            ),
            false,
        )
        .unwrap();
        assert!(evaluated.value.is_none());
    }
    #[test]
    fn dependency_collection_retains_both_conditional_arms() {
        let (dependencies, errors) =
            super::super::super::dependencies::collect_expression_dependencies(
                "piecewise.eqi",
                &expression("if true then a else math.abs(b)"),
                |name| matches!(name, "a" | "b"),
                ExpressionContext::Default,
            );
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            dependencies.keys().map(String::as_str).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }
}
