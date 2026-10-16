//! Nominal discrete predicates use the same typed demand evaluator as numeric branches.
use super::*;
use eqiora_core::{Id, ValueLiteral, ValueType};
use eqiora_schema::kernel::{ComparisonOp, ExprDagBuilder};

#[test]
fn enum_predicates_select_numeric_values_without_reading_inactive_symbols() {
    let ty = ValueType::enumeration(Id::new(), 2).unwrap();
    let mode = SymbolRef::Parameter(Id::new());
    let inactive = SymbolRef::Parameter(Id::new());
    let mut builder = ExprDagBuilder::new();
    let input = builder.symbol(mode).unwrap();
    let member = builder
        .constant(ValueLiteral::enum_value(ty.clone(), 1).unwrap())
        .unwrap();
    let equal = builder.compare(ComparisonOp::Equal, input, member).unwrap();
    let unequal = builder
        .compare(ComparisonOp::NotEqual, input, member)
        .unwrap();
    let numeric = builder
        .constant(eqiora_core::DynQuantity::new(
            7.0,
            eqiora_core::DimExponents::DIMENSIONLESS,
        ))
        .unwrap();
    let missing = builder.symbol(inactive).unwrap();
    let selected = builder.select(equal, numeric, missing).unwrap();
    let dag = builder.finish([equal, unequal, selected]).unwrap();
    let ir = ScalarOperatorIr::lower(&dag).unwrap();
    let values = ir
        .evaluate_typed(&[equal, unequal, selected], &mut |symbol| {
            assert_eq!(symbol, mode, "inactive branch must not resolve its symbol");
            Some(ValueLiteral::enum_value(ty.clone(), 1).unwrap())
        })
        .unwrap();
    assert_eq!(values[0].as_bool(), Some(true));
    assert_eq!(values[1].as_bool(), Some(false));
    assert_eq!(values[2].real_scalar_value().unwrap().value(), 7.0);
    assert!(
        ir.evaluate_typed(&[equal], &mut |_| Some(
            ValueLiteral::enum_value(ValueType::enumeration(Id::new(), 2).unwrap(), 1).unwrap()
        ))
        .is_err()
    );
    let values = ir
        .evaluate_typed(&[equal, unequal], &mut |_| {
            Some(ValueLiteral::enum_value(ty.clone(), 0).unwrap())
        })
        .unwrap();
    assert_eq!(values[0].as_bool(), Some(false));
    assert_eq!(values[1].as_bool(), Some(true));
}

#[test]
fn enums_have_no_numeric_execution_or_differentiation_profile() {
    let ty = ValueType::enumeration(Id::new(), 2).unwrap();
    let value = ValueLiteral::enum_value(ty, 0).unwrap();
    let mut builder = ExprDagBuilder::new();
    let input = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
    let member = builder.constant(value.clone()).unwrap();
    let ordered = builder.compare(ComparisonOp::Less, input, member).unwrap();
    let sum = builder.add(input, member).unwrap();
    let dag = builder.finish([input, ordered, sum]).unwrap();
    let ir = ScalarOperatorIr::lower(&dag).unwrap();
    assert_eq!(
        ir.evaluate_typed(&[input], &mut |_| Some(value.clone()))
            .unwrap()[0],
        value
    );
    for root in [ordered, sum] {
        assert!(
            ir.evaluate_typed(&[root], &mut |_| Some(value.clone()))
                .is_err()
        );
    }
    assert!(ir.evaluate(&[0.0]).is_err());
    for role in [
        DifferentiationRole::Unknown,
        DifferentiationRole::Parameter,
        DifferentiationRole::Frozen,
    ] {
        // Frozen nominal values may now select real branches, but neither
        // ordered enum comparisons nor enum addition acquire numeric meaning.
        let rejection = if role == DifferentiationRole::Frozen {
            "incompatible expression types"
        } else {
            "real scalar"
        };
        assert!(
            ir.linearize_typed(std::slice::from_ref(&value), &[role])
                .unwrap_err()
                .message()
                .contains(rejection)
        );
    }
}

#[test]
fn frozen_nominal_outcomes_preserve_real_parameter_actions_and_normalization() {
    use eqiora_core::{DimExponents, DynQuantity};
    use eqiora_schema::kernel::UnaryMathFunction;
    let outcome_type = ValueType::enumeration(Id::new(), 2).unwrap();
    let outcome = SymbolRef::Outcome(Id::new());
    let parameter = SymbolRef::Parameter(Id::new());
    let mut builder = ExprDagBuilder::new();
    let draw = builder.symbol(outcome).unwrap();
    let theta = builder.symbol(parameter).unwrap();
    let low = builder
        .constant(ValueLiteral::enum_value(outcome_type.clone(), 0).unwrap())
        .unwrap();
    let two = builder
        .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let three = builder
        .constant(DynQuantity::new(3.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let six = builder
        .constant(DynQuantity::new(6.0, DimExponents::DIMENSIONLESS))
        .unwrap();
    let low = builder.compare(ComparisonOp::Equal, draw, low).unwrap();
    let low_loss = builder.add(two, theta).unwrap();
    let twice = builder.mul(two, theta).unwrap();
    let high_loss = builder.add(six, twice).unwrap();
    let loss = builder.select(low, low_loss, high_loss).unwrap();
    let weight = builder.select(low, three, theta).unwrap();
    let total = builder.add(three, theta).unwrap();
    let probability = builder.div(weight, total).unwrap();
    let log_probability = builder
        .unary_math(UnaryMathFunction::Log, probability)
        .unwrap();
    let ir = ScalarOperatorIr::lower(&builder.finish([loss, log_probability]).unwrap()).unwrap();
    let roles = [DifferentiationRole::Frozen, DifferentiationRole::Parameter];
    let real =
        |x| ValueLiteral::try_from(DynQuantity::new(x, DimExponents::DIMENSIONLESS)).unwrap();
    let mut expectation_gradient = 0.0;
    // Weights (3, t), losses (2+t, 6+2t) give E=(6+9t+2t²)/(3+t).
    // At t=1, E=17/4 and E'=35/16. The two normalized log-weight
    // derivatives are -1/4 and 3/4, including the derivative of total mass.
    for (tag, loss, probability, loss_derivative, score) in [
        (0, 3.0, 0.75_f64, 1.0, -0.25),
        (1, 8.0, 0.25_f64, 2.0, 0.75),
    ] {
        let point = [
            ValueLiteral::enum_value(outcome_type.clone(), tag).unwrap(),
            real(1.0),
        ];
        let relation = ir.linearize_typed(&point, &roles).unwrap();
        assert_eq!(relation.unknown_dimension(), 0);
        assert_eq!(relation.parameter_dimension(), 1);
        let mut values = [0.0; 2];
        relation.primal(&mut values).unwrap();
        assert_eq!(values, [loss, probability.ln()]);
        relation
            .jvp(RelationTangent::Parameter(&[1.0]), &mut values)
            .unwrap();
        assert_eq!(values, [loss_derivative, score]);
        let mut combined = [0.0];
        relation
            .vjp(&[1.0, loss], RelationCotangent::Parameter(&mut combined))
            .unwrap();
        assert_eq!(combined, [loss_derivative + loss * score]);
        expectation_gradient += probability * combined[0];
        // Discrete coordinates cannot acquire an active tangent space.
        assert!(
            ir.linearize_typed(&point, &[DifferentiationRole::Parameter; 2])
                .is_err()
        );
        assert!(ir.linearize_typed(&point[..1], &roles).is_err());
    }
    assert_eq!(expectation_gradient, 35.0 / 16.0);
    let foreign =
        ValueLiteral::enum_value(ValueType::enumeration(Id::new(), 2).unwrap(), 0).unwrap();
    assert!(ir.linearize_typed(&[foreign, real(1.0)], &roles).is_err());
    let high = ValueLiteral::enum_value(outcome_type, 1).unwrap();
    assert!(ir.linearize_typed(&[high, real(0.0)], &roles).is_err());
}
