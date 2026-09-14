use super::*;
use crate::kernel::pure_operator::{CalculusBuilder, CalculusNode, PureValueClass};
use crate::kernel::{ExprDagBuilder, ExprNode, SymbolRef, UnaryMathFunction};
use eqiora_core::{DimExponents, DynQuantity};

fn constant(builder: &mut ExprDagBuilder, value: f64) -> ExprId {
    builder
        .constant(DynQuantity::new(value, DimExponents::DIMENSIONLESS))
        .unwrap()
}

#[test]
fn independent_cubic_binomial_identity_rejects_coefficient_sign_and_field_mutations() {
    // d[(x+y)^3]/dt = 3(x+y)^2(x'+y'), independently by the binomial theorem.
    let mut builder = ExprDagBuilder::new();
    let x_field = Id::new();
    let y_field = Id::new();
    let x = builder.symbol(SymbolRef::Field(x_field)).unwrap();
    let y = builder.symbol(SymbolRef::Field(y_field)).unwrap();
    let dx = builder.symbol(SymbolRef::Derivative(x_field)).unwrap();
    let dy = builder.symbol(SymbolRef::Derivative(y_field)).unwrap();
    let z_rate = builder.symbol(SymbolRef::Derivative(Id::new())).unwrap();
    let sum = builder.add(x, y).unwrap();
    let storage = builder.powi(sum, 3).unwrap();
    let squared = builder.powi(sum, 2).unwrap();
    let three = constant(&mut builder, 3.0);
    let six = constant(&mut builder, 6.0);
    let scale = builder.mul(three, squared).unwrap();
    let rate = builder.add(dx, dy).unwrap();
    let correct = builder.mul(scale, rate).unwrap();
    let wrong_scale = builder.mul(six, squared).unwrap();
    let wrong_coefficient = builder.mul(wrong_scale, rate).unwrap();
    let wrong_sign = builder.neg(correct).unwrap();
    let wrong_rate = builder.add(dx, z_rate).unwrap();
    let wrong_field = builder.mul(scale, wrong_rate).unwrap();
    let missing_chain = builder.mul(scale, dx).unwrap();
    // Storage itself is deliberately not an output root or a balance ancestor.
    let dag = builder.finish([correct]).unwrap();
    assert_eq!(dag.verify_time_derivative(storage, correct), Ok(()));
    for mutation in [wrong_coefficient, wrong_sign, wrong_field, missing_chain] {
        assert_eq!(
            dag.verify_time_derivative(storage, mutation),
            Err(TimeDerivativeProofError::Mismatch)
        );
    }
}

#[test]
fn explicit_time_parameter_and_field_coefficients_keep_exact_symbol_identity() {
    // d[p*t^2*x]/dt = p*(2*t*x + t^2*x'); p is fixed.
    let mut builder = ExprDagBuilder::new();
    let field = Id::new();
    let parameter = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
    let other_parameter = builder.symbol(SymbolRef::Parameter(Id::new())).unwrap();
    let time = builder.symbol(SymbolRef::Time).unwrap();
    let x = builder.symbol(SymbolRef::Field(field)).unwrap();
    let dx = builder.symbol(SymbolRef::Derivative(field)).unwrap();
    let squared_time = builder.powi(time, 2).unwrap();
    let time_x = builder.mul(squared_time, x).unwrap();
    let storage = builder.mul(parameter, time_x).unwrap();
    let two = constant(&mut builder, 2.0);
    let twice_time = builder.mul(two, time).unwrap();
    let first = builder.mul(twice_time, x).unwrap();
    let second = builder.mul(squared_time, dx).unwrap();
    let sum = builder.add(first, second).unwrap();
    let correct = builder.mul(parameter, sum).unwrap();
    let wrong_parameter = builder.mul(other_parameter, sum).unwrap();
    let missing_time = builder.mul(parameter, second).unwrap();
    let dag = builder.finish([correct]).unwrap();
    assert_eq!(dag.verify_time_derivative(storage, correct), Ok(()));
    for mutation in [wrong_parameter, missing_time] {
        assert_eq!(
            dag.verify_time_derivative(storage, mutation),
            Err(TimeDerivativeProofError::Mismatch)
        );
    }
}

#[test]
fn retained_pure_definition_keeps_exact_thirds_and_substitutes_formal_arguments() {
    // d[x^3/3]/dt = x^2*x'; no binary64 representation of 1/3 participates.
    let mut calculus = CalculusBuilder::new(
        vec![PureValueClass::invariant_scalar()],
        PureValueClass::invariant_scalar(),
    )
    .unwrap();
    let formal = calculus
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let third = calculus
        .push(CalculusNode::Rational {
            value: ExactRational::new(1, 3).unwrap(),
            dimension: DimExponents::DIMENSIONLESS,
        })
        .unwrap();
    let square = calculus.push(CalculusNode::Mul(formal, formal)).unwrap();
    let cube = calculus.push(CalculusNode::Mul(square, formal)).unwrap();
    let scaled = calculus.push(CalculusNode::Mul(third, cube)).unwrap();
    let definition = calculus.finish(scaled).unwrap();
    let mut builder = ExprDagBuilder::new();
    let field = Id::new();
    let x = builder.symbol(SymbolRef::Field(field)).unwrap();
    let dx = builder.symbol(SymbolRef::Derivative(field)).unwrap();
    let storage = builder.pure_operator(&definition, [x]).unwrap();
    let x_squared = builder.mul(x, x).unwrap();
    let correct = builder.mul(x_squared, dx).unwrap();
    let rounded_third = constant(&mut builder, 1.0 / 3.0);
    let three = constant(&mut builder, 3.0);
    let rounded_one = builder.mul(three, rounded_third).unwrap();
    let rounded = builder.mul(rounded_one, correct).unwrap();
    let dag = builder.finish([correct]).unwrap();
    assert_eq!(dag.verify_time_derivative(storage, correct), Ok(()));
    assert_eq!(
        dag.verify_time_derivative(storage, rounded),
        Err(TimeDerivativeProofError::Mismatch)
    );
}

#[test]
fn constants_cancellation_and_unrelated_dead_nodes_are_handled_without_weakening_admission() {
    let mut builder = ExprDagBuilder::new();
    let field = Id::new();
    let x = builder.symbol(SymbolRef::Field(field)).unwrap();
    let zero = constant(&mut builder, 0.0);
    let one = constant(&mut builder, 1.0);
    let canceled = builder.sub(x, x).unwrap();
    let unsupported_dead = builder.unary_math(UnaryMathFunction::Sin, x).unwrap();
    let hidden_unsupported = builder.mul(zero, unsupported_dead).unwrap();
    let dag = builder.finish([zero]).unwrap();
    assert_eq!(dag.verify_time_derivative(canceled, zero), Ok(()));
    assert_eq!(dag.verify_time_derivative(one, canceled), Ok(()));
    assert_eq!(
        dag.verify_time_derivative(hidden_unsupported, zero),
        Err(TimeDerivativeProofError::UnsupportedExpression)
    );
}

#[test]
fn unsupported_storage_symbols_and_nonsmooth_or_nonpolynomial_expressions_fail_closed() {
    let mut builder = ExprDagBuilder::new();
    let field = Id::new();
    let x = builder.symbol(SymbolRef::Field(field)).unwrap();
    let zero = constant(&mut builder, 0.0);
    let mut unsupported_symbols = Vec::new();
    for symbol in [
        SymbolRef::Derivative(field),
        SymbolRef::Pre(field),
        SymbolRef::Next(field),
        SymbolRef::Port(Id::new()),
    ] {
        unsupported_symbols.push(builder.symbol(symbol).unwrap());
    }
    let predicate = builder
        .constant(eqiora_core::ValueLiteral::boolean(true))
        .unwrap();
    let require = builder.require(predicate, x).unwrap();
    let select = builder.select(predicate, x, x).unwrap();
    let division = builder.div(x, x).unwrap();
    let negative_power = builder.powi(x, -1).unwrap();
    let sqrt = builder.unary_math(UnaryMathFunction::Sqrt, x).unwrap();
    let dag = builder.finish([zero]).unwrap();
    for storage in unsupported_symbols {
        assert_eq!(
            dag.verify_time_derivative(storage, zero),
            Err(TimeDerivativeProofError::UnsupportedStorageSymbol)
        );
    }
    for storage in [require, select, division, negative_power, sqrt] {
        assert_eq!(
            dag.verify_time_derivative(storage, zero),
            Err(TimeDerivativeProofError::UnsupportedExpression)
        );
    }
    assert_eq!(
        dag.verify_time_derivative(ExprId(u32::MAX), zero),
        Err(TimeDerivativeProofError::InvalidExpression)
    );
}

#[test]
fn graph_and_polynomial_growth_limits_reject_before_unbounded_work() {
    let mut builder = ExprDagBuilder::new();
    let x = builder.symbol(SymbolRef::Field(Id::new())).unwrap();
    let zero = constant(&mut builder, 0.0);
    let enormous = builder.powi(x, i32::MAX).unwrap();
    let dag = builder.finish([enormous]).unwrap();
    assert!(matches!(
        dag.verify_time_derivative(enormous, zero),
        Err(TimeDerivativeProofError::Limit
            | TimeDerivativeProofError::Polynomial(ExactPolynomialError::Limit))
    ));
    let mut builder = ExprDagBuilder::new();
    let zero = constant(&mut builder, 0.0);
    for _ in 0..MAX_PROOF_NODES {
        builder.push(ExprNode::Neg(zero)).unwrap();
    }
    let dag = builder.finish([zero]).unwrap();
    assert_eq!(
        dag.verify_time_derivative(zero, zero),
        Err(TimeDerivativeProofError::Limit)
    );
}
