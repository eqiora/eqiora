//! Independent typed branch/domain falsifiers; numerical laziness belongs to runtime tests.
use super::*;
use crate::kernel::typing::{RootContract, TypedResidual};
use crate::kernel::{ComparisonOp, ExprDagBuilder, ExprNode, UnaryMathFunction};
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};

fn class(dimension: DimExponents) -> PureValueClass {
    PureValueClass::invariant_scalar()
        .with_dimension(dimension)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap()
}
fn formal(builder: &mut CalculusBuilder, index: u16) -> CalculusNodeId {
    builder
        .push(CalculusNode::FormalComponent {
            formal: index,
            axes: Box::new([]),
        })
        .unwrap()
}
fn literal(builder: &mut CalculusBuilder, value: i64, dimension: DimExponents) -> CalculusNodeId {
    builder
        .push(CalculusNode::Rational {
            value: ExactRational::integer(value),
            dimension,
        })
        .unwrap()
}
fn ty(dimension: DimExponents) -> ExpressionType<u32> {
    ExpressionType::new(
        ValueType::scalar(ScalarDomain::Real, dimension).expect("checked scalar type"),
        None,
    )
}

#[test]
fn typed_zero_threshold_and_inactive_arms_retain_exact_dimensions() {
    let voltage = DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).unwrap();
    let current = DimExponents::from_integers([0, 0, 0, 1, 0, 0, 0]).unwrap();
    let conductance = current.div(voltage).unwrap();
    for (zero_dimension, wrong_arm, valid) in [
        (voltage, false, true),
        (DimExponents::DIMENSIONLESS, false, false),
        (voltage, true, false),
    ] {
        let mut builder = CalculusBuilder::new(
            [class(voltage), class(conductance), class(conductance)],
            class(current),
        )
        .unwrap();
        let v = formal(&mut builder, 0);
        let negative = formal(&mut builder, 1);
        let positive = formal(&mut builder, 2);
        let zero = literal(&mut builder, 0, zero_dimension);
        let predicate = builder
            .push(CalculusNode::Compare(ComparisonOp::Less, v, zero))
            .unwrap();
        let left = builder.push(CalculusNode::Mul(negative, v)).unwrap();
        let right = if wrong_arm {
            v
        } else {
            builder.push(CalculusNode::Mul(positive, v)).unwrap()
        };
        let root = builder
            .push(CalculusNode::Select {
                condition: predicate,
                then_value: left,
                else_value: right,
            })
            .unwrap();
        assert_eq!(builder.finish(root).is_ok(), valid);
    }
}

#[test]
fn square_root_uses_rational_formal_exponents_and_typed_literal_factors() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let root_dimension = length.pow(1, 2).unwrap();
    let mut builder = CalculusBuilder::new([class(length)], class(root_dimension)).unwrap();
    let input = formal(&mut builder, 0);
    let root = builder
        .push(CalculusNode::UnaryMath(UnaryMathFunction::Sqrt, input))
        .unwrap();
    let definition = builder.finish(root).unwrap();
    assert_eq!(
        definition.dimension_monomial().exponents(),
        [ExactRational::new(1, 2).unwrap()]
    );
    assert_eq!(
        definition
            .instantiate(&[ty(length)])
            .unwrap()
            .result_type()
            .dimension(),
        root_dimension
    );

    let mut literal_builder =
        CalculusBuilder::new([class(DimExponents::DIMENSIONLESS)], class(length)).unwrap();
    let value = literal(&mut literal_builder, 4, length.pow(2, 1).unwrap());
    let root = literal_builder
        .push(CalculusNode::UnaryMath(UnaryMathFunction::Sqrt, value))
        .unwrap();
    let definition = literal_builder.finish(root).unwrap();
    assert_eq!(definition.dimension_monomial().fixed_dimension(), length);
    assert!(
        definition
            .dimension_monomial()
            .exponents()
            .iter()
            .all(|power| power.is_zero())
    );
    let instance = definition
        .instantiate(&[ty(DimExponents::DIMENSIONLESS)])
        .unwrap();
    let mut dag = ExprDagBuilder::new();
    let unused = dag
        .constant(
            ValueLiteral::from_real(
                ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                    .expect("checked scalar type"),
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
    let projected = dag
        .project_operator_component(&instance, &[[unused]], &[], 3)
        .unwrap();
    let dag = dag.finish([projected]).unwrap();
    assert!(
        matches!(&dag.nodes()[1],ExprNode::Constant(value) if value.value_type().dimension()==length.pow(2,1).unwrap())
    );
    assert!(matches!(
        dag.node(projected),
        Some(ExprNode::UnaryMath(UnaryMathFunction::Sqrt, _))
    ));
}

#[test]
fn require_and_select_validate_all_static_operands_without_numeric_conditions() {
    for numeric_guard in [false, true] {
        let mut builder = ExprDagBuilder::new();
        let value = builder
            .constant(
                ValueLiteral::from_real(
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("checked scalar type"),
                    2.0,
                )
                .unwrap(),
            )
            .unwrap();
        let truth = builder.constant(ValueLiteral::boolean(false)).unwrap();
        let guarded = builder
            .require(if numeric_guard { value } else { truth }, value)
            .unwrap();
        let selected = builder.select(truth, guarded, value).unwrap();
        let dag = builder.finish([selected, value]).unwrap();
        assert_eq!(
            TypedResidual::<u32>::infer(
                dag,
                None,
                |_| None,
                RootContract::EquationSides,
                |_| Err::<ExpressionType<u32>, ()>(())
            )
            .is_ok(),
            !numeric_guard
        );
    }
    let mut builder = CalculusBuilder::new(
        [class(DimExponents::DIMENSIONLESS)],
        class(DimExponents::DIMENSIONLESS),
    )
    .unwrap();
    let value = formal(&mut builder, 0);
    let bad = builder
        .push(CalculusNode::Require {
            condition: value,
            value,
        })
        .unwrap();
    assert_eq!(
        builder.finish(bad).unwrap_err(),
        PureOperatorError::FormalTypeMismatch
    );
    let mut builder = CalculusBuilder::new(
        [class(DimExponents::DIMENSIONLESS)],
        class(DimExponents::DIMENSIONLESS),
    )
    .unwrap();
    let value = formal(&mut builder, 0);
    let condition = builder.push(CalculusNode::Boolean(false)).unwrap();
    let guarded = builder
        .push(CalculusNode::Require { condition, value })
        .unwrap();
    let root = builder
        .push(CalculusNode::Select {
            condition,
            then_value: guarded,
            else_value: value,
        })
        .unwrap();
    let definition = builder.finish(root).unwrap();
    let instance = definition
        .instantiate(&[ty(DimExponents::DIMENSIONLESS)])
        .unwrap();
    let mut dag = ExprDagBuilder::new();
    let argument = dag
        .constant(
            ValueLiteral::from_real(
                ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                    .expect("checked scalar type"),
                2.0,
            )
            .unwrap(),
        )
        .unwrap();
    let root = dag
        .project_operator_component(&instance, &[[argument]], &[], 4)
        .unwrap();
    let dag = dag.finish([root]).unwrap();
    assert!(matches!(dag.nodes()[2], ExprNode::Require { .. }));
    assert!(matches!(dag.node(root), Some(ExprNode::Select { .. })));
}

#[test]
fn nonlinear_profile_and_fraction_bounds_reject_before_projection() {
    let generic = PureValueClass::invariant_scalar();
    let mut builder = CalculusBuilder::new([generic], generic).unwrap();
    let input = formal(&mut builder, 0);
    let condition = builder.push(CalculusNode::Boolean(true)).unwrap();
    let root = builder
        .push(CalculusNode::Select {
            condition,
            then_value: input,
            else_value: input,
        })
        .unwrap();
    assert_eq!(
        builder.finish(root).unwrap_err(),
        PureOperatorError::FormalTypeMismatch
    );
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let mut builder = CalculusBuilder::new([class(length)], class(length)).unwrap();
    let mut root = formal(&mut builder, 0);
    assert!({
        let mut invalid = CalculusBuilder::new([class(length)], class(length)).unwrap();
        let root = formal(&mut invalid, 0);
        let root = invalid
            .push(CalculusNode::UnaryMath(UnaryMathFunction::Sin, root))
            .unwrap();
        invalid.finish(root).is_err()
    });
    for _ in 0..31 {
        root = builder
            .push(CalculusNode::UnaryMath(UnaryMathFunction::Sqrt, root))
            .unwrap();
    }
    assert_eq!(
        builder.finish(root).unwrap_err(),
        PureOperatorError::ResultDimensionOverflow
    );
}

#[test]
fn builder_component_type_uses_the_same_concrete_dimension_proof() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let mut builder = CalculusBuilder::new([class(length)], class(length)).unwrap();
    let value = formal(&mut builder, 0);
    assert_eq!(
        builder.value_type(value).unwrap(),
        ValueType::scalar(ScalarDomain::Real, length).expect("checked scalar type")
    );
    let predicate = builder
        .push(CalculusNode::Compare(ComparisonOp::Equal, value, value))
        .unwrap();
    assert_eq!(builder.value_type(predicate).unwrap(), ValueType::boolean());
    let mut generic = CalculusBuilder::new(
        [PureValueClass::invariant_scalar()],
        PureValueClass::invariant_scalar(),
    )
    .unwrap();
    let value = formal(&mut generic, 0);
    assert_eq!(
        generic.value_type(value),
        Err(PureOperatorError::FormalTypeMismatch)
    );
}
