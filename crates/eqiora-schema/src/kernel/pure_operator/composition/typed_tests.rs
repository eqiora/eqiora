use super::*;
use eqiora_core::{ScalarDomain, ValueType};

fn scalar() -> PureValueClass {
    PureValueClass::from_value_type(
        ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap(),
    )
    .unwrap()
}

fn matrix(rows: u32, columns: u32) -> PureValueClass {
    PureValueClass::from_value_type(
        scalar()
            .exact_type()
            .unwrap()
            .clone()
            .array(columns)
            .unwrap()
            .array(rows)
            .unwrap(),
    )
    .unwrap()
}

fn projection(
    input: PureValueClass,
    output: PureValueClass,
    axes: impl Into<Box<[ComponentIndex]>>,
) -> PureOperatorDefinition {
    let mut builder = CalculusBuilder::new([input], output).unwrap();
    let root = builder
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: axes.into(),
        })
        .unwrap();
    builder.finish(root).unwrap()
}

#[test]
fn nested_rectangular_values_rebind_their_own_result_axes() {
    use ComponentIndex::{Fixed, Result as Axis};
    let input = matrix(3, 2);
    let transposed = matrix(2, 3);
    let argument = projection(input.clone(), transposed.clone(), [Axis(1), Axis(0)]);
    for (result, axes, expected) in [
        (
            input.clone(),
            vec![Axis(1), Axis(0)],
            vec![Axis(0), Axis(1)],
        ),
        (scalar(), vec![Fixed(1), Fixed(0)], vec![Fixed(0), Fixed(1)]),
    ] {
        let callee = projection(transposed.clone(), result.clone(), axes.into_boxed_slice());
        let mut caller = CalculusBuilder::new([input.clone()], result).unwrap();
        let root = caller
            .apply_inputs(&callee, &[argument.clone().into()])
            .unwrap();
        assert_eq!(
            caller.nodes[root.index() as usize],
            CalculusNode::FormalComponent {
                formal: 0,
                axes: expected.into_boxed_slice(),
            }
        );
        let definition = caller.finish(root).unwrap();
        definition
            .instantiate(&[ExpressionType::<()>::new(
                input.exact_type().unwrap().clone(),
                None,
            )])
            .unwrap();
    }
}

#[test]
fn whole_arguments_reject_wrong_contracts_and_foreign_environments_atomically() {
    use ComponentIndex::Result as Axis;
    let expected = matrix(2, 3);
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let callee = projection(expected.clone(), expected.clone(), [Axis(0), Axis(1)]);
    for actual in [
        matrix(3, 2),
        expected.clone().with_dimension(length),
        expected
            .clone()
            .with_scalar_domain(ScalarDomain::Real)
            .unwrap(),
    ] {
        let value = projection(actual.clone(), actual.clone(), [Axis(0), Axis(1)]);
        let mut caller = CalculusBuilder::new([actual], expected.clone()).unwrap();
        let before = (caller.nodes.clone(), caller.depths.clone());
        assert!(caller.apply_inputs(&callee, &[value.into()]).is_err());
        assert_eq!((caller.nodes, caller.depths), before);
    }
    let foreign = projection(expected.clone(), expected.clone(), [Axis(0), Axis(1)]);
    let mut caller = CalculusBuilder::new([matrix(3, 2)], expected).unwrap();
    assert_eq!(
        caller.apply_inputs(&callee, &[foreign.into()]),
        Err(PureOperatorError::FormalTypeMismatch)
    );
    assert!(caller.nodes.is_empty());
}

#[test]
fn unused_whole_inputs_still_retain_frames_channels_and_nominal_bases() {
    use ComponentIndex::Result as Axis;
    use eqiora_core::{FiniteBasis, Id, ValueFrame, ValueShape};
    let first = FiniteBasis::new(Id::new(), 2).unwrap();
    let second = FiniteBasis::new(Id::new(), 2).unwrap();
    let spatial = |rank| {
        ValueType::shaped(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            ValueShape::new(vec![2; rank]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap()
    };
    let types = [
        matrix(2, 2).exact_type().unwrap().clone(),
        spatial(2),
        spatial(1).array(2).unwrap(),
        ValueType::linear_map(
            first,
            second,
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        ValueType::linear_map(
            second,
            first,
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
    ];
    for (expected_index, expected) in types.iter().enumerate() {
        let expected = PureValueClass::from_value_type(expected.clone()).unwrap();
        let mut callee = CalculusBuilder::new([expected], scalar()).unwrap();
        let zero = callee
            .push(CalculusNode::Rational {
                value: ExactRational::integer(0),
                dimension: DimExponents::DIMENSIONLESS,
            })
            .unwrap();
        let callee = callee.finish(zero).unwrap();
        for (actual_index, actual) in types.iter().enumerate() {
            let actual = PureValueClass::from_value_type(actual.clone()).unwrap();
            let value = projection(actual.clone(), actual.clone(), [Axis(0), Axis(1)]);
            let mut caller = CalculusBuilder::new([actual], scalar()).unwrap();
            let result = caller.apply_inputs(&callee, &[value.into()]);
            assert_eq!(result.is_ok(), expected_index == actual_index);
            if let Ok(root) = result {
                caller.finish(root).unwrap();
            } else {
                assert!(caller.nodes.is_empty());
            }
        }
    }
}

#[test]
fn failed_whole_value_expansion_restores_the_callers_arena() {
    let class = matrix(2, 3);
    let mut value = CalculusBuilder::new([class.clone()], class.clone()).unwrap();
    let zero = CalculusNode::Rational {
        value: ExactRational::integer(0),
        dimension: DimExponents::DIMENSIONLESS,
    };
    for _ in 0..MAX_NODES - 1 {
        value.push(zero.clone()).unwrap();
    }
    let root = value
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([ComponentIndex::Result(0), ComponentIndex::Result(1)]),
        })
        .unwrap();
    let value = value.finish(root).unwrap();
    let callee = projection(
        class.clone(),
        class.clone(),
        [ComponentIndex::Result(0), ComponentIndex::Result(1)],
    );
    let mut caller = CalculusBuilder::new([class.clone()], class).unwrap();
    caller.push(zero).unwrap();
    let before = (caller.nodes.clone(), caller.depths.clone());
    assert_eq!(
        caller.apply_inputs(&callee, &[value.into()]),
        Err(PureOperatorError::NodeLimit)
    );
    assert_eq!((caller.nodes, caller.depths), before);
}

#[test]
fn indexing_rejects_invalid_channels_and_bounded_expansion_atomically() {
    use ComponentIndex::Result as Axis;
    for (rows, index, expected) in [
        (2, 2, PureOperatorError::FormalTypeMismatch),
        (MAX_NODES as u32, 0, PureOperatorError::NodeLimit),
    ] {
        let input = matrix(rows, 2);
        let output = PureValueClass::from_value_type(
            scalar().exact_type().unwrap().clone().array(2).unwrap(),
        )
        .unwrap();
        let argument = projection(input.clone(), input.clone(), [Axis(0), Axis(1)]);
        let mut caller = CalculusBuilder::new([input], output).unwrap();
        caller
            .push(CalculusNode::Rational {
                value: ExactRational::integer(7),
                dimension: DimExponents::DIMENSIONLESS,
            })
            .unwrap();
        let before = (caller.nodes.clone(), caller.depths.clone());
        assert_eq!(caller.index(&argument.into(), index), Err(expected));
        assert_eq!((caller.nodes, caller.depths), before);
    }
}

#[test]
fn complete_index_domains_fit_existing_depth_limits_for_wide_arrays() {
    let input = matrix(300, 1);
    let output =
        PureValueClass::from_value_type(scalar().exact_type().unwrap().clone().array(1).unwrap())
            .unwrap();
    let argument = projection(
        input.clone(),
        input.clone(),
        [ComponentIndex::Result(0), ComponentIndex::Result(1)],
    );
    let mut caller = CalculusBuilder::new([input], output).unwrap();
    let value = caller.index(&argument.into(), 299).unwrap();
    caller.finish(value).unwrap();
}
