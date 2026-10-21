use super::*;
use eqiora_core::{FiniteBasis, Id, ValueFrame, ValueShape};

fn whole(formals: &[PureValueClass], formal: u16) -> CalculusArgument {
    let result = formals[usize::from(formal)].clone();
    let axes = (0..result.rank())
        .map(|axis| ComponentIndex::Result(axis as u16))
        .collect();
    let mut value = CalculusBuilder::new(formals.iter().cloned(), result).unwrap();
    let root = value
        .push(CalculusNode::FormalComponent { formal, axes })
        .unwrap();
    value.finish(root).unwrap().into()
}

#[test]
fn actions_reject_wrong_complete_contracts_before_mutation() {
    let unit = DimExponents::DIMENSIONLESS;
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let scalar = ValueType::scalar(ScalarDomain::Complex, unit).unwrap();
    let array = scalar.clone().array(2).unwrap();
    let first = FiniteBasis::new(Id::new(), 2).unwrap();
    let second = FiniteBasis::new(Id::new(), 2).unwrap();
    let map = |to, from| ValueType::linear_map(to, from, ScalarDomain::Complex, unit).unwrap();
    let pairs = [
        (
            array.clone(),
            array
                .clone()
                .with_scalar_domain(ScalarDomain::Real)
                .unwrap(),
        ),
        (array.clone(), scalar.array(3).unwrap()),
        (array.clone(), array.clone().with_dimension(length).unwrap()),
        (
            array,
            ValueType::shaped(
                ScalarDomain::Complex,
                unit,
                ValueShape::new([2]).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap(),
        ),
        (map(first, second), map(second, first)),
    ];
    for (expected, actual) in pairs {
        if expected.dimension() == actual.dimension() {
            assert!(
                PureOperatorDefinition::real_pairing(expected.clone(), actual.clone()).is_err()
            );
        }
        let result = PureValueClass::from_value_type(expected).unwrap();
        let formals = [
            result.clone(),
            PureValueClass::from_value_type(actual).unwrap(),
        ];
        let source = whole(&formals, 0);
        let wrong = whole(&formals, 1);
        let mut builder = CalculusBuilder::new(formals, result.clone()).unwrap();
        let root = builder
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: (0..result.rank())
                    .map(|axis| ComponentIndex::Result(axis as u16))
                    .collect(),
            })
            .unwrap();
        let before = (builder.nodes.clone(), builder.depths.clone());
        assert_eq!(
            builder.jvp(&root.into(), &[(0, wrong.clone())]),
            Err(PureOperatorError::FormalTypeMismatch)
        );
        assert_eq!((builder.nodes.clone(), builder.depths.clone()), before);
        assert_eq!(
            builder.vjp(&source, &wrong, 0),
            Err(PureOperatorError::FormalTypeMismatch)
        );
        assert_eq!((builder.nodes, builder.depths), before);
    }
}

#[test]
fn real_pairing_contracts_all_axes_and_multiplies_units() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let frequency = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let array = |dimension| {
        ValueType::scalar(ScalarDomain::Complex, dimension)
            .unwrap()
            .array(2)
            .unwrap()
            .array(3)
            .unwrap()
    };
    let left = array(length);
    let right = array(frequency);
    let pairing = PureOperatorDefinition::real_pairing(left.clone(), right.clone()).unwrap();
    let instance = pairing
        .instantiate(&[
            ExpressionType::<()>::new(left, None),
            ExpressionType::new(right, None),
        ])
        .unwrap();
    assert_eq!(
        instance.result_type().value_type,
        ValueType::scalar(ScalarDomain::Real, speed).unwrap()
    );
    assert!(
        PureOperatorDefinition::real_pairing(ValueType::boolean(), ValueType::boolean()).is_err()
    );
}

#[test]
fn actions_roll_back_partially_appended_work_at_the_shared_node_limit() {
    let result = PureValueClass::from_value_type(
        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
    )
    .unwrap();
    let mut builder = CalculusBuilder::new([result.clone(), result.clone()], result).unwrap();
    let x = builder
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let direction = builder
        .push(CalculusNode::FormalComponent {
            formal: 1,
            axes: Box::new([]),
        })
        .unwrap();
    let source = builder.push(CalculusNode::Mul(x, x)).unwrap();
    while builder.nodes.len() < MAX_NODES - 3 {
        builder
            .push(CalculusNode::Rational {
                value: ExactRational::integer(0),
                dimension: DimExponents::DIMENSIONLESS,
            })
            .unwrap();
    }
    // Three spare nodes admit part of either expansion, but not the final action.
    let before = (builder.nodes.clone(), builder.depths.clone());
    assert_eq!(
        builder.jvp(&source.into(), &[(0, direction.into())]),
        Err(PureOperatorError::NodeLimit)
    );
    assert_eq!((builder.nodes.clone(), builder.depths.clone()), before);
    assert_eq!(
        builder.vjp(&source.into(), &direction.into(), 0),
        Err(PureOperatorError::NodeLimit)
    );
    assert_eq!((builder.nodes, builder.depths), before);
}

#[test]
fn total_derivative_preserves_rate_units_shapes_and_transactional_validation() {
    let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = length.div(time).unwrap();
    let scalar = |domain, dimension| ValueType::scalar(domain, dimension).unwrap();
    let field = scalar(ScalarDomain::Complex, length).array(2).unwrap();
    let rate = scalar(ScalarDomain::Complex, speed).array(2).unwrap();
    let formals = [
        scalar(ScalarDomain::Real, time),
        field.clone(),
        rate.clone(),
        field,
    ]
    .map(|ty| PureValueClass::from_value_type(ty).unwrap());
    let source = whole(&formals, 1);
    let valid = whole(&formals, 2);
    let wrong_units = whole(&formals, 3);
    let result = PureValueClass::from_value_type(rate.clone()).unwrap();
    let mut builder = CalculusBuilder::new(formals.clone(), result).unwrap();
    let before = (builder.nodes.clone(), builder.depths.clone());
    for (coordinate, rates) in [
        (0, vec![(1, wrong_units)]),
        (1, vec![(1, valid.clone())]),
        (0, vec![(1, valid.clone()), (1, valid.clone())]),
        (0, vec![(0, valid.clone())]),
    ] {
        assert!(
            builder
                .total_derivative(&source, coordinate, &rates)
                .is_err()
        );
        assert_eq!((builder.nodes.clone(), builder.depths.clone()), before);
    }
    let derivative = builder.total_derivative(&source, 0, &[(1, valid)]).unwrap();
    let definition = builder.finish(derivative).unwrap();
    let inputs = formals
        .iter()
        .map(|class| ExpressionType::<()>::new(class.exact_type().unwrap().clone(), None))
        .collect::<Vec<_>>();
    assert_eq!(
        definition
            .instantiate(&inputs)
            .unwrap()
            .result_type()
            .value_type,
        rate
    );
}
