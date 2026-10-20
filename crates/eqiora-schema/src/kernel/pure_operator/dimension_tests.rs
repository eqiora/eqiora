use super::*;
use eqiora_core::{ScalarDomain, ValueType};

fn temperature() -> DimExponents {
    DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap()
}
fn conductivity_dimension() -> DimExponents {
    DimExponents::from_integers([1, 1, -3, 0, -1, 0, 0]).unwrap()
}
fn constrained(dimension: DimExponents) -> PureValueClass {
    PureValueClass::invariant_scalar().with_dimension(dimension)
}
fn formal(builder: &mut CalculusBuilder, slot: u16) -> CalculusNodeId {
    builder
        .push(CalculusNode::FormalComponent {
            formal: slot,
            axes: Box::new([]),
        })
        .unwrap()
}
fn conductivity() -> PureOperatorDefinition {
    let mut builder = CalculusBuilder::new(
        [
            constrained(temperature()),
            constrained(conductivity_dimension()),
            constrained(temperature().pow(-1, 1).unwrap()),
        ],
        constrained(conductivity_dimension()),
    )
    .unwrap();
    let t = formal(&mut builder, 0);
    let k = formal(&mut builder, 1);
    let a = formal(&mut builder, 2);
    let one = builder
        .push(CalculusNode::Rational {
            value: ExactRational::integer(1),
            dimension: DimExponents::DIMENSIONLESS,
        })
        .unwrap();
    let at = builder.push(CalculusNode::Mul(a, t)).unwrap();
    let a2 = builder.push(CalculusNode::Mul(a, a)).unwrap();
    let t2 = builder.push(CalculusNode::Mul(t, t)).unwrap();
    let square = builder.push(CalculusNode::Mul(a2, t2)).unwrap();
    let linear = builder.push(CalculusNode::Add(one, at)).unwrap();
    let quadratic = builder.push(CalculusNode::Add(linear, square)).unwrap();
    let root = builder.push(CalculusNode::Mul(k, quadratic)).unwrap();
    builder.finish(root).unwrap()
}
fn arguments() -> Vec<ExpressionType<u32>> {
    [
        temperature(),
        conductivity_dimension(),
        temperature().pow(-1, 1).unwrap(),
    ]
    .map(|dimension| {
        ExpressionType::new(
            ValueType::scalar(ScalarDomain::Real, dimension).expect("checked scalar type"),
            None,
        )
    })
    .to_vec()
}

#[test]
fn concrete_conductivity_dimensions_admit_scalar_calls_without_a_volume() {
    let definition = conductivity();
    let args = arguments();
    let application = definition.instantiate(&args).unwrap();
    assert_eq!(
        application.result_type().dimension(),
        conductivity_dimension()
    );
    assert_eq!(application.result_type().support, None);
    let mut wrong = args.clone();
    wrong[2] = ExpressionType::new(
        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type"),
        None,
    );
    assert_eq!(
        definition.instantiate(&wrong).unwrap_err(),
        PureOperatorError::FormalTypeMismatch
    );
    let mut supported = args;
    supported[0].support = Some(SpatialSupport::Volume {
        domain: 3,
        dimensions: 2,
    });
    assert_eq!(
        definition
            .instantiate(&supported)
            .unwrap()
            .result_type()
            .support,
        supported[0].support
    );
    supported[1].support = Some(SpatialSupport::Volume {
        domain: 4,
        dimensions: 2,
    });
    assert_eq!(
        definition.instantiate(&supported).unwrap_err(),
        PureOperatorError::CommonSupportMismatch
    );
}

#[test]
fn polynomial_addition_requires_provable_dimensions_and_pins_result_identity() {
    let scalar = PureValueClass::invariant_scalar();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let stiffness = DimExponents::from_integers([1, 0, -2, 0, 0, 0, 0]).unwrap();
    let force = stiffness.mul(length).unwrap();
    let cubic = stiffness.div(length.pow(2, 1).unwrap()).unwrap();
    for (formals, result, valid) in [
        (
            vec![
                constrained(length),
                constrained(stiffness),
                constrained(cubic),
            ],
            constrained(force),
            true,
        ),
        (vec![scalar; 3], scalar, false),
        (
            vec![
                constrained(length),
                constrained(stiffness),
                constrained(stiffness),
            ],
            constrained(force),
            false,
        ),
        (
            vec![
                constrained(length),
                constrained(stiffness),
                constrained(cubic),
            ],
            constrained(length),
            false,
        ),
    ] {
        let mut builder = CalculusBuilder::new(formals, result).unwrap();
        let x = formal(&mut builder, 0);
        let k = formal(&mut builder, 1);
        let c = formal(&mut builder, 2);
        let linear = builder.push(CalculusNode::Mul(k, x)).unwrap();
        let x2 = builder.push(CalculusNode::Mul(x, x)).unwrap();
        let x3 = builder.push(CalculusNode::Mul(x2, x)).unwrap();
        let nonlinear = builder.push(CalculusNode::Mul(c, x3)).unwrap();
        let root = builder.push(CalculusNode::Add(linear, nonlinear)).unwrap();
        assert_eq!(builder.finish(root).is_ok(), valid);
    }
    let identity = |class| {
        let mut builder = CalculusBuilder::new([class], class).unwrap();
        let root = formal(&mut builder, 0);
        builder.finish(root).unwrap()
    };
    assert_ne!(
        identity(scalar).digest(),
        identity(constrained(length)).digest()
    );
    assert_ne!(
        identity(constrained(length)).digest(),
        identity(constrained(force)).digest()
    );
}

#[test]
fn scalar_composition_preserves_order_and_rejects_mistyped_calls() {
    let definition = conductivity();
    let mut outer = CalculusBuilder::new(
        definition.formals().iter().copied(),
        definition.result_rule(),
    )
    .unwrap();
    let args = (0..3)
        .map(|slot| formal(&mut outer, slot))
        .collect::<Vec<_>>();
    let root = outer.apply_scalar(&definition, &args).unwrap();
    let composed = outer.finish(root).unwrap();
    assert_eq!(composed.canonical_bytes(), definition.canonical_bytes());
    let mut wrong = CalculusBuilder::new(
        [constrained(DimExponents::DIMENSIONLESS); 3],
        PureValueClass::invariant_scalar(),
    )
    .unwrap();
    let args = (0..3)
        .map(|slot| formal(&mut wrong, slot))
        .collect::<Vec<_>>();
    assert!(wrong.apply_scalar(&definition, &args).is_err());
    assert_eq!(wrong.finish(args[0]).unwrap().nodes().len(), 3);
}
