use eqiora_core::{DimExponents, ValueShape};

use super::*;

fn volume_tensor(domain: &str) -> ExpressionType<&str> {
    ExpressionType::shaped(
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2, 2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(SpatialSupport::Volume {
            domain,
            dimensions: 2,
        }),
    )
    .unwrap()
}

fn volume_scalar(domain: &str) -> ExpressionType<&str> {
    ExpressionType::scalar(
        DimExponents::DIMENSIONLESS,
        Some(SpatialSupport::Volume {
            domain,
            dimensions: 2,
        }),
    )
}

fn volume_vector(domain: &str, dimension: DimExponents) -> ExpressionType<&str> {
    ExpressionType::shaped(
        dimension,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(SpatialSupport::Volume {
            domain,
            dimensions: 2,
        }),
    )
    .unwrap()
}

#[test]
fn canonical_rational_parts_cover_the_complete_unsigned_denominator_wire() {
    let denominator = (i64::MAX as u64) + 2;
    let value = ExactRational::from_canonical_parts(1, denominator).unwrap();
    assert_eq!(value.numerator(), 1);
    assert_eq!(value.denominator(), denominator);
    assert_eq!(
        ExactRational::from_canonical_parts(2, 4),
        Err(PureOperatorError::InvalidRational)
    );
}

#[test]
fn polynomial_operators_preserve_common_scalar_domain_and_exact_component_roles() {
    use eqiora_core::{ScalarDomain, ValueType};
    let complex = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
        .expect("checked scalar type");
    let mut tensor = volume_tensor("body");
    tensor.value_type = tensor
        .value_type
        .with_common_scalar_domain(&complex)
        .unwrap();
    let symmetric = PureOperatorDefinition::symmetric_part().unwrap();
    assert_eq!(
        symmetric
            .instantiate(std::slice::from_ref(&tensor))
            .unwrap()
            .result_type(),
        &tensor
    );
    let mut left = volume_vector("body", DimExponents::DIMENSIONLESS);
    let right = left.clone();
    left.value_type = left.value_type.with_common_scalar_domain(&complex).unwrap();
    let dyadic = PureOperatorDefinition::dyadic_product().unwrap();
    let application = dyadic.instantiate(&[left.clone(), right]).unwrap();
    assert_eq!(application.result_type(), &tensor);
    left.value_type = complex.array(2).unwrap();
    assert_eq!(
        dyadic.instantiate(&[left.clone(), left]).unwrap_err(),
        PureOperatorError::FormalTypeMismatch
    );
}

#[test]
fn standard_definitions_derive_their_exact_result_types() {
    let tensor = volume_tensor("body");
    let symmetric = PureOperatorDefinition::symmetric_part().unwrap();
    let symmetric_application = symmetric
        .instantiate(std::slice::from_ref(&tensor))
        .unwrap();
    assert_eq!(symmetric_application.result_type(), &tensor);

    let isotropic = PureOperatorDefinition::isotropic_lift().unwrap();
    let isotropic_application = isotropic.instantiate(&[volume_scalar("body")]).unwrap();
    assert_eq!(
        isotropic_application.result_type().shape().extents().len(),
        2
    );
    assert_eq!(
        isotropic_application.result_type().shape().extents()[0].get(),
        2
    );
    assert_eq!(
        isotropic_application.result_type().shape().extents()[1].get(),
        2
    );
}

#[test]
fn definition_identity_excludes_names_but_includes_exact_body() {
    let first = PureOperatorDefinition::symmetric_part().unwrap();
    let second = PureOperatorDefinition::symmetric_part().unwrap();
    let isotropic = PureOperatorDefinition::isotropic_lift().unwrap();
    assert_eq!(first.canonical_bytes(), second.canonical_bytes());
    assert_eq!(first.digest(), second.digest());
    assert_ne!(first.digest(), isotropic.digest());
    assert!(
        first
            .canonical_bytes()
            .starts_with(b"eqiora.pure-operator-definition/v5\0")
    );
    assert_ne!(
        first.digest().to_string(),
        "2b1d8bbaf99a2c1b1fd2d14dc384e6ce2624ce54cad65e337fbe7cdc01b0e99a"
    );
    assert_ne!(
        isotropic.digest().to_string(),
        "fe648a6a0f5b9bf2460389e3232822747d5ec85cceb38fcf8fdea977921c63f6"
    );
}

#[test]
fn dyadic_product_derives_shape_support_and_product_dimension() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
    let force = DimExponents::from_integers([1, 1, -2, 0, 0, 0, 0]).expect("bounded dimension");
    let definition = PureOperatorDefinition::dyadic_product().unwrap();
    assert_eq!(definition.formals().len(), 2);
    assert_eq!(
        definition.dimension_monomial().exponents(),
        &[ExactRational::integer(1), ExactRational::integer(1)]
    );
    let application = definition
        .instantiate(&[volume_vector("body", length), volume_vector("body", force)])
        .unwrap();
    let result = application.result_type();
    assert_eq!(result.shape(), &ValueShape::new([2, 2]).unwrap());
    assert_eq!(result.frame(), ValueFrame::SpatialCartesian);
    assert_eq!(
        result.support,
        Some(SpatialSupport::Volume {
            domain: "body",
            dimensions: 2,
        })
    );
    assert_eq!(
        result.dimension(),
        DimExponents::from_integers([1, 2, -2, 0, 0, 0, 0]).expect("bounded dimension")
    );
    assert_ne!(
        definition.digest(),
        PureOperatorDefinition::symmetric_part().unwrap().digest()
    );
    assert_ne!(
        definition.digest().to_string(),
        "293e3645a9a7a74a15caaad0214fc5f1e59111bb71bf89a28e6471ae80f6775a"
    );
}

#[test]
fn dyadic_product_requires_one_exact_volume_and_checked_si_dimension() {
    let definition = PureOperatorDefinition::dyadic_product().unwrap();
    let dimensionless = DimExponents::DIMENSIONLESS;
    assert!(matches!(
        definition.instantiate(&[
            volume_vector("left", dimensionless),
            volume_vector("right", dimensionless),
        ]),
        Err(PureOperatorError::CommonVolumeMismatch)
    ));

    let large =
        DimExponents::from_integers([0, i32::MAX, 0, 0, 0, 0, 0]).expect("bounded dimension");
    assert!(matches!(
        definition.instantiate(&[volume_vector("body", large), volume_vector("body", large),]),
        Err(PureOperatorError::ResultDimensionOverflow)
    ));
}

#[test]
fn symbolic_dimension_monomials_are_bounded() {
    let scalar = PureValueClass::invariant_scalar();
    let mut builder = CalculusBuilder::new([scalar], scalar).unwrap();
    let mut body = builder
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::default(),
        })
        .unwrap();
    for _ in 0..7 {
        body = builder.push(CalculusNode::Mul(body, body)).unwrap();
    }
    let overflow = builder.push(CalculusNode::Mul(body, body)).unwrap();
    assert_eq!(
        builder.finish(overflow),
        Err(PureOperatorError::FormalExponentLimit)
    );
}
