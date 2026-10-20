//! Numerical derivative products obey the same declared property contract.
use eqiora_core::{DimExponents, DynQuantity, Id, ScalarDomain, ValueType};
use eqiora_ir::{
    ComponentScalarization, DifferentiationRole, LinearizedRelation, ScalarOperatorIr,
};
use eqiora_schema::kernel::{
    ComparisonOp, ExprDag, ExprDagBuilder, PropertyDerivatives, PropertyMeaning, PropertyRelease,
    SymbolRef,
    pure_operator::{CalculusBuilder, CalculusNode, PureValueClass},
    typing::{ExpressionType, RootContract, TypedResidual},
};

fn fixture(profile: PropertyDerivatives, inactive: bool) -> (TypedResidual<()>, SymbolRef) {
    fixture_with_power(profile, inactive, true)
}

fn fixture_with_power(
    profile: PropertyDerivatives,
    inactive: bool,
    squared: bool,
) -> (TypedResidual<()>, SymbolRef) {
    let class = PureValueClass::invariant_scalar()
        .with_dimension(DimExponents::DIMENSIONLESS)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap();
    let mut calculus = CalculusBuilder::new([class], class).unwrap();
    let x = calculus
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let value = if squared {
        calculus.push(CalculusNode::Mul(x, x)).unwrap()
    } else {
        x
    };
    let release = PropertyRelease::new(
        ("example::Response".into(), "example::Square".into()),
        vec!["value".into()],
        None,
        profile,
        ("org.example.manufactured".into(), "spdx.CC0_1_0".into()),
        PropertyMeaning::Analytic(calculus.finish(value).unwrap()),
    )
    .unwrap();
    let input = SymbolRef::Parameter(Id::new());
    let other = SymbolRef::Parameter(Id::new());
    let mut dag = ExprDagBuilder::new();
    let x = dag.symbol(input).unwrap();
    let y = dag.symbol(other).unwrap();
    let material = dag.property(release, [x]).unwrap();
    let root = if inactive {
        let zero = dag
            .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
            .unwrap();
        let positive = dag.compare(ComparisonOp::Greater, x, zero).unwrap();
        dag.select(positive, y, material).unwrap()
    } else {
        dag.mul(material, y).unwrap()
    };
    (type_scalar(dag.finish([root]).unwrap()), input)
}

fn type_scalar(expression: ExprDag) -> TypedResidual<()> {
    TypedResidual::infer(
        expression,
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| {
            Ok::<_, ()>(ExpressionType::new(
                ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
                None,
            ))
        },
    )
    .unwrap()
}

#[test]
fn value_only_rejects_active_input_products_in_both_scalar_projections() {
    for input_value in [0., 2.] {
        for role in [DifferentiationRole::Unknown, DifferentiationRole::Parameter] {
            let (typed, input) = fixture(PropertyDerivatives::ValueOnly, false);
            let scalar = ScalarOperatorIr::lower_typed_scalar(&typed).unwrap();
            let point = scalar
                .symbols()
                .iter()
                .map(|s| if *s == input { input_value } else { 3. })
                .collect::<Vec<_>>();
            let roles = scalar
                .symbols()
                .iter()
                .map(|s| {
                    if *s == input {
                        role
                    } else {
                        DifferentiationRole::Frozen
                    }
                })
                .collect::<Vec<_>>();
            assert!(
                scalar
                    .linearize(&point, &roles)
                    .unwrap_err()
                    .message()
                    .contains("value-only contract"),
                "scalar projection lost ValueOnly"
            );
            let components = ComponentScalarization::lower(&typed).unwrap();
            assert!(
                components
                    .linearize(|coordinate| Some(if coordinate.symbol() == input {
                        (input_value, role)
                    } else {
                        (3., DifferentiationRole::Frozen)
                    }))
                    .unwrap_err()
                    .message()
                    .contains("value-only contract"),
                "component projection lost ValueOnly"
            );
        }
    }
}

#[test]
fn frozen_inputs_and_inactive_branches_do_not_request_property_derivatives() {
    for inactive in [false, true] {
        let (typed, input) = fixture(PropertyDerivatives::ValueOnly, inactive);
        let scalar = ScalarOperatorIr::lower_typed_scalar(&typed).unwrap();
        let point = scalar
            .symbols()
            .iter()
            .map(|s| if *s == input { 2. } else { 3. })
            .collect::<Vec<_>>();
        let roles = scalar
            .symbols()
            .iter()
            .map(|s| {
                if *s != input {
                    DifferentiationRole::Unknown
                } else if inactive {
                    DifferentiationRole::Parameter
                } else {
                    DifferentiationRole::Frozen
                }
            })
            .collect::<Vec<_>>();
        let scalar_action = scalar.linearize(&point, &roles).unwrap();
        check_unrelated_derivative(&scalar_action, inactive);
        let components = ComponentScalarization::lower(&typed).unwrap();
        let action = components
            .linearize(|coordinate| {
                Some(if coordinate.symbol() == input {
                    (
                        2.,
                        if inactive {
                            DifferentiationRole::Parameter
                        } else {
                            DifferentiationRole::Frozen
                        },
                    )
                } else {
                    (3., DifferentiationRole::Unknown)
                })
            })
            .unwrap();
        check_unrelated_derivative(&action, inactive);
    }
}

fn check_unrelated_derivative(action: &impl LinearizedRelation<f64>, inactive: bool) {
    let mut primal = [0.];
    action.primal(&mut primal).unwrap();
    assert_eq!(primal, [if inactive { 3. } else { 12. }]);
    let mut tangent = [0.];
    action
        .jvp(eqiora_ir::RelationTangent::Unknown(&[1.]), &mut tangent)
        .unwrap();
    assert_eq!(tangent, [if inactive { 1. } else { 4. }]);
    action
        .vjp(&[2.], eqiora_ir::RelationCotangent::Unknown(&mut tangent))
        .unwrap();
    assert_eq!(tangent, [if inactive { 2. } else { 8. }]);
}

#[test]
fn declared_first_partials_keep_the_independently_known_material_products() {
    let (typed, input) = fixture(PropertyDerivatives::FirstPartials, false);
    let scalar = ScalarOperatorIr::lower_typed_scalar(&typed).unwrap();
    let point = scalar
        .symbols()
        .iter()
        .map(|s| if *s == input { 2. } else { 3. })
        .collect::<Vec<_>>();
    let roles = scalar
        .symbols()
        .iter()
        .map(|s| {
            if *s == input {
                DifferentiationRole::Parameter
            } else {
                DifferentiationRole::Frozen
            }
        })
        .collect::<Vec<_>>();
    let action = scalar.linearize(&point, &roles).unwrap();
    let mut output = [0.];
    action
        .jvp(eqiora_ir::RelationTangent::Parameter(&[1.]), &mut output)
        .unwrap();
    assert_eq!(output, [12.]); // d(x²y)/dx=2xy at (2,3).
    action
        .vjp(&[2.], eqiora_ir::RelationCotangent::Parameter(&mut output))
        .unwrap();
    assert_eq!(output, [24.]);
}

#[test]
fn affine_coefficient_directions_cannot_bypass_value_only_contracts() {
    let (typed, input) = fixture(PropertyDerivatives::ValueOnly, false);
    let components = ComponentScalarization::lower(&typed).unwrap();
    let row = &components.rows()[0];
    let selected = row
        .symbols()
        .iter()
        .filter(|coordinate| coordinate.symbol() != input)
        .cloned()
        .collect::<Vec<_>>();
    let coefficient = row
        .symbols()
        .iter()
        .find(|coordinate| coordinate.symbol() == input)
        .unwrap();
    assert!(row.affine_in(&selected).unwrap());
    for value in [0., 2.] {
        // x²*y remains affine in y when x is held fixed.
        assert!(
            row.bind_affine(&selected, &[(coefficient.clone(), value)])
                .is_ok()
        );
        assert!(
            row.bind_affine_direction(&selected, &[(coefficient.clone(), value, None)])
                .is_ok()
        );
        // Some(0) still requests a derivative; x=0 must not hide that request.
        for tangent in [0., 1.] {
            assert!(
                row.bind_affine_direction(
                    &selected,
                    &[(coefficient.clone(), value, Some(tangent))],
                )
                .is_err(),
                "a bound material input silently admitted a derivative product"
            );
        }
    }
}

#[test]
fn an_affine_material_law_still_requires_derivative_permission() {
    for profile in [
        PropertyDerivatives::ValueOnly,
        PropertyDerivatives::FirstPartials,
    ] {
        let (typed, input) = fixture_with_power(profile, false, false);
        let components = ComponentScalarization::lower(&typed).unwrap();
        let row = &components.rows()[0];
        let selected = row
            .symbols()
            .iter()
            .filter(|coordinate| coordinate.symbol() == input)
            .cloned()
            .collect::<Vec<_>>();
        let bindings = row
            .symbols()
            .iter()
            .filter(|coordinate| coordinate.symbol() != input)
            .map(|coordinate| (coordinate.clone(), 3.))
            .collect::<Vec<_>>();
        let permitted = profile == PropertyDerivatives::FirstPartials;
        assert_eq!(row.affine_in(&selected).unwrap(), permitted);
        assert_eq!(row.bind_affine(&selected, &bindings).is_ok(), permitted);
    }
}

#[test]
fn a_sampled_zero_coefficient_cannot_hide_an_active_property_argument() {
    let input = SymbolRef::Parameter(Id::new());
    let coefficient = SymbolRef::Parameter(Id::new());
    let mut dag = ExprDagBuilder::new();
    let x = dag.symbol(input).unwrap();
    let p = dag.symbol(coefficient).unwrap();
    let product = dag.mul(p, x).unwrap();
    let root = dag.value_only(product, product).unwrap();
    let typed = type_scalar(dag.finish([root]).unwrap());
    let components = ComponentScalarization::lower(&typed).unwrap();
    let row = &components.rows()[0];
    let selected = row
        .symbols()
        .iter()
        .filter(|c| c.symbol() == input)
        .cloned()
        .collect::<Vec<_>>();
    let parameter = row
        .symbols()
        .iter()
        .find(|c| c.symbol() == coefficient)
        .unwrap();
    for value in [0., 2.] {
        assert!(
            row.bind_affine(&selected, &[(parameter.clone(), value)])
                .is_err()
        );
        assert!(
            row.bind_affine(
                &[],
                &[(selected[0].clone(), 3.), (parameter.clone(), value),]
            )
            .is_ok()
        );
    }
}
