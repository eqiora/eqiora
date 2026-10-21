//! Unitful guarded components share scalar point-domain and derivative decisions.
use super::*;
use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueType};
use eqiora_schema::kernel::{ComparisonOp, ExprDagBuilder, typing::RootContract};

#[test]
fn guarded_complex_root_keeps_units_and_real_paired_actions() {
    let symbol = SymbolRef::Parameter(Id::new());
    let area = DimExponents::from_integers([0, 2, 0, 0, 0, 0, 0]).unwrap();
    let mut builder = ExprDagBuilder::new();
    let z = builder.symbol(symbol).unwrap();
    let real = builder
        .unary_math(eqiora_schema::kernel::UnaryMathFunction::Real, z)
        .unwrap();
    let zero = builder.constant(DynQuantity::new(0., area)).unwrap();
    let positive = builder.compare(ComparisonOp::Greater, real, zero).unwrap();
    let root = builder
        .unary_math(eqiora_schema::kernel::UnaryMathFunction::Sqrt, z)
        .unwrap();
    let root = builder.require(positive, root).unwrap();
    let typed = TypedResidual::<()>::infer(
        builder.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| {
            Ok::<_, ()>(ExpressionType::new(
                ValueType::scalar(ScalarDomain::Complex, area).unwrap(),
                None,
            ))
        },
    )
    .unwrap();
    let rows = ComponentScalarization::lower(&typed).unwrap();
    let action = rows
        .linearize(|coordinate| {
            Some((
                if coordinate.is_imaginary() { 4. } else { 3. },
                DifferentiationRole::Parameter,
            ))
        })
        .unwrap();
    let close = |actual: &[f64], expected: &[f64]| {
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 16. * f64::EPSILON);
        }
    };
    let mut output = [0.; 2];
    action.primal(&mut output).unwrap();
    close(&output, &[2., 1.]);
    // sqrt(3+4i)=2+i, f'=(2-i)/10, f''=(-2+11i)/500.
    action
        .jvp(RelationTangent::Parameter(&[1., 2.]), &mut output)
        .unwrap();
    close(&output, &[0.4, 0.3]);
    action
        .vjp(&[2., 3.], RelationCotangent::Parameter(&mut output))
        .unwrap();
    close(&output, &[0.1, 0.8]);
    for (index, row) in rows.rows().iter().enumerate() {
        let point = row
            .symbols()
            .iter()
            .map(|c| if c.is_imaginary() { 4. } else { 3. })
            .collect::<Vec<_>>();
        let action = row
            .linearize(&point, &vec![DifferentiationRole::Parameter; point.len()])
            .unwrap();
        let left = row
            .symbols()
            .iter()
            .map(|c| f64::from(!c.is_imaginary()))
            .collect::<Vec<_>>();
        let right = row
            .symbols()
            .iter()
            .map(|c| f64::from(c.is_imaginary()))
            .collect::<Vec<_>>();
        let mut mixed = [0.];
        action
            .mixed_jvp(
                RelationTangent::Parameter(&left),
                RelationTangent::Parameter(&right),
                RelationTangent::Parameter(&vec![0.; point.len()]),
                &mut mixed,
            )
            .unwrap();
        close(&mixed, &[[-0.022, -0.004][index]]);
    }
    assert!(rows.evaluate(|_| Some(0.)).is_err());
}

#[test]
fn guarded_vector_keeps_units_and_rejects_active_boundaries() {
    let x = SymbolRef::Field(Id::new());
    let p = SymbolRef::Parameter(Id::new());
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let mut builder = ExprDagBuilder::new();
    let field = builder.symbol(x).unwrap();
    let gain = builder.symbol(p).unwrap();
    let lower = builder.constant(DynQuantity::new(0.0, length)).unwrap();
    let upper = builder.constant(DynQuantity::new(2.0, length)).unwrap();
    let above = builder
        .compare(ComparisonOp::GreaterEqual, field, lower)
        .unwrap();
    let below = builder
        .compare(ComparisonOp::LessEqual, field, upper)
        .unwrap();
    let condition = builder.and(above, below).unwrap();
    let product = builder.mul(field, gain).unwrap();
    let double = builder.add(product, product).unwrap();
    let vector = builder.array([product, double]).unwrap();
    let root = builder.require(condition, vector).unwrap();
    let typed = TypedResidual::<()>::infer(
        builder.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |symbol| {
            Ok::<_, ()>(ExpressionType::new(
                ValueType::scalar(
                    ScalarDomain::Real,
                    if symbol == x {
                        length
                    } else {
                        DimExponents::DIMENSIONLESS
                    },
                )
                .unwrap(),
                None,
            ))
        },
    )
    .unwrap();
    let program = ComponentScalarization::lower(&typed).unwrap();
    let point = |value, role| {
        program.linearize(|coordinate| {
            Some(if coordinate.symbol() == x {
                (value, role)
            } else {
                (3.0, DifferentiationRole::Parameter)
            })
        })
    };
    let action = point(1.0, DifferentiationRole::Unknown).unwrap();
    let mut primal = [0.0; 2];
    action.primal(&mut primal).unwrap();
    assert_eq!(primal, [3.0, 6.0]);
    let mut tangent = [0.0; 2];
    action
        .jvp(
            RelationTangent::Both {
                unknown: &[1.0],
                parameter: &[2.0],
            },
            &mut tangent,
        )
        .unwrap();
    assert_eq!(tangent, [5.0, 10.0]);
    let (mut dx, mut dp) = ([0.0], [0.0]);
    action
        .vjp(
            &[2.0, -0.5],
            RelationCotangent::Both {
                unknown: &mut dx,
                parameter: &mut dp,
            },
        )
        .unwrap();
    assert_eq!(dx, [3.0]);
    assert_eq!(dp, [1.0]);
    for outside in [-1.0, 3.0] {
        assert!(point(outside, DifferentiationRole::Unknown).is_err());
        assert!(point(outside, DifferentiationRole::Frozen).is_err());
    }
    for boundary in [0.0, 2.0] {
        assert!(point(boundary, DifferentiationRole::Unknown).is_err());
        point(boundary, DifferentiationRole::Frozen).unwrap();
    }
}

fn guarded_map(inverse: bool) -> (ComponentScalarization, SymbolRef, SymbolRef) {
    use eqiora_core::FiniteBasis;
    use eqiora_schema::kernel::FiniteUnaryOperation;
    let matrix = SymbolRef::Field(Id::new());
    let gate = SymbolRef::Parameter(Id::new());
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let basis = FiniteBasis::new(Id::new(), 2).unwrap();
    let map = ValueType::linear_map(basis, basis, ScalarDomain::Real, length).unwrap();
    let mut builder = ExprDagBuilder::new();
    let a = builder.symbol(matrix).unwrap();
    let p = builder.symbol(gate).unwrap();
    let zero = builder
        .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
        .unwrap();
    let condition = builder
        .compare(ComparisonOp::GreaterEqual, p, zero)
        .unwrap();
    // Guard each map coordinate before gathering the invariant's operands.
    let a = builder.require(condition, a).unwrap();
    let det = builder
        .finite_unary(FiniteUnaryOperation::Determinant, a)
        .unwrap();
    let mut roots = vec![det];
    if inverse {
        roots.push(
            builder
                .finite_unary(FiniteUnaryOperation::Inverse, a)
                .unwrap(),
        );
    }
    let typed = TypedResidual::<()>::infer(
        builder.finish(roots).unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(ExpressionType::new(
                if symbol == matrix {
                    map.clone()
                } else {
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap()
                },
                None,
            ))
        },
    )
    .unwrap();
    (ComponentScalarization::lower(&typed).unwrap(), matrix, gate)
}

#[test]
fn guarded_unitful_maps_share_values_forward_and_reverse_actions() {
    let (program, matrix, _) = guarded_map(true);
    let flat =
        |c: &ScalarSymbolCoordinate| (2 * c.component_index()[0] + c.component_index()[1]) as usize;
    let values = [2., 1., 3., 4.];
    // det(A)=ad-bc=5 m²; inverse(A)=[[d,-b],[-c,a]]/det(A), in 1/m.
    let expected = [5., 4. / 5., -1. / 5., -3. / 5., 2. / 5.];
    let actual = program
        .evaluate(|c| {
            Some(if c.symbol() == matrix {
                values[flat(c)]
            } else {
                0.
            })
        })
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-13);
    }
    let action = program
        .linearize(|c| {
            Some(if c.symbol() == matrix {
                (values[flat(c)], DifferentiationRole::Unknown)
            } else {
                (0., DifferentiationRole::Frozen)
            })
        })
        .unwrap();
    let h = [1., 2., -1., 3.];
    let direction = action
        .unknown_coordinates()
        .iter()
        .map(|c| h[flat(c)])
        .collect::<Vec<_>>();
    let mut tangent = [0.; 5];
    action
        .jvp(RelationTangent::Unknown(&direction), &mut tangent)
        .unwrap();
    // Differentiate the explicit two-by-two polynomial/quotients above.
    for (actual, expected) in tangent
        .into_iter()
        .zip([5., -1. / 5., -1. / 5., 4. / 5., -1. / 5.])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
    let mut reverse = vec![0.; direction.len()];
    action
        .vjp(
            &[2., 3., -1., 4., -2.],
            RelationCotangent::Unknown(&mut reverse),
        )
        .unwrap();
    // Gradient of 2(ad-bc)+(3d+b-4c-2a)/(ad-bc).
    let expected = [202. / 25., -154. / 25., -73. / 25., 121. / 25.];
    for (c, actual) in action.unknown_coordinates().iter().zip(reverse) {
        assert!((actual - expected[flat(c)]).abs() < 1e-12);
    }
    // A demanded boundary is regular only when the condition is held fixed.
    assert!(
        program
            .linearize(|c| Some((
                if c.symbol() == matrix {
                    values[flat(c)]
                } else {
                    0.
                },
                DifferentiationRole::Unknown
            )))
            .is_err()
    );
    assert!(
        program
            .evaluate(|c| Some(if c.symbol() == matrix {
                values[flat(c)]
            } else {
                -1.
            }))
            .is_err()
    );
}

#[test]
fn guarded_map_domains_keep_singular_determinants_and_reject_invalid_inverses() {
    let (program, matrix, _) = guarded_map(false);
    let flat =
        |c: &ScalarSymbolCoordinate| (2 * c.component_index()[0] + c.component_index()[1]) as usize;
    let values = [1., 2., 2., 4.];
    let action = program
        .linearize(|c| {
            Some(if c.symbol() == matrix {
                (values[flat(c)], DifferentiationRole::Unknown)
            } else {
                (1., DifferentiationRole::Frozen)
            })
        })
        .unwrap();
    let h = [1., 2., -1., 3.];
    let direction = action
        .unknown_coordinates()
        .iter()
        .map(|c| h[flat(c)])
        .collect::<Vec<_>>();
    let mut value = [0.];
    action.primal(&mut value).unwrap();
    assert_eq!(value, [0.]);
    action
        .jvp(RelationTangent::Unknown(&direction), &mut value)
        .unwrap();
    assert!((value[0] - 5.).abs() < 1e-12);
    let (program, matrix, _) = guarded_map(true);
    for values in [values, [1., 0., 0., 1e-18]] {
        for role in [DifferentiationRole::Unknown, DifferentiationRole::Frozen] {
            assert!(
                program
                    .linearize(|c| Some((
                        if c.symbol() == matrix {
                            values[flat(c)]
                        } else {
                            1.
                        },
                        role
                    )))
                    .is_err()
            );
        }
    }
}

#[test]
fn pure_guarded_polynomial_retains_values_products_and_validity() {
    use crate::{CalculusBuilder, CalculusNode, ExactRational, PureValueClass};
    let unit = DimExponents::DIMENSIONLESS;
    let class = PureValueClass::invariant_scalar()
        .with_dimension(unit)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap();
    let mut c = CalculusBuilder::new([class.clone()], class).unwrap();
    let x = c
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let zero = c
        .push(CalculusNode::Rational {
            value: ExactRational::integer(0),
            dimension: unit,
        })
        .unwrap();
    let three = c
        .push(CalculusNode::Rational {
            value: ExactRational::integer(3),
            dimension: unit,
        })
        .unwrap();
    let lower = c
        .push(CalculusNode::Compare(ComparisonOp::GreaterEqual, x, zero))
        .unwrap();
    let upper = c
        .push(CalculusNode::Compare(ComparisonOp::LessEqual, x, three))
        .unwrap();
    let valid = c.push(CalculusNode::And(lower, upper)).unwrap();
    let square = c.push(CalculusNode::Mul(x, x)).unwrap();
    let value = c.push(CalculusNode::Add(three, square)).unwrap();
    let root = c
        .push(CalculusNode::Require {
            condition: valid,
            value,
        })
        .unwrap();
    let definition = c.finish(root).unwrap();
    let mut b = ExprDagBuilder::new();
    let x = b.symbol(SymbolRef::Parameter(Id::new())).unwrap();
    let root = b.pure_operator(&definition, [x]).unwrap();
    let typed = TypedResidual::<()>::infer(
        b.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| Ok::<_, ()>(ExpressionType::scalar(unit, None)),
    )
    .unwrap();
    let program = ComponentScalarization::lower(&typed).unwrap();
    assert_eq!(program.evaluate(|_| Some(2.)).unwrap(), [7.]);
    let linear = program
        .linearize(|_| Some((2., DifferentiationRole::Parameter)))
        .unwrap();
    let mut out = [0.];
    linear
        .jvp(RelationTangent::Parameter(&[3.]), &mut out)
        .unwrap();
    assert_eq!(out, [12.]);
    linear
        .vjp(&[5.], RelationCotangent::Parameter(&mut out))
        .unwrap();
    assert_eq!(out, [20.]);
    // d²(3+x²)[3,5] = 2*3*5, independently of x.
    program.rows()[0]
        .linearize(&[2.], &[DifferentiationRole::Parameter])
        .unwrap()
        .mixed_jvp(
            RelationTangent::Parameter(&[3.]),
            RelationTangent::Parameter(&[5.]),
            RelationTangent::Parameter(&[0.]),
            &mut out,
        )
        .unwrap();
    assert_eq!(out, [30.]);
    for boundary in [0., 3.] {
        assert!(program.evaluate(|_| Some(boundary)).is_ok());
        assert!(
            program
                .linearize(|_| Some((boundary, DifferentiationRole::Parameter)))
                .is_err()
        );
        assert!(
            program
                .linearize(|_| Some((boundary, DifferentiationRole::Frozen)))
                .is_ok()
        );
    }
    for outside in [-1., 4.] {
        assert!(program.evaluate(|_| Some(outside)).is_err());
    }
}

#[test]
fn pure_piecewise_math_uses_lazy_boolean_demand_and_active_derivatives() {
    use crate::{CalculusBuilder, CalculusNode, ExactRational, PureValueClass};
    use eqiora_schema::kernel::UnaryMathFunction;
    let unit = DimExponents::DIMENSIONLESS;
    let class = PureValueClass::invariant_scalar()
        .with_dimension(unit)
        .with_scalar_domain(ScalarDomain::Real)
        .unwrap();
    let mut c = CalculusBuilder::new([class.clone()], class).unwrap();
    let x = c
        .push(CalculusNode::FormalComponent {
            formal: 0,
            axes: Box::new([]),
        })
        .unwrap();
    let zero = c
        .push(CalculusNode::Rational {
            value: ExactRational::integer(0),
            dimension: unit,
        })
        .unwrap();
    let positive = c
        .push(CalculusNode::Compare(ComparisonOp::Greater, x, zero))
        .unwrap();
    let false_value = c.push(CalculusNode::Boolean(false)).unwrap();
    let true_value = c.push(CalculusNode::Not(false_value)).unwrap();
    let condition = c.push(CalculusNode::Or(false_value, positive)).unwrap();
    let condition = c.push(CalculusNode::And(true_value, condition)).unwrap();
    let negative = c.push(CalculusNode::Neg(x)).unwrap();
    let positive_root = c
        .push(CalculusNode::UnaryMath(UnaryMathFunction::Sqrt, x))
        .unwrap();
    let negative_root = c
        .push(CalculusNode::UnaryMath(UnaryMathFunction::Sqrt, negative))
        .unwrap();
    let root = c
        .push(CalculusNode::Select {
            condition,
            then_value: positive_root,
            else_value: negative_root,
        })
        .unwrap();
    let definition = c.finish(root).unwrap();
    let mut b = ExprDagBuilder::new();
    let x = b.symbol(SymbolRef::Parameter(Id::new())).unwrap();
    let root = b.pure_operator(&definition, [x]).unwrap();
    let zero = b.constant(DynQuantity::new(0., unit)).unwrap();
    let condition = b.compare(ComparisonOp::Greater, x, zero).unwrap();
    let false_value = b
        .constant(eqiora_core::ValueLiteral::boolean(false))
        .unwrap();
    let true_value = b.not(false_value).unwrap();
    let condition = b.or(false_value, condition).unwrap();
    let condition = b.require(true_value, condition).unwrap();
    let condition = b.select(true_value, condition, false_value).unwrap();
    let positive_root = b.unary_math(UnaryMathFunction::Sqrt, x).unwrap();
    let negative = b.neg(x).unwrap();
    let negative_root = b.unary_math(UnaryMathFunction::Sqrt, negative).unwrap();
    let direct = b.select(condition, positive_root, negative_root).unwrap();
    let typed = TypedResidual::<()>::infer(
        b.finish([root, direct]).unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| Ok::<_, ()>(ExpressionType::scalar(unit, None)),
    )
    .unwrap();
    let program = ComponentScalarization::lower(&typed).unwrap();
    for (x, slope) in [(4., 0.25), (-4., -0.25)] {
        // The unselected square root has a negative argument at either point.
        assert_eq!(program.evaluate(|_| Some(x)).unwrap(), [2., 2.]);
        let action = program
            .linearize(|_| Some((x, DifferentiationRole::Parameter)))
            .unwrap();
        let mut out = [0.; 2];
        action
            .jvp(RelationTangent::Parameter(&[1.]), &mut out)
            .unwrap();
        assert_eq!(out, [slope, slope]);
        let mut cotangent = [0.];
        action
            .vjp(&[1., 1.], RelationCotangent::Parameter(&mut cotangent))
            .unwrap();
        assert_eq!(cotangent, [2. * slope]);
    }
    assert_eq!(program.evaluate(|_| Some(0.)).unwrap(), [0., 0.]);
    assert!(
        program
            .linearize(|_| Some((0., DifferentiationRole::Parameter)))
            .is_err()
    );
}

#[test]
fn derivative_permissions_retain_every_complex_channel_even_for_constant_output() {
    let symbol = SymbolRef::Parameter(Id::new());
    let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
        .unwrap()
        .array(2)
        .unwrap();
    use crate::{CalculusBuilder, CalculusNode, ExactRational, PureValueClass};
    use eqiora_schema::kernel::{PropertyDerivatives, PropertyMeaning, PropertyRelease};
    let result = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let mut calculus = CalculusBuilder::new(
        [PureValueClass::from_value_type(ty.clone()).unwrap()],
        PureValueClass::from_value_type(result).unwrap(),
    )
    .unwrap();
    let constant = calculus
        .push(CalculusNode::Rational {
            value: ExactRational::integer(9),
            dimension: DimExponents::DIMENSIONLESS,
        })
        .unwrap();
    let release = PropertyRelease::new(
        ("org.example.Response".into(), "org.example.Measured".into()),
        vec!["x".into()],
        Some("single".into()),
        PropertyDerivatives::ValueOnly,
        ("org.example.manufactured".into(), "spdx.CC0_1_0".into()),
        PropertyMeaning::Analytic(calculus.finish(constant).unwrap()),
    )
    .unwrap();
    let mut builder = ExprDagBuilder::new();
    let input = builder.symbol(symbol).unwrap();
    let root = builder.property(release, [input]).unwrap();
    let typed = TypedResidual::<()>::infer(
        builder.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| Ok::<_, ()>(ExpressionType::new(ty.clone(), None)),
    )
    .unwrap();
    let rows = ComponentScalarization::lower(&typed).unwrap();
    let coordinates = rows.rows()[0]
        .symbols()
        .iter()
        .map(|coordinate| {
            (
                coordinate.component_index().to_vec(),
                coordinate.is_imaginary(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        coordinates,
        [
            (vec![0], false),
            (vec![0], true),
            (vec![1], false),
            (vec![1], true)
        ]
    );
    let frozen = rows
        .linearize(|_| Some((2., DifferentiationRole::Frozen)))
        .unwrap();
    let mut value = [0.];
    frozen.primal(&mut value).unwrap();
    assert_eq!(value, [9.]);
    for (component, imaginary) in coordinates {
        let error = rows
            .linearize(|coordinate| {
                Some((
                    2.,
                    if coordinate.component_index() == component
                        && coordinate.is_imaginary() == imaginary
                    {
                        DifferentiationRole::Parameter
                    } else {
                        DifferentiationRole::Frozen
                    },
                ))
            })
            .unwrap_err();
        assert!(error.message().contains("value-only contract"), "{error:?}");
    }
}
