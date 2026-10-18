//! Independent spatial polynomial/rational witnesses for finite-map calculus.
use super::*;
use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_core::{FiniteBasis, ValueType};
use eqiora_schema::kernel::FiniteUnaryOperation;

fn program(n: u32, inverse: bool) -> ComponentScalarization {
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let basis = FiniteBasis::new(Id::new(), n).unwrap();
    let ty = ValueType::linear_map(basis, basis, ScalarDomain::Real, length).unwrap();
    let mut builder = ExprDagBuilder::new();
    let a = builder.symbol(field).unwrap();
    let value = if inverse {
        let inverse = builder
            .finite_unary(FiniteUnaryOperation::Inverse, a)
            .unwrap();
        builder
            .finite_unary(FiniteUnaryOperation::MatrixTrace, inverse)
            .unwrap()
    } else {
        builder
            .finite_unary(FiniteUnaryOperation::Determinant, a)
            .unwrap()
    };
    let gradient = builder.gradient(value).unwrap();
    let hessian = builder.gradient(gradient).unwrap();
    let typed = TypedResidual::infer(
        builder.finish([value, gradient, hessian]).unwrap(),
        Some(support.clone()),
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(ExpressionType::new(ty.clone(), Some(support.clone()))),
    )
    .unwrap();
    ComponentScalarization::lower_spatial_selected(&typed, &[value, gradient, hessian], None)
        .unwrap()
}

fn value(c: &ScalarSymbolCoordinate, singular: bool) -> f64 {
    let component = c.component_index();
    let (r, s) = (component[0], component[1]);
    match c.spatial_derivative_axes() {
        [] => match (r, s) {
            (0, 0) => {
                if singular {
                    1.
                } else {
                    2.
                }
            }
            (0, 1) => {
                if singular {
                    2.
                } else {
                    1.
                }
            }
            (1, 0) => {
                if singular {
                    2.
                } else {
                    3.
                }
            }
            (1, 1) => 4.,
            (0, 2) => 7.,
            (1, 2) => 8.,
            (2, 2) => 5.,
            _ => 0.,
        },
        [0] => match (r, s) {
            (0, 0) => 1.,
            (1, 0) => -1.,
            _ => 0.,
        },
        [1] => match (r, s) {
            (0, 1) => 2.,
            (1, 1) => 3.,
            _ => 0.,
        },
        [_, _] => 0.,
        _ => panic!("unexpected spatial order"),
    }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-11 * expected.abs().max(1.),
        "{actual} != {expected}"
    );
}

#[test]
fn finite_map_spatial_gradients_and_hessians_keep_the_same_parameter_actions() {
    for n in [1, 2, 3] {
        for inverse in [false, true] {
            let program = program(n, inverse);
            // A=[[2+x,1+2y],[3-x,4+3y]], det(A)=5+5x+5xy.
            // tr(A^-1)=(6+x+3y)/(5+5x+5xy). The third dimension,
            // when present, is an upper triangular extension with diagonal 5.
            let mut expected = if inverse {
                [6. / 5., -1., 3. / 5., 2., -9. / 5., -9. / 5., 0.]
            } else {
                [5., 5., 0., 0., 5., 5., 0.]
            };
            if n == 1 {
                expected = if inverse {
                    [0.5, -0.25, 0., 0.25, 0., 0., 0.]
                } else {
                    [2., 1., 0., 0., 0., 0., 0.]
                };
            }
            if n == 3 {
                if inverse {
                    expected[0] += 1. / 5.;
                } else {
                    expected = expected.map(|v| 5. * v);
                }
            }
            let degree = if inverse { -1. } else { f64::from(n) };
            for (row, expected) in program.rows().iter().zip(expected) {
                let inputs = row
                    .symbols()
                    .iter()
                    .map(|c| value(c, false))
                    .collect::<Vec<_>>();
                close(row.evaluate(&inputs).unwrap(), expected);
                let action = row
                    .linearize(&inputs, &vec![DifferentiationRole::Unknown; inputs.len()])
                    .unwrap();
                let mut output = [0.];
                // Under A(x,y) -> c*A(x,y), all these outputs have degree n
                // for det and -1 for the inverse, including spatial derivatives.
                action
                    .jvp(RelationTangent::Unknown(&inputs), &mut output)
                    .unwrap();
                close(output[0], degree * expected);
                let mut reverse = vec![0.; inputs.len()];
                action
                    .vjp(&[1.], RelationCotangent::Unknown(&mut reverse))
                    .unwrap();
                close(
                    reverse.iter().zip(&inputs).map(|(a, b)| a * b).sum(),
                    degree * expected,
                );
                action
                    .mixed_jvp(
                        RelationTangent::Unknown(&inputs),
                        RelationTangent::Unknown(&inputs),
                        RelationTangent::Unknown(&vec![0.; inputs.len()]),
                        &mut output,
                    )
                    .unwrap();
                close(output[0], degree * (degree - 1.) * expected);
            }
        }
    }
}

#[test]
fn singular_spatial_determinants_keep_polynomial_derivatives() {
    for n in [2, 3] {
        let program = program(n, false);
        // det([[1+x,2+2y],[2-x,4+3y]])=6x-y+5xy.
        let factor = if n == 3 { 5. } else { 1. };
        for (actual, expected) in program
            .evaluate(|c| Some(value(c, true)))
            .unwrap()
            .into_iter()
            .zip([0., 6., -1., 0., 5., 5., 0.])
        {
            close(actual, factor * expected);
        }
        assert!(
            self::program(n, true)
                .evaluate(|c| Some(value(c, true)))
                .is_err()
        );
    }
}

#[test]
fn spatial_cofactors_keep_the_existing_derivative_underflow_convention() {
    let program = program(3, false);
    let actual = program
        .evaluate(|c| {
            let component = c.component_index();
            Some(match c.spatial_derivative_axes() {
                [] if component[0] == component[1] => {
                    if component[0] == 2 {
                        1e100
                    } else {
                        1e-200
                    }
                }
                [0] if component == [2, 2] => 1.,
                _ => 0.,
            })
        })
        .unwrap();
    // det(A)=1e-300 is representable; d(det)/dx=a00*a11=1e-400
    // rounds to zero, exactly as the existing numerical cofactor evaluator.
    assert!((actual[0] / 1e-300 - 1.).abs() < 1e-12);
    assert!(actual[1..].iter().all(|value| *value == 0.));
}
