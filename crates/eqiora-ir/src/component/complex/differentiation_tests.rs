//! Real-pairing products derived directly in x/y coordinates, without holomorphy.
use super::*;
use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
use eqiora_core::{DimExponents, Id, ValueType, entity::kinds};
use eqiora_schema::kernel::{ExprDagBuilder, typing::RootContract};

#[test]
fn complex_and_nonholomorphic_products_obey_the_real_pairing() {
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let parameter = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let mut dag = ExprDagBuilder::new();
    let z = dag.symbol(field).unwrap();
    let p = dag.symbol(parameter).unwrap();
    let square = dag.mul(z, z).unwrap();
    let conjugate = dag.unary_math(UnaryMathFunction::Conj, z).unwrap();
    let abs2 = dag.unary_math(UnaryMathFunction::Abs2, z).unwrap();
    let real = dag.unary_math(UnaryMathFunction::Real, z).unwrap();
    let imag = dag.unary_math(UnaryMathFunction::Imag, z).unwrap();
    let scaled = dag.mul(p, z).unwrap();
    let typed = TypedResidual::infer(
        dag.finish([square, conjugate, abs2, real, imag, scaled])
            .unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            let domain = if symbol == field {
                ScalarDomain::Complex
            } else {
                ScalarDomain::Real
            };
            Ok::<_, ()>(ExpressionType::<()>::new(
                ValueType::scalar(domain, DimExponents::DIMENSIONLESS).unwrap(),
                None,
            ))
        },
    )
    .unwrap();
    let lowered = ComponentScalarization::lower(&typed).unwrap();
    assert!(lowered.linearize(|_| None).is_err());
    assert!(
        lowered
            .linearize(|_| Some((f64::NAN, DifferentiationRole::Frozen)))
            .is_err()
    );
    let mut resolved = std::collections::HashSet::new();
    let frozen = lowered
        .linearize(|coordinate| {
            assert!(
                resolved.insert(coordinate.clone()),
                "resolve each source coordinate once"
            );
            Some(if coordinate.symbol() == parameter {
                (5., DifferentiationRole::Parameter)
            } else {
                (
                    if coordinate.is_imaginary() { 3. } else { 2. },
                    DifferentiationRole::Frozen,
                )
            })
        })
        .unwrap();
    assert_eq!(resolved.len(), 3);
    assert_eq!(frozen.unknown_dimension(), 0);
    let mut frozen_tangent = [0.; 9];
    frozen
        .jvp(RelationTangent::Parameter(&[1.]), &mut frozen_tangent)
        .unwrap();
    assert_eq!(frozen_tangent, [0., 0., 0., 0., 0., 0., 0., 2., 3.]);
    assert!(
        frozen
            .jvp(RelationTangent::Unknown(&[1.]), &mut frozen_tangent)
            .is_err()
    );
    assert!(
        frozen
            .jvp(
                RelationTangent::Parameter(&[f64::INFINITY]),
                &mut frozen_tangent
            )
            .is_err()
    );
    assert!(
        frozen
            .jvp(RelationTangent::Parameter(&[1.]), &mut [0.; 8])
            .is_err()
    );
    assert!(
        frozen
            .vjp(&[1.; 8], RelationCotangent::Parameter(&mut [0.]))
            .is_err()
    );
    // At (x,y,p)=(2,3,5): z²=(x²-y²)+2xy i; conj(z)=x-yi;
    // |z|²=x²+y²; pz=px+py i. Each row below is d(output)/d(x,y,p).
    let jacobian = [
        [4., -6., 0.],
        [6., 4., 0.],
        [1., 0., 0.],
        [0., -1., 0.],
        [4., 6., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [5., 0., 2.],
        [0., 5., 3.],
    ];
    let primal = [-5., 12., 2., -3., 13., 2., 3., 10., 15.];
    let direction = [-2., 4., 3.];
    let weights = [2., -3., 5., 7., -2., 11., -5., 3., -4.];
    assert_eq!(lowered.rows().len(), jacobian.len());
    let mut pullback = [0.; 3];
    let mut pairing = 0.;
    for (index, row) in lowered.rows().iter().enumerate() {
        let slots = row
            .symbols()
            .iter()
            .map(|coordinate| {
                if coordinate.symbol() == parameter {
                    2
                } else {
                    usize::from(coordinate.is_imaginary())
                }
            })
            .collect::<Vec<_>>();
        let inputs = slots
            .iter()
            .map(|&slot| [2., 3., 5.][slot])
            .collect::<Vec<_>>();
        let roles = slots
            .iter()
            .map(|&slot| {
                if slot == 2 {
                    DifferentiationRole::Parameter
                } else {
                    DifferentiationRole::Unknown
                }
            })
            .collect::<Vec<_>>();
        let linearized = row.linearize(&inputs, &roles).unwrap();
        let mut value = [0.];
        linearized.primal(&mut value).unwrap();
        assert_eq!(value[0], primal[index]);
        let unknown_slots = slots
            .iter()
            .copied()
            .filter(|&slot| slot != 2)
            .collect::<Vec<_>>();
        let parameter_slots = slots
            .iter()
            .copied()
            .filter(|&slot| slot == 2)
            .collect::<Vec<_>>();
        let tangent = unknown_slots
            .iter()
            .map(|&slot| direction[slot])
            .collect::<Vec<_>>();
        let parameter_tangent = parameter_slots
            .iter()
            .map(|&slot| direction[slot])
            .collect::<Vec<_>>();
        linearized
            .jvp(
                RelationTangent::Both {
                    unknown: &tangent,
                    parameter: &parameter_tangent,
                },
                &mut value,
            )
            .unwrap();
        let expected: f64 = jacobian[index]
            .iter()
            .zip(direction)
            .map(|(a, b)| a * b)
            .sum();
        assert_eq!(value[0], expected);
        pairing += weights[index] * value[0];
        let mut adjoint = vec![0.; unknown_slots.len()];
        let mut parameter_adjoint = vec![0.; parameter_slots.len()];
        linearized
            .vjp(
                &[weights[index]],
                RelationCotangent::Both {
                    unknown: &mut adjoint,
                    parameter: &mut parameter_adjoint,
                },
            )
            .unwrap();
        for (slot, actual) in unknown_slots
            .into_iter()
            .zip(adjoint)
            .chain(parameter_slots.into_iter().zip(parameter_adjoint))
        {
            assert_eq!(actual, weights[index] * jacobian[index][slot]);
            pullback[slot] += actual;
        }
    }
    // Re(conj(w) Jv) = Re(conj(J* w) v), including mixed real inputs/outputs.
    let combined = lowered
        .linearize(|coordinate| {
            Some(if coordinate.symbol() == parameter {
                (5., DifferentiationRole::Parameter)
            } else {
                (
                    if coordinate.is_imaginary() { 3. } else { 2. },
                    DifferentiationRole::Unknown,
                )
            })
        })
        .unwrap();
    let unknown_direction = combined
        .unknown_coordinates()
        .iter()
        .map(|c| direction[usize::from(c.is_imaginary())])
        .collect::<Vec<_>>();
    let mut combined_primal = vec![f64::NAN; primal.len()];
    combined.primal(&mut combined_primal).unwrap();
    assert_eq!(combined_primal, primal);
    let mut combined_tangent = vec![0.; primal.len()];
    combined
        .jvp(
            RelationTangent::Both {
                unknown: &unknown_direction,
                parameter: &[3.],
            },
            &mut combined_tangent,
        )
        .unwrap();
    assert_eq!(
        combined_tangent
            .iter()
            .zip(weights)
            .map(|(a, b)| a * b)
            .sum::<f64>(),
        pairing
    );
    let mut combined_unknown = vec![0.; combined.unknown_dimension()];
    let mut combined_parameter = [0.];
    combined
        .vjp(
            &weights,
            RelationCotangent::Both {
                unknown: &mut combined_unknown,
                parameter: &mut combined_parameter,
            },
        )
        .unwrap();
    for (coordinate, actual) in combined.unknown_coordinates().iter().zip(combined_unknown) {
        assert_eq!(actual, pullback[usize::from(coordinate.is_imaginary())]);
    }
    assert_eq!(combined_parameter[0], pullback[2]);
    assert_eq!(
        pairing,
        pullback
            .iter()
            .zip(direction)
            .map(|(a, b)| a * b)
            .sum::<f64>()
    );
}
