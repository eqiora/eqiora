use super::*;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueFrame, ValueShape, entity::kinds};
use eqiora_schema::kernel::{
    ExprDagBuilder,
    typing::{RootContract, SpatialSupport},
};

#[test]
fn field_gradients_keep_channel_component_direction_phase_and_units_separate() {
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let support = SpatialSupport::Volume {
        domain: (),
        dimensions: 2,
    };
    for complex in [false, true] {
        for channels in [false, true] {
            let element = ExpressionType::shaped(
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                ValueShape::new([2]).unwrap(),
                ValueFrame::SpatialCartesian,
                Some(support.clone()),
            )
            .unwrap();
            let ty = element
                .value_type
                .with_scalar_domain(if complex {
                    ScalarDomain::Complex
                } else {
                    ScalarDomain::Real
                })
                .unwrap();
            let ty = if channels { ty.array(3).unwrap() } else { ty };
            let mut builder = ExprDagBuilder::new();
            let value = builder.symbol(field).unwrap();
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
            assert!(ComponentScalarization::lower_selected(&typed, &[gradient]).is_err());
            let lowered = ComponentScalarization::lower_spatial_selected(
                &typed,
                &[value, gradient, hessian],
                None,
            )
            .unwrap();
            for row in lowered.rows() {
                let order = row.root_index();
                let coordinate = &row.symbols()[0];
                assert_eq!(coordinate.symbol(), field);
                let rank = ty.shape().rank();
                assert_eq!(coordinate.component_index(), &row.component_index()[..rank]);
                assert_eq!(
                    coordinate.spatial_derivative_axes(),
                    &row.component_index()[rank..]
                        .iter()
                        .map(|i| *i as usize)
                        .collect::<Vec<_>>()
                );
                assert_eq!(coordinate.spatial_derivative_axes().len(), order);
                assert_eq!(coordinate.is_imaginary(), row.is_imaginary());
                assert_eq!(
                    row.input_slots()[0].dimension(),
                    DimExponents::from_integers([0, 1 - order as i32, 0, 0, 0, 0, 0]).unwrap()
                );
            }
            let values = lowered
                .evaluate(|coordinate| {
                    let component =
                        ty.shape().flat_index(coordinate.component_index()).unwrap() as f64;
                    let direction = coordinate
                        .spatial_derivative_axes()
                        .iter()
                        .fold(0., |value, axis| 3. * value + *axis as f64 + 1.);
                    Some(100. * component + 10. * direction + f64::from(coordinate.is_imaginary()))
                })
                .unwrap();
            let width = ty.shape().component_count().unwrap() * if complex { 2 } else { 1 };
            assert_eq!(values.len(), width * (1 + 2 + 4));
        }
    }
}

#[test]
fn partial_divergence_and_oblique_normal_keep_complex_coordinates_and_axes() {
    let volume = Id::<kinds::Domain>::new();
    let boundary = Id::<kinds::Domain>::new();
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let support = SpatialSupport::Volume {
        domain: volume.erase(),
        dimensions: 2,
    };
    let surface = SpatialSupport::Boundary {
        domain: boundary.erase(),
        parent: volume.erase(),
        dimensions: 2,
    };
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let ty = eqiora_core::ValueType::shaped(
        ScalarDomain::Complex,
        length,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let mut builder = ExprDagBuilder::new();
    let u = builder.symbol(field).unwrap();
    let y = builder
        .symbol(SymbolRef::Coordinate {
            support: volume,
            factor: volume,
            axis: 1,
        })
        .unwrap();
    let partial = builder.coordinate_partial(u, y).unwrap();
    let divergence = builder.divergence(u).unwrap();
    let trace = builder.trace(u, boundary).unwrap();
    let gradient = builder.gradient(u).unwrap();
    let normal = builder.normal_component(gradient, boundary).unwrap();
    let typed = TypedResidual::infer(
        builder
            .finish([partial, divergence, trace, normal])
            .unwrap(),
        None,
        |id| (id == boundary).then(|| surface.clone()),
        RootContract::ValueRoots,
        |symbol| {
            Ok::<_, ()>(if symbol == field {
                ExpressionType::new(ty.clone(), Some(support.clone()))
            } else {
                ExpressionType::scalar(length, Some(support.clone()))
            })
        },
    )
    .unwrap();
    for absent_or_wrong in [
        None,
        Some(&[0.0, 0.0][..]),
        Some(&[1.0][..]),
        Some(&[f64::NAN, 1.0][..]),
    ] {
        assert!(
            ComponentScalarization::lower_spatial_selected(&typed, &[normal], absent_or_wrong)
                .is_err()
        );
    }
    assert!(ComponentScalarization::lower(&typed).is_err());
    let lowered = ComponentScalarization::lower_spatial_selected(
        &typed,
        typed.expression().roots(),
        Some(&[0.6, -0.8]),
    )
    .unwrap();
    let actual = lowered
        .evaluate(|input| {
            assert_eq!(input.symbol(), field);
            let component = input.component_index()[0] as usize;
            let phase = usize::from(input.is_imaginary());
            Some(match input.spatial_derivative_axes() {
                [] => [[2., 3.], [5., -7.]][component][phase],
                [axis] => {
                    [[[11., 13.], [23., 29.]], [[17., 19.], [31., 37.]]][component][phase][*axis]
                }
                _ => panic!("first derivatives only"),
            })
        })
        .unwrap();
    let expected = [
        13., 29., 19., 37., 30., 60., 2., 3., 5., -7., -3.8, -9.4, -5., -11.,
    ];
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12);
    }
}
