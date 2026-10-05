use eqiora_core::ValueFrame;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, Id, ValueShape};
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
use eqiora_schema::kernel::typing::{ExpressionType, RootContract, SpatialSupport, TypedResidual};
use eqiora_schema::kernel::{ExprDagBuilder, SymbolRef};

use super::ComponentScalarization;

#[test]
fn typed_constant_arrays_retain_both_complex_parts() {
    use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
    for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
        let value_type = ValueType::scalar(domain, DimExponents::DIMENSIONLESS)
            .expect("fixture uses a numeric scalar domain")
            .array(3)
            .unwrap();
        let mut builder = ExprDagBuilder::new();
        let root = builder
            .constant(ValueLiteral::from_real(value_type.clone(), 0.0).unwrap())
            .unwrap();
        let typed = TypedResidual::infer(
            builder.finish([root]).unwrap(),
            None::<SpatialSupport<()>>,
            RootContract::ComponentwiseResidual,
            |_| -> Result<ExpressionType<()>, ()> { unreachable!("constant has no symbols") },
        )
        .unwrap();
        assert_eq!(typed.node_type(root).unwrap().value_type, value_type);
        let result = ComponentScalarization::lower(&typed);
        match domain {
            ScalarDomain::Integer | ScalarDomain::Boolean | ScalarDomain::Enum => {
                unreachable!("test iterates real and complex domains")
            }
            ScalarDomain::Real => assert_eq!(result.unwrap().rows().len(), 3),
            ScalarDomain::Complex => assert_eq!(result.unwrap().rows().len(), 6),
        }
    }
}

#[test]
fn scalarization_retains_exact_field_identity_for_both_complex_parts() {
    use eqiora_core::ScalarDomain;
    use eqiora_core::ValueType;
    let mut builder = ExprDagBuilder::new();
    let root = builder.symbol(SymbolRef::Field(Id::new())).unwrap();
    let expression = builder.finish([root]).unwrap();
    for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
        let typed = TypedResidual::infer(
            expression.clone(),
            None::<SpatialSupport<()>>,
            RootContract::ComponentwiseResidual,
            |_| {
                Ok::<_, ()>(ExpressionType::new(
                    ValueType::scalar(domain, DimExponents::DIMENSIONLESS)
                        .expect("fixture uses a numeric scalar domain"),
                    None,
                ))
            },
        )
        .unwrap();
        let result = ComponentScalarization::lower(&typed);
        match domain {
            ScalarDomain::Integer | ScalarDomain::Boolean | ScalarDomain::Enum => {
                unreachable!("test iterates real and complex domains")
            }
            ScalarDomain::Real => assert_eq!(result.unwrap().rows().len(), 1),
            ScalarDomain::Complex => {
                let result = result.unwrap();
                assert_eq!(result.rows().len(), 2);
                let first = &result.rows()[0];
                let second = &result.rows()[1];
                assert_eq!(first.symbols()[0].symbol(), second.symbols()[0].symbol());
                assert!(!first.is_imaginary());
                assert!(second.is_imaginary());
                assert!(!first.symbols()[0].is_imaginary());
                assert!(second.symbols()[0].is_imaginary());
            }
        }
    }
}

#[test]
fn vector_root_scalarizes_in_root_then_row_major_component_order() {
    let port = Id::<kinds::Port>::new();
    let parameter = Id::<kinds::Parameter>::new();
    let mut expression = ExprDagBuilder::new();
    let trace = expression.symbol(SymbolRef::PortTrace(port)).unwrap();
    let scale = expression.symbol(SymbolRef::Parameter(parameter)).unwrap();
    let scaled = expression.mul(trace, scale).unwrap();
    let dag = expression.finish([scaled]).unwrap();
    let vector = ValueShape::new([2]).unwrap();
    let typed = TypedResidual::infer(dag, None, RootContract::ComponentwiseResidual, |symbol| {
        Ok::<_, ()>(match symbol {
            SymbolRef::PortTrace(_) => ExpressionType::shaped(
                DimExponents::DIMENSIONLESS,
                vector.clone(),
                eqiora_core::ValueFrame::SpatialCartesian,
                None::<eqiora_schema::kernel::typing::SpatialSupport<()>>,
            )
            .unwrap(),
            SymbolRef::Parameter(_) => ExpressionType::scalar(DimExponents::DIMENSIONLESS, None),
            _ => unreachable!(),
        })
    })
    .unwrap();
    let lowering = ComponentScalarization::lower(&typed).unwrap();

    assert_eq!(lowering.rows().len(), 2);
    assert_eq!(lowering.rows()[0].component_index(), [0]);
    assert_eq!(lowering.rows()[1].component_index(), [1]);
    let values = lowering
        .evaluate(|coordinate| match coordinate.symbol() {
            SymbolRef::PortTrace(_) => Some(if coordinate.component_index() == [0] {
                2.0
            } else {
                4.0
            }),
            SymbolRef::Parameter(_) => Some(3.0),
            _ => None,
        })
        .unwrap();
    assert_eq!(values, [6.0, 12.0]);
}

#[test]
fn multiple_roots_preserve_root_then_last_axis_fastest_order() {
    let first = Id::<kinds::Port>::new();
    let second = Id::<kinds::Port>::new();
    let mut expression = ExprDagBuilder::new();
    let scalar_root = expression.symbol(SymbolRef::PortTrace(first)).unwrap();
    let tensor_root = expression.symbol(SymbolRef::PortFlux(second)).unwrap();
    let dag = expression.finish([scalar_root, tensor_root]).unwrap();
    let tensor = ValueShape::new([2, 2]).unwrap();
    let typed = TypedResidual::infer(dag, None, RootContract::ComponentwiseResidual, |symbol| {
        Ok::<_, ()>(match symbol {
            SymbolRef::PortTrace(_) => ExpressionType::scalar(DimExponents::DIMENSIONLESS, None),
            SymbolRef::PortFlux(_) => ExpressionType::shaped(
                DimExponents::DIMENSIONLESS,
                tensor.clone(),
                eqiora_core::ValueFrame::SpatialCartesian,
                None::<eqiora_schema::kernel::typing::SpatialSupport<()>>,
            )
            .unwrap(),
            _ => unreachable!(),
        })
    })
    .unwrap();
    let lowering = ComponentScalarization::lower(&typed).unwrap();

    let order = lowering
        .rows()
        .iter()
        .map(|row| (row.root_index(), row.component_index().to_vec()))
        .collect::<Vec<_>>();
    assert_eq!(
        order,
        [
            (0, vec![]),
            (1, vec![0, 0]),
            (1, vec![0, 1]),
            (1, vec![1, 0]),
            (1, vec![1, 1]),
        ]
    );
}

#[test]
fn symmetric_part_reads_direct_and_swapped_tensor_coordinates() {
    let stress = Id::<kinds::Port>::new();
    let mut expression = ExprDagBuilder::new();
    let tensor = expression.symbol(SymbolRef::PortFlux(stress)).unwrap();
    let symmetric = expression.symmetric_part(tensor).unwrap();
    let dag = expression.finish([symmetric]).unwrap();
    let tensor_shape = ValueShape::new([2, 2]).unwrap();
    let support = SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    };
    let typed = TypedResidual::infer(
        dag,
        Some(support.clone()),
        RootContract::ComponentwiseResidual,
        |_| {
            Ok::<_, ()>(
                ExpressionType::shaped(
                    DimExponents::DIMENSIONLESS,
                    tensor_shape.clone(),
                    ValueFrame::SpatialCartesian,
                    Some(support.clone()),
                )
                .unwrap(),
            )
        },
    )
    .unwrap();
    let lowering = ComponentScalarization::lower(&typed).unwrap();

    assert_eq!(
        lowering.rows()[1]
            .symbols()
            .iter()
            .map(|coordinate| coordinate.component_index().to_vec())
            .collect::<Vec<_>>(),
        [vec![0, 1], vec![1, 0]]
    );
    assert_eq!(
        lowering.rows()[2]
            .symbols()
            .iter()
            .map(|coordinate| coordinate.component_index().to_vec())
            .collect::<Vec<_>>(),
        [vec![1, 0], vec![0, 1]]
    );

    let values = lowering
        .evaluate(|coordinate| match coordinate.component_index() {
            [0, 0] => Some(2.0),
            [0, 1] => Some(6.0),
            [1, 0] => Some(10.0),
            [1, 1] => Some(4.0),
            _ => None,
        })
        .unwrap();
    assert_eq!(values, [2.0, 8.0, 8.0, 4.0]);
}

#[test]
fn isotropic_lift_preserves_ordered_scalar_reads_for_every_component() {
    let pressure = Id::<kinds::Port>::new();
    let mut expression = ExprDagBuilder::new();
    let scalar = expression.symbol(SymbolRef::PortTrace(pressure)).unwrap();
    let isotropic = expression.isotropic_lift(scalar).unwrap();
    let dag = expression.finish([isotropic]).unwrap();
    let dimension = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
    let support = SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    };
    let typed = TypedResidual::infer(
        dag,
        Some(support.clone()),
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(ExpressionType::scalar(dimension, Some(support.clone()))),
    )
    .unwrap();
    let lowering = ComponentScalarization::lower(&typed).unwrap();

    for row in lowering.rows() {
        assert_eq!(row.symbols().len(), 1);
        assert_eq!(row.symbols()[0].symbol(), SymbolRef::PortTrace(pressure));
        assert!(row.symbols()[0].component_index().is_empty());
    }

    let values = lowering.evaluate(|_| Some(7.0)).unwrap();
    assert_eq!(values, [7.0, 0.0, 0.0, 7.0]);
}

#[test]
fn generic_dyadic_application_scalarizes_from_its_ordered_definition() {
    let left = Id::<kinds::Field>::new();
    let right = Id::<kinds::Field>::new();
    let definition = PureOperatorDefinition::dyadic_product().unwrap();
    let mut expression = ExprDagBuilder::new();
    let left_value = expression.symbol(SymbolRef::Field(left)).unwrap();
    let right_value = expression.symbol(SymbolRef::Field(right)).unwrap();
    let dyadic = expression
        .pure_operator(&definition, [left_value, right_value])
        .unwrap();
    let dag = expression.finish([dyadic]).unwrap();
    let support = SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    };
    let vector_type = ExpressionType::shaped(
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
        Some(support.clone()),
    )
    .unwrap();
    let typed = TypedResidual::infer(
        dag,
        Some(support),
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(vector_type.clone()),
    )
    .unwrap();

    let lowering = ComponentScalarization::lower(&typed).unwrap();
    assert_eq!(lowering.rows().len(), 4);
    for row in lowering.rows() {
        assert_eq!(row.symbols().len(), 2);
        assert_eq!(row.symbols()[0].symbol(), SymbolRef::Field(left));
        assert_eq!(row.symbols()[1].symbol(), SymbolRef::Field(right));
        assert_eq!(
            row.symbols()[0].component_index(),
            &row.component_index()[..1]
        );
        assert_eq!(
            row.symbols()[1].component_index(),
            &row.component_index()[1..]
        );
    }

    let values = lowering
        .evaluate(|coordinate| match coordinate.symbol() {
            SymbolRef::Field(field) if field == left => match coordinate.component_index() {
                [0] => Some(2.0),
                [1] => Some(3.0),
                _ => None,
            },
            SymbolRef::Field(field) if field == right => match coordinate.component_index() {
                [0] => Some(5.0),
                [1] => Some(7.0),
                _ => None,
            },
            _ => None,
        })
        .unwrap();
    assert_eq!(values, [10.0, 14.0, 15.0, 21.0]);
}

#[cfg(test)]
mod channel_tests {
    use super::*;
    use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};
    use eqiora_schema::kernel::{ExprDagBuilder, typing::RootContract};

    #[test]
    fn complete_constant_components_and_explicit_channel_index_keep_order() {
        let ty = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("numeric scalar type")
            .array(2)
            .unwrap();
        let mut dag = ExprDagBuilder::new();
        let left = dag
            .constant(ValueLiteral::new(ty.clone(), [(2.0, 0.0), (3.0, 0.0)]).unwrap())
            .unwrap();
        let right = dag
            .constant(ValueLiteral::new(ty, [(5.0, 0.0), (7.0, 0.0)]).unwrap())
            .unwrap();
        let channels = dag.array([left, right]).unwrap();
        let selected = dag.index(channels, 1).unwrap();
        let typed = TypedResidual::infer(
            dag.finish([channels, selected]).unwrap(),
            None,
            RootContract::ComponentwiseResidual,
            |_| -> Result<ExpressionType<()>, ()> { unreachable!() },
        )
        .unwrap();
        let lowered = ComponentScalarization::lower(&typed).unwrap();
        assert_eq!(
            lowered.evaluate(|_| None).unwrap(),
            [2.0, 3.0, 5.0, 7.0, 5.0, 7.0]
        );
    }
}

#[test]
fn selected_numeric_roots_do_not_admit_or_demand_discrete_roots() {
    use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
    let mut builder = ExprDagBuilder::new();
    let real = builder
        .constant(eqiora_core::DynQuantity::new(
            2.,
            DimExponents::DIMENSIONLESS,
        ))
        .unwrap();
    let integer = builder
        .constant(
            ValueLiteral::from_integer(
                ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS).unwrap(),
                9_007_199_254_740_993,
            )
            .unwrap(),
        )
        .unwrap();
    let typed = TypedResidual::infer(
        builder.finish([real, real, integer, integer]).unwrap(),
        None::<SpatialSupport<()>>,
        RootContract::InitialConditions,
        |_| -> Result<ExpressionType<()>, ()> { unreachable!("constant has no symbols") },
    )
    .unwrap();
    let projected = ComponentScalarization::lower_selected(&typed, &[real]).unwrap();
    assert_eq!(projected.rows()[0].evaluate(&[]).unwrap(), 2.);
    assert!(ComponentScalarization::lower_selected(&typed, &[integer]).is_err());
    assert!(ComponentScalarization::lower(&typed).is_err());
}

#[test]
fn constant_rate_matrix_retains_complex_parts_without_freezing_other_symbols() {
    use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
    let field = Id::<kinds::Field>::new();
    let rate = SymbolRef::Derivative(field, std::num::NonZeroU32::MIN);
    let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap();
    let selected = super::ScalarSymbolCoordinate::for_value(rate, &ty).unwrap();
    for profile in ["constant", "state-dependent", "nonlinear-rate"] {
        let mut builder = ExprDagBuilder::new();
        let w = builder.symbol(SymbolRef::Field(field)).unwrap();
        let dw = builder.symbol(rate).unwrap();
        let coefficient = match profile {
            "constant" => builder
                .constant(ValueLiteral::new(ty.clone(), [(0., 1.)]).unwrap())
                .unwrap(),
            "state-dependent" => w,
            _ => dw,
        };
        let mass = builder.mul(coefficient, dw).unwrap();
        let forcing = builder.mul(w, w).unwrap();
        let root = builder.add(mass, forcing).unwrap();
        let typed = TypedResidual::infer(
            builder.finish([root]).unwrap(),
            None::<SpatialSupport<()>>,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(ExpressionType::new(ty.clone(), None)),
        )
        .unwrap();
        let operator = ComponentScalarization::lower(&typed).unwrap();
        let proof = operator
            .rows()
            .iter()
            .map(|row| row.constant_coordinate_jacobian(&selected))
            .collect::<Result<Vec<_>, _>>();
        if profile == "constant" {
            let matrix = proof
                .unwrap()
                .into_iter()
                .flat_map(|row| row.coefficients().to_vec())
                .collect::<Vec<_>>();
            // Multiplication by i maps (a,b) to (-b,a), regardless of w^2.
            assert_eq!(matrix, [0., -1., 1., 0.]);
            assert!(
                operator.rows()[0]
                    .constant_coordinate_jacobian(&[selected[0].clone(), selected[0].clone(),])
                    .is_err()
            );
        } else {
            assert!(proof.is_err(), "{profile}");
        }
    }
}
