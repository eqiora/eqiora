use eqiora_core::{DimExponents, ScalarDomain, ValueFrame, ValueLiteral, ValueShape, ValueType};
use eqiora_ir::{ComponentScalarization, PureOperatorDefinition};
use eqiora_schema::kernel::{
    ExprDagBuilder,
    typing::{ExpressionType, RootContract, TypedResidual},
};

fn tensor(shape: &[u32], values: &[f64]) -> ValueLiteral {
    let ty = ValueType::shaped(
        ScalarDomain::Real,
        DimExponents::DIMENSIONLESS,
        ValueShape::new(shape.iter().copied()).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    ValueLiteral::new(ty, values.iter().map(|value| (*value, 0.0))).unwrap()
}

fn execute(definition: &PureOperatorDefinition, inputs: &[ValueLiteral]) -> Vec<f64> {
    let mut builder = ExprDagBuilder::new();
    let arguments = inputs
        .iter()
        .map(|value| builder.constant(value.clone()).unwrap())
        .collect::<Vec<_>>();
    let root = builder.pure_operator(definition, arguments).unwrap();
    let typed = TypedResidual::<()>::infer(
        builder.finish([root]).unwrap(),
        None,
        RootContract::ComponentwiseResidual,
        |_| -> Result<ExpressionType<()>, ()> { unreachable!("closed constants") },
    )
    .unwrap();
    ComponentScalarization::lower(&typed)
        .unwrap()
        .evaluate(|_| None)
        .unwrap()
}

#[test]
fn nonsymmetric_map_and_axis_permutation_execute_through_existing_scalar_ir() {
    let map = tensor(&[2, 2], &[2.0, 3.0, 5.0, 7.0]);
    let vector = tensor(&[2], &[11.0, 13.0]);
    assert_eq!(
        execute(
            &PureOperatorDefinition::contract(2, 2, 1, &[(1, 0)]).unwrap(),
            &[map.clone(), vector]
        ),
        vec![61.0, 146.0]
    );
    assert_eq!(
        execute(
            &PureOperatorDefinition::permute_axes(2, &[1, 0]).unwrap(),
            std::slice::from_ref(&map)
        ),
        vec![2.0, 5.0, 3.0, 7.0]
    );
    assert_eq!(
        execute(
            &PureOperatorDefinition::tensor_component(2, &[1, 0]).unwrap(),
            &[map]
        ),
        vec![5.0]
    );
}

#[test]
fn full_rank_four_shear_keeps_both_off_diagonal_contributions() {
    // Independently specified full coordinates: C0000=10, C1111=20,
    // C0011=C1100=3 and C0101=C0110=C1001=C1010=4.
    let stiffness = tensor(
        &[2, 2, 2, 2],
        &[
            10.0, 0.0, 0.0, 3.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0, 0.0, 3.0, 0.0, 0.0, 20.0,
        ],
    );
    let shear = tensor(&[2, 2], &[0.0, 0.01, 0.01, 0.0]);
    let contract = PureOperatorDefinition::contract(2, 4, 2, &[(2, 0), (3, 1)]).unwrap();
    let stress = execute(&contract, &[stiffness.clone(), shear.clone()]);
    assert_eq!(stress, vec![0.0, 0.08, 0.08, 0.0]);
    let twice_energy = execute(
        &PureOperatorDefinition::contract(2, 2, 2, &[(0, 0), (1, 1)]).unwrap(),
        &[shear, tensor(&[2, 2], &stress)],
    );
    assert!((twice_energy[0] / 2.0 - 0.0008).abs() < 1e-15);
    let normal = execute(
        &contract,
        &[stiffness, tensor(&[2, 2], &[0.01, 0.0, 0.0, 0.02])],
    );
    for (actual, expected) in normal.iter().zip([0.16, 0.0, 0.0, 0.43]) {
        assert!((actual - expected).abs() < 1e-15);
    }
}

#[test]
fn explicit_axes_and_extent_constraints_reject_invalid_applications() {
    assert!(PureOperatorDefinition::contract(2, 2, 2, &[(0, 0), (0, 1)]).is_err());
    assert!(PureOperatorDefinition::contract(2, 2, 2, &[(2, 0)]).is_err());
    assert!(PureOperatorDefinition::permute_axes(2, &[0, 0]).is_err());
    assert!(PureOperatorDefinition::tensor_component(2, &[2, 0]).is_err());
    assert!(
        PureOperatorDefinition::contract(u32::MAX, 4, 4, &[(0, 0), (1, 1), (2, 2), (3, 3)])
            .is_err()
    );
    let definition = PureOperatorDefinition::contract(2, 2, 1, &[(1, 0)]).unwrap();
    let types = [tensor(&[3, 3], &[0.0; 9]), tensor(&[3], &[0.0; 3])]
        .map(|value| ExpressionType::<()>::new(value.value_type().clone(), None));
    assert!(definition.instantiate(&types).is_err());
}

#[test]
fn complex_contraction_is_bilinear_without_implicit_conjugation() {
    let ty = ValueType::shaped(
        ScalarDomain::Complex,
        DimExponents::DIMENSIONLESS,
        ValueShape::new([2]).unwrap(),
        ValueFrame::SpatialCartesian,
    )
    .unwrap();
    let left = ValueLiteral::new(ty.clone(), [(1.0, 2.0), (3.0, -1.0)]).unwrap();
    let right = ValueLiteral::new(ty, [(4.0, -3.0), (-2.0, 5.0)]).unwrap();
    // (1+2i)(4-3i)=10+5i; (3-i)(-2+5i)=-1+17i.
    let dot = PureOperatorDefinition::contract(2, 1, 1, &[(0, 0)]).unwrap();
    assert_eq!(
        execute(&dot, &[left.clone(), right.clone()]),
        vec![9.0, 22.0]
    );
    assert_eq!(
        execute(&dot, &[left, tensor(&[2], &[4.0, -2.0])]),
        vec![-2.0, 10.0]
    );
    assert_eq!(
        execute(&dot, &[tensor(&[2], &[1.0, 3.0]), right]),
        vec![-2.0, 12.0]
    );
}

#[test]
fn scalar_component_projection_preserves_axes_and_bounds() {
    let map = tensor(&[2, 2], &[2.0, 3.0, 5.0, 7.0]);
    let vector = tensor(&[2], &[11.0, 13.0]);
    let types =
        [&map, &vector].map(|value| ExpressionType::<()>::new(value.value_type().clone(), None));
    let mut builder = ExprDagBuilder::new();
    let mut arguments = Vec::new();
    for value in [&map, &vector] {
        arguments.push(
            value
                .components()
                .unwrap()
                .map(|(real, imaginary)| {
                    assert_eq!(imaginary, 0.0);
                    builder
                        .constant(eqiora_core::DynQuantity::new(
                            real,
                            DimExponents::DIMENSIONLESS,
                        ))
                        .unwrap()
                })
                .collect::<Vec<_>>(),
        );
    }
    let transpose = PureOperatorDefinition::permute_axes(2, &[1, 0]).unwrap();
    let transposed = transpose.instantiate(&types[..1]).unwrap();
    assert!(
        builder
            .project_operator_component(&transposed, &[&arguments[0][..3]], &[0, 1], 100)
            .is_err()
    );
    assert!(
        builder
            .project_operator_component(&transposed, &arguments[..1], &[2, 0], 100)
            .is_err()
    );
    assert!(
        builder
            .project_operator_component(&transposed, &arguments[..1], &[], 100)
            .is_err()
    );
    assert!(
        builder
            .project_operator_component(&transposed, &arguments[..1], &[0, 1], 1)
            .is_err()
    );
    let selected = builder
        .project_operator_component(&transposed, &arguments[..1], &[0, 1], 100)
        .unwrap();
    let contraction = PureOperatorDefinition::contract(2, 2, 1, &[(1, 0)]).unwrap();
    let contracted = contraction.instantiate(&types).unwrap();
    let first = builder
        .project_operator_component(&contracted, &arguments, &[0], 100)
        .unwrap();
    let second = builder
        .project_operator_component(&contracted, &arguments, &[1], 100)
        .unwrap();
    let dag = builder.finish([selected, first, second]).unwrap();
    assert_eq!(
        eqiora_ir::ScalarOperatorIr::lower(&dag)
            .unwrap()
            .evaluate(&[])
            .unwrap(),
        vec![5.0, 61.0, 146.0]
    );
}
