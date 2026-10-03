use super::*;
use eqiora_core::{DimExponents, FiniteBasis, Id, entity::kinds};
use eqiora_ir::ComponentScalarization;
use eqiora_schema::kernel::{
    ExprDagBuilder, FiniteBinaryOperation as B, FiniteUnaryOperation as U,
    typing::{ExpressionType, RootContract, TypedResidual},
};

fn coordinates(basis: FiniteBasis, values: &[(f64, f64)]) -> ValueLiteral {
    ValueLiteral::new(
        ValueType::coordinates(basis, ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap(),
        values.iter().copied(),
    )
    .unwrap()
}
fn map(source: FiniteBasis, target: FiniteBasis, values: &[(f64, f64)]) -> ValueLiteral {
    ValueLiteral::new(
        ValueType::linear_map(
            source,
            target,
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap(),
        values.iter().copied(),
    )
    .unwrap()
}
fn evaluate(dag: &ExprDag) -> Result<Vec<ValueLiteral>, Diagnostic> {
    evaluate_selected(
        Id::<kinds::Relation>::new().erase(),
        dag,
        dag.roots(),
        &mut |_| None,
    )
}
fn typed(dag: ExprDag) -> TypedResidual<()> {
    TypedResidual::infer(dag, None, RootContract::ComponentwiseResidual, |_| {
        Err::<ExpressionType<()>, _>(())
    })
    .unwrap()
}

#[test]
fn finite_gaussian_integer_products_distinguish_transpose_and_adjoint() {
    let s = FiniteBasis::new(Id::new(), 2).unwrap();
    let t = FiniteBasis::new(Id::new(), 3).unwrap();
    let c = FiniteBasis::new(Id::new(), 2).unwrap();
    let mut b = ExprDagBuilder::new();
    let a = b
        .constant(map(
            s,
            t,
            &[(1., 0.), (0., 1.), (2., 0.), (-1., 0.), (0., -1.), (3., 0.)],
        ))
        .unwrap();
    let other = b
        .constant(map(c, s, &[(1., 0.), (2., 0.), (0., 1.), (-1., 0.)]))
        .unwrap();
    let z = b.constant(coordinates(s, &[(1., 1.), (2., -1.)])).unwrap();
    let apply = b.finite_binary(B::Apply, a, z).unwrap();
    let compose = b.finite_binary(B::Compose, a, other).unwrap();
    let transpose = b.finite_unary(U::Transpose, a).unwrap();
    let adjoint = b.finite_unary(U::Adjoint, a).unwrap();
    let zt = b.finite_unary(U::Transpose, z).unwrap();
    let zh = b.finite_unary(U::Adjoint, z).unwrap();
    let bilinear = b.finite_binary(B::Pair, zt, z).unwrap();
    let hermitian = b.finite_binary(B::Pair, zh, z).unwrap();
    let dag = b
        .finish([apply, compose, transpose, adjoint, bilinear, hermitian])
        .unwrap();
    let actual = evaluate(&dag).unwrap();
    // A=[[1,i],[2,-1],[-i,3]], B=[[1,2],[i,-1]], z=[1+i,2-i].
    // Distribute products using i²=-1; zᵀz=3-2i, z†z=2+5=7.
    let expected: Vec<Vec<(f64, f64)>> = vec![
        vec![(2., 3.), (0., 3.), (7., -4.)],
        vec![
            (0., 0.),
            (2., -1.),
            (2., -1.),
            (5., 0.),
            (0., 2.),
            (-3., -2.),
        ],
        vec![(1., 0.), (2., 0.), (0., -1.), (0., 1.), (-1., 0.), (3., 0.)],
        vec![(1., 0.), (2., 0.), (0., 1.), (0., -1.), (-1., 0.), (3., 0.)],
        vec![(3., -2.)],
        vec![(7., 0.)],
    ];
    for (value, expected) in actual.iter().zip(&expected) {
        assert_eq!(
            (0..value.component_count())
                .map(|i| value.component(i).unwrap())
                .collect::<Vec<_>>(),
            *expected
        );
    }
    assert_eq!(actual[0].value_type().coordinate_basis(), Some(t));
    assert_eq!(actual[1].value_type().map_bases(), Some((c, t)));
    assert_eq!(
        actual[2].value_type().map_bases(),
        Some((t.dual(), s.dual()))
    );
    assert_eq!(actual[3].value_type().map_bases(), Some((t, s)));
    let lowered = ComponentScalarization::lower(&typed(dag))
        .unwrap()
        .evaluate(|_| None)
        .unwrap();
    let expected_parts = expected
        .iter()
        .flatten()
        .flat_map(|&(real, imaginary)| [real, imaginary])
        .collect::<Vec<_>>();
    assert_eq!(lowered, expected_parts);
}

#[test]
fn finite_operations_reject_foreign_equal_extents_and_preserve_dimensions_under_scaling() {
    let s = FiniteBasis::new(Id::new(), 2).unwrap();
    let foreign = FiniteBasis::new(Id::new(), 2).unwrap();
    let unit = DimExponents::DIMENSIONLESS;
    let v = ExpressionType::<()>::new(
        ValueType::coordinates(s, ScalarDomain::Complex, unit).unwrap(),
        None,
    );
    let alien = ExpressionType::<()>::new(
        ValueType::coordinates(foreign, ScalarDomain::Complex, unit).unwrap(),
        None,
    );
    let matrix = ExpressionType::<()>::new(
        ValueType::linear_map(s, s, ScalarDomain::Complex, unit).unwrap(),
        None,
    );
    assert!(
        matrix
            .clone()
            .finite_binary(B::Apply, alien.clone())
            .is_err()
    );
    assert!(v.clone().finite_binary(B::Pair, v.clone()).is_err());
    assert!(
        matrix
            .clone()
            .finite_binary(
                B::Compose,
                ExpressionType::<()>::new(
                    ValueType::linear_map(s, foreign, ScalarDomain::Complex, unit).unwrap(),
                    None
                )
            )
            .is_err()
    );
    let array = ExpressionType::<()>::new(
        ValueType::scalar(ScalarDomain::Complex, unit)
            .unwrap()
            .array(2)
            .unwrap(),
        None,
    );
    assert!(matrix.finite_binary(B::Apply, array).is_err());
    let spatial = ExpressionType::<()>::new(
        ValueType::shaped(
            ScalarDomain::Complex,
            unit,
            eqiora_core::ValueShape::new([2]).unwrap(),
            eqiora_core::ValueFrame::SpatialCartesian,
        )
        .unwrap(),
        None,
    );
    assert!(spatial.finite_unary(U::Adjoint).is_err());
    let seconds = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let mut b = ExprDagBuilder::new();
    let value = b.constant(coordinates(s, &[(1., 1.), (2., -1.)])).unwrap();
    let factor = b
        .constant(
            ValueLiteral::from_real(ValueType::scalar(ScalarDomain::Real, seconds).unwrap(), 2.)
                .unwrap(),
        )
        .unwrap();
    let scaled = b.mul(factor, value).unwrap();
    let restored = b.div(scaled, factor).unwrap();
    let values = evaluate(&b.finish([scaled, restored]).unwrap()).unwrap();
    assert_eq!(values[0].value_type().dimension(), seconds);
    assert_eq!(values[0].value_type().coordinate_basis(), Some(s));
    assert_eq!(values[0].component(1), Some((4., -2.)));
    assert_eq!(values[1], coordinates(s, &[(1., 1.), (2., -1.)]));
}

#[test]
fn finite_contraction_work_rejects_before_large_expansion() {
    let basis = FiniteBasis::new(Id::new(), 100).unwrap();
    let map_type = ValueType::linear_map(
        basis,
        basis,
        ScalarDomain::Real,
        DimExponents::DIMENSIONLESS,
    )
    .unwrap();
    let mut b = ExprDagBuilder::new();
    let zero = b
        .constant(ValueLiteral::from_real(map_type, 0.).unwrap())
        .unwrap();
    let product = b.finite_binary(B::Compose, zero, zero).unwrap();
    let error = evaluate(&b.finish([product]).unwrap()).unwrap_err();
    assert!(
        error
            .message()
            .contains("one-million component work budget"),
        "{error:?}"
    );
    let large = FiniteBasis::new(Id::new(), 1_000_001).unwrap();
    let mut b = ExprDagBuilder::new();
    let v = b
        .constant(
            ValueLiteral::from_real(
                ValueType::coordinates(large, ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                    .unwrap(),
                0.,
            )
            .unwrap(),
        )
        .unwrap();
    let dual = b.finite_unary(U::Transpose, v).unwrap();
    let pair = b.finite_binary(B::Pair, dual, v).unwrap();
    let error = ComponentScalarization::lower(&typed(b.finish([pair]).unwrap())).unwrap_err();
    assert!(
        error.message().contains("one million component products"),
        "{error:?}"
    );
}
