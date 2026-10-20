//! Affinity is structural; numerical bindings never choose the solver class.
use super::*;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueType};
use eqiora_schema::kernel::{
    ComparisonOp, ExprDagBuilder, UnaryMathFunction, typing::RootContract,
};

#[test]
fn affine_classification_retains_coefficients_branches_and_primal_domains() {
    let field = SymbolRef::Field(Id::new());
    let parameter = SymbolRef::Parameter(Id::new());
    let unit = DimExponents::DIMENSIONLESS;
    let ty = ValueType::scalar(ScalarDomain::Real, unit).unwrap();
    let u_coordinate = ScalarSymbolCoordinate::for_value(field, &ty).unwrap();
    let p_coordinate = ScalarSymbolCoordinate::for_value(parameter, &ty).unwrap();
    let mut b = ExprDagBuilder::new();
    let u = b.symbol(field).unwrap();
    let p = b.symbol(parameter).unwrap();
    let zero = b.constant(DynQuantity::new(0., unit)).unwrap();
    let product = b.mul(p, u).unwrap();
    let square = b.powi(u, 2).unwrap();
    let hidden_square = b.mul(p, square).unwrap();
    let root = b.unary_math(UnaryMathFunction::Sqrt, u).unwrap();
    let hidden_domain = b.powi(root, 0).unwrap();
    let frozen_guard = b.compare(ComparisonOp::Greater, p, zero).unwrap();
    let active_guard = b.compare(ComparisonOp::Greater, u, zero).unwrap();
    let guarded_affine = b.require(frozen_guard, product).unwrap();
    let guarded_unknown = b.require(active_guard, product).unwrap();
    let minus = b.neg(product).unwrap();
    let frozen_branch = b.select(frozen_guard, product, minus).unwrap();
    let active_branch = b.select(active_guard, product, minus).unwrap();
    let restricted_product = b.derivative_limit(vec![product], product, 0).unwrap();
    let typed = TypedResidual::<()>::infer(
        b.finish([
            product,
            hidden_square,
            hidden_domain,
            guarded_affine,
            guarded_unknown,
            frozen_branch,
            active_branch,
            restricted_product,
        ])
        .unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| Ok::<_, ()>(ExpressionType::new(ty.clone(), None)),
    )
    .unwrap();
    let lowered = ComponentScalarization::lower(&typed).unwrap();
    let rows = lowered.rows();
    for (row, expected) in rows
        .iter()
        .zip([true, false, false, true, false, true, false, false])
    {
        assert_eq!(row.affine_in(&u_coordinate).unwrap(), expected);
    }
    let mut selected = u_coordinate.clone();
    selected.extend(p_coordinate.iter().cloned());
    assert!(!rows[0].affine_in(&selected).unwrap());
    assert!(rows[0].affine_in(&[]).unwrap());
    // A frozen coefficient sampled at zero cannot hide an active input inside
    // the value-only law, even when ordinary affine coefficients cancel.
    for value in [0., 2.] {
        assert!(
            rows[7]
                .bind_affine(&u_coordinate, &[(p_coordinate[0].clone(), value)])
                .is_err()
        );
    }
    let degrees = rows[4]
        .symbols()
        .iter()
        .map(|coordinate| Some(usize::from(coordinate.symbol() == field)))
        .collect::<Vec<_>>();
    // Where its guard holds, p*u is degree one; its selected-dependent domain
    // still prevents an affine binding that never receives the selected value.
    assert_eq!(rows[4].polynomial_degree(&degrees), Some(1));
    assert!(!rows[4].affine_in(&u_coordinate).unwrap());
    assert!(
        rows[0]
            .affine_in(&[u_coordinate[0].clone(), u_coordinate[0].clone()])
            .is_err()
    );

    let values = |row: &ComponentScalarRow, u, p| {
        row.symbols()
            .iter()
            .map(|coordinate| if coordinate.symbol() == field { u } else { p })
            .collect::<Vec<_>>()
    };
    // p=0 cannot hide nonlinear unknown dependence, and u^0 cannot hide sqrt's domain.
    assert_eq!(rows[1].evaluate(&values(&rows[1], 2., 0.)).unwrap(), 0.);
    assert!(!rows[1].affine_in(&u_coordinate).unwrap());
    assert!(rows[2].evaluate(&values(&rows[2], -1., 2.)).is_err());
    assert_eq!(rows[2].evaluate(&values(&rows[2], 4., 2.)).unwrap(), 1.);
    assert!(rows[3].evaluate(&values(&rows[3], 2., -1.)).is_err());
    assert_eq!(rows[5].evaluate(&values(&rows[5], 3., -2.)).unwrap(), 6.);
    assert_eq!(rows[5].evaluate(&values(&rows[5], 3., 2.)).unwrap(), 6.);
}
