//! Independent derivatives of retained affine coefficients, not sampled unknowns.
use super::*;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueType, entity::kinds};
use eqiora_schema::kernel::{ExprDagBuilder, UnaryMathFunction, typing::RootContract};

#[test]
fn affine_direction_differentiates_coefficients_and_offsets_in_one_proof() {
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let parameter = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let ty = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let mut dag = ExprDagBuilder::new();
    let u = dag.symbol(field).unwrap();
    let p = dag.symbol(parameter).unwrap();
    let square = dag.powi(p, 2).unwrap();
    let sine = dag.unary_math(UnaryMathFunction::Sin, p).unwrap();
    let coefficient = dag.add(square, sine).unwrap();
    let product = dag.mul(coefficient, u).unwrap();
    let offset = dag.powi(p, 3).unwrap();
    let root = dag.add(product, offset).unwrap();
    let nonlinear = dag.mul(product, u).unwrap();
    let typed = TypedResidual::infer(
        dag.finish([root, nonlinear]).unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(ExpressionType::<()>::new(ty.clone(), None)),
    )
    .unwrap();
    let lowered = ComponentScalarization::lower(&typed).unwrap();
    let selected = ScalarSymbolCoordinate::for_value(field, &ty).unwrap();
    let p = ScalarSymbolCoordinate::for_value(parameter, &ty)
        .unwrap()
        .remove(0);
    for x in [0., 0.5, 2.] {
        let (primal, tangent) = lowered.rows()[0]
            .bind_affine_direction(&selected, &[(p.clone(), x, -3.)])
            .unwrap();
        assert_eq!(primal.coefficients(), &[x * x + x.sin()]);
        assert_eq!(primal.offsets(), &[x * x * x]);
        assert!((tangent.coefficients()[0] + 3. * (2. * x + x.cos())).abs() < 1e-14);
        assert_eq!(tangent.offsets(), &[-9. * x * x]);
        // At x=0 the primal multiplier vanishes, but its direction does not:
        // a nonlinear unknown dependence cannot disappear from the derivative.
        assert!(
            lowered.rows()[1]
                .bind_affine_direction(&selected, &[(p.clone(), x, -3.)])
                .is_err()
        );
    }
    for bindings in [
        vec![(p.clone(), 1., f64::NAN)],
        vec![(p.clone(), 1., 2.), (p.clone(), 1., 2.)],
        vec![(selected[0].clone(), 1., 2.)],
    ] {
        assert!(
            lowered.rows()[0]
                .bind_affine_direction(&selected, &bindings)
                .is_err()
        );
    }
}

#[test]
fn affine_complex_division_retains_all_real_coordinate_directions() {
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let parameter = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap();
    let mut dag = ExprDagBuilder::new();
    let u = dag.symbol(field).unwrap();
    let p = dag.symbol(parameter).unwrap();
    let root = dag.div(u, p).unwrap();
    let typed = TypedResidual::infer(
        dag.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(ExpressionType::<()>::new(ty.clone(), None)),
    )
    .unwrap();
    let lowered = ComponentScalarization::lower(&typed).unwrap();
    let selected = ScalarSymbolCoordinate::for_value(field, &ty).unwrap();
    let p = ScalarSymbolCoordinate::for_value(parameter, &ty).unwrap();
    // p=1+i, dp=2-i: d(1/p)=-dp/p^2=(1+2i)/2.
    for (row, expected) in lowered.rows().iter().zip([[0.5, -1.], [1., 0.5]]) {
        let (_, tangent) = row
            .bind_affine_direction(
                &selected,
                &[(p[0].clone(), 1., 2.), (p[1].clone(), 1., -1.)],
            )
            .unwrap();
        assert_eq!(tangent.coefficients(), expected);
        assert_eq!(tangent.offsets(), &[0.]);
    }
}

#[test]
fn affine_directions_preserve_primal_domains_and_reject_singular_derivatives() {
    use eqiora_schema::kernel::ComparisonOp;
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let parameter = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let ty = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let mut dag = ExprDagBuilder::new();
    let u = dag.symbol(field).unwrap();
    let p = dag.symbol(parameter).unwrap();
    let sqrt = dag.unary_math(UnaryMathFunction::Sqrt, p).unwrap();
    let root = dag.mul(sqrt, u).unwrap();
    let zero = dag
        .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
        .unwrap();
    let condition = dag.compare(ComparisonOp::GreaterEqual, p, zero).unwrap();
    let guarded = dag.require(condition, u).unwrap();
    let square = dag.powi(p, 2).unwrap();
    let isolated = dag.compare(ComparisonOp::LessEqual, square, zero).unwrap();
    let isolated = dag.require(isolated, u).unwrap();
    let typed = TypedResidual::infer(
        dag.finish([root, guarded, isolated]).unwrap(),
        None,
        |_| None,
        RootContract::ComponentwiseResidual,
        |_| Ok::<_, ()>(ExpressionType::<()>::new(ty.clone(), None)),
    )
    .unwrap();
    let lowered = ComponentScalarization::lower(&typed).unwrap();
    let selected = ScalarSymbolCoordinate::for_value(field, &ty).unwrap();
    let p = ScalarSymbolCoordinate::for_value(parameter, &ty)
        .unwrap()
        .remove(0);
    for row in &lowered.rows()[..2] {
        assert!(row.bind_affine(&selected, &[(p.clone(), 0.)]).is_ok());
        for direction in [0., 1.] {
            assert!(
                row.bind_affine_direction(&selected, &[(p.clone(), 0., direction)])
                    .is_err()
            );
        }
        assert!(
            row.bind_affine_direction(&selected, &[(p.clone(), -1., 1.)])
                .is_err()
        );
        assert!(
            row.bind_affine_direction(&selected, &[(p.clone(), 4., 1.)])
                .is_ok()
        );
    }
    // p^2 <= 0 is satisfied only at p=0; d(p^2)=0 cannot certify a neighborhood.
    assert!(
        lowered.rows()[2]
            .bind_affine(&selected, &[(p.clone(), 0.)])
            .is_ok()
    );
    assert!(
        lowered.rows()[2]
            .bind_affine_direction(&selected, &[(p, 0., 1.)])
            .is_err()
    );
}
