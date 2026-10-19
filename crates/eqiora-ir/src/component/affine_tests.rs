//! Independent derivatives of retained affine coefficients, not sampled unknowns.
use super::*;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueType, entity::kinds};
use eqiora_schema::kernel::{ExprDagBuilder, UnaryMathFunction, typing::RootContract};

#[test]
fn affine_directions_retain_range_until_coefficients_and_offsets_are_requested() {
    for domain in [ScalarDomain::Real, ScalarDomain::Complex] {
        for exponent in [-700, -400, 400, 700] {
            let scale = 2f64.powi(exponent);
            let field = SymbolRef::Field(Id::<kinds::Field>::new());
            let parameter = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
            let ty = ValueType::scalar(domain, DimExponents::DIMENSIONLESS).unwrap();
            let mut dag = ExprDagBuilder::new();
            let u = dag.symbol(field).unwrap();
            let p = dag.symbol(parameter).unwrap();
            let one = dag
                .constant(DynQuantity::new(1., DimExponents::DIMENSIONLESS))
                .unwrap();
            let factor = dag
                .constant(DynQuantity::new(scale, DimExponents::DIMENSIONLESS))
                .unwrap();
            let reciprocal = dag.div(one, p).unwrap();
            let coefficient = dag.mul(factor, reciprocal).unwrap();
            let product = dag.mul(coefficient, u).unwrap();
            let root = dag.add(product, coefficient).unwrap();
            let typed = TypedResidual::infer(
                dag.finish([root, reciprocal]).unwrap(),
                None,
                |_| None,
                RootContract::ComponentwiseResidual,
                |_| Ok::<_, ()>(ExpressionType::<()>::new(ty.clone(), None)),
            )
            .unwrap();
            let lowered = ComponentScalarization::lower(&typed).unwrap();
            let selected = ScalarSymbolCoordinate::for_value(field, &ty).unwrap();
            let bindings = ScalarSymbolCoordinate::for_value(parameter, &ty)
                .unwrap()
                .into_iter()
                .enumerate()
                .map(|(i, c)| {
                    (
                        c,
                        if i == 0 { scale } else { 0. },
                        Some(if i == 0 { 1. } else { 0. }),
                    )
                })
                .collect::<Vec<_>>();
            // f(p,u)=(s/p)(u+1), so at p=s: f=u+1, df/dp=-(u+1)/s.
            // The intermediate derivative -1/p^2 overflows/underflows at ±700.
            for (i, row) in lowered.rows()[..selected.len()].iter().enumerate() {
                let (primal, tangent) = row.bind_affine_direction(&selected, &bindings).unwrap();
                for j in 0..selected.len() {
                    assert_eq!(primal.coefficients()[j], if i == j { 1. } else { 0. });
                    assert_eq!(
                        tangent.coefficients()[j] * scale,
                        if i == j { -1. } else { 0. }
                    );
                }
                assert_eq!(primal.offsets(), &[if i == 0 { 1. } else { 0. }]);
                assert_eq!(tangent.offsets()[0] * scale, if i == 0 { -1. } else { 0. });
            }
            if exponent == -700 {
                // Exposing that intermediate as the requested output must still fail.
                let error = lowered.rows()[selected.len()]
                    .bind_affine_direction(&[], &bindings)
                    .unwrap_err();
                assert!(error.message().contains("affine output"), "{error:?}");
            }
        }
    }
}

#[test]
fn affine_power_and_math_composition_retains_intermediate_directions() {
    for exponent in [-700, 700] {
        let scale = 2f64.powi(exponent);
        let parameter = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
        let ty = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        let mut dag = ExprDagBuilder::new();
        let p = dag.symbol(parameter).unwrap();
        let reciprocal = dag.powi(p, -1).unwrap();
        let log = dag.unary_math(UnaryMathFunction::Log, reciprocal).unwrap();
        let factor = dag
            .constant(DynQuantity::new(scale, DimExponents::DIMENSIONLESS))
            .unwrap();
        let root = dag.mul(factor, log).unwrap();
        let typed = TypedResidual::infer(
            dag.finish([root]).unwrap(),
            None,
            |_| None,
            RootContract::ComponentwiseResidual,
            |_| Ok::<_, ()>(ExpressionType::<()>::new(ty.clone(), None)),
        )
        .unwrap();
        let lowered = ComponentScalarization::lower(&typed).unwrap();
        let p = ScalarSymbolCoordinate::for_value(parameter, &ty)
            .unwrap()
            .remove(0);
        // f(p)=s log(1/p), f'(s)=-1. The original primal remains finite.
        let (_, tangent) = lowered.rows()[0]
            .bind_affine_direction(&[], &[(p, scale, Some(1.))])
            .unwrap();
        assert_eq!(tangent.offsets(), &[-1.]);
    }
}

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
            .bind_affine_direction(&selected, &[(p.clone(), x, Some(-3.))])
            .unwrap();
        assert_eq!(primal.coefficients(), &[x * x + x.sin()]);
        assert_eq!(primal.offsets(), &[x * x * x]);
        assert!((tangent.coefficients()[0] + 3. * (2. * x + x.cos())).abs() < 1e-14);
        assert_eq!(tangent.offsets(), &[-9. * x * x]);
        // At x=0 the primal multiplier vanishes, but its direction does not:
        // a nonlinear unknown dependence cannot disappear from the derivative.
        assert!(
            lowered.rows()[1]
                .bind_affine_direction(&selected, &[(p.clone(), x, Some(-3.))])
                .is_err()
        );
    }
    for bindings in [
        vec![(p.clone(), 1., Some(f64::NAN))],
        vec![(p.clone(), 1., Some(2.)), (p.clone(), 1., Some(2.))],
        vec![(selected[0].clone(), 1., Some(2.))],
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
                &[(p[0].clone(), 1., Some(2.)), (p[1].clone(), 1., Some(-1.))],
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
        let (_, tangent) = row
            .bind_affine_direction(&selected, &[(p.clone(), 0., None)])
            .unwrap();
        assert_eq!(tangent.coefficients(), &[0.]);
        assert_eq!(tangent.offsets(), &[0.]);
        assert!(
            row.bind_affine_direction(&selected, &[(p.clone(), -1., None)])
                .is_err()
        );
        for direction in [0., 1.] {
            assert!(
                row.bind_affine_direction(&selected, &[(p.clone(), 0., Some(direction))])
                    .is_err()
            );
        }
        assert!(
            row.bind_affine_direction(&selected, &[(p.clone(), -1., Some(1.))])
                .is_err()
        );
        assert!(
            row.bind_affine_direction(&selected, &[(p.clone(), 4., Some(1.))])
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
            .bind_affine_direction(&selected, &[(p, 0., Some(1.))])
            .is_err()
    );
}

#[test]
fn retained_primal_dependencies_do_not_invent_gradient_value_columns() {
    let field = SymbolRef::Field(Id::<kinds::Field>::new());
    let parameter = SymbolRef::Parameter(Id::<kinds::Parameter>::new());
    let ty = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
    let support = eqiora_schema::kernel::typing::SpatialSupport::Volume {
        domain: (),
        dimensions: 1,
    };
    let mut dag = ExprDagBuilder::new();
    let u = dag.symbol(field).unwrap();
    let p = dag.symbol(parameter).unwrap();
    let sqrt = dag.unary_math(UnaryMathFunction::Sqrt, p).unwrap();
    let product = dag.mul(sqrt, u).unwrap();
    let gradient = dag.gradient(product).unwrap();
    let typed = TypedResidual::infer(
        dag.finish([gradient]).unwrap(),
        Some(support.clone()),
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            Ok::<_, ()>(ExpressionType::new(
                ty.clone(),
                (symbol == field).then(|| support.clone()),
            ))
        },
    )
    .unwrap();
    let lowered =
        ComponentScalarization::lower_spatial_selected(&typed, &[gradient], None).unwrap();
    let row = &lowered.rows()[0];
    let value = ScalarSymbolCoordinate::for_value(field, &ty).unwrap();
    // d_x(sqrt(p)*u) = sqrt(p)*d_x(u), with p fixed in space.
    // The retained primal sqrt(p)*u has a variable coefficient in u, but
    // the derivative result has exactly zero coefficient in that value slot.
    assert_eq!(
        row.constant_coordinate_jacobian(&value)
            .unwrap()
            .coefficients(),
        &[0.]
    );
    let gradient = row
        .symbols()
        .iter()
        .find(|c| c.symbol() == field && c.spatial_derivative_axes() == [0])
        .unwrap();
    assert!(
        row.constant_coordinate_jacobian(std::slice::from_ref(gradient))
            .is_err()
    );
    for p in [0., 4., -1.] {
        let values = row
            .symbols()
            .iter()
            .map(|c| {
                if c.symbol() == parameter {
                    p
                } else if c.spatial_derivative_axes().is_empty() {
                    3.
                } else {
                    2.
                }
            })
            .collect::<Vec<_>>();
        if p < 0. {
            assert!(row.evaluate(&values).is_err());
        } else {
            assert_eq!(row.evaluate(&values).unwrap(), 2. * p.sqrt());
        }
    }
}
