//! Branch selection cannot change a structurally constant selected Jacobian.
use eqiora_core::{DimExponents, DynQuantity, Id, entity::kinds};
use eqiora_ir::{ScalarOperatorIr, SymbolicLinearityFailure};
use eqiora_schema::kernel::{ComparisonOp, ExprDagBuilder, SymbolRef};

fn fixture(
    rate_in_condition: bool,
    differing_coefficients: bool,
) -> (ScalarOperatorIr, SymbolRef, SymbolRef) {
    let x = SymbolRef::Field(Id::<kinds::Field>::new());
    let rate = SymbolRef::Derivative(Id::<kinds::Field>::new(), std::num::NonZeroU32::MIN);
    let mut b = ExprDagBuilder::new();
    let state = b.symbol(x).unwrap();
    let derivative = b.symbol(rate).unwrap();
    let zero = b
        .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
        .unwrap();
    let condition = b
        .compare(
            ComparisonOp::Greater,
            if rate_in_condition { derivative } else { state },
            zero,
        )
        .unwrap();
    let square = b.mul(state, state).unwrap();
    let positive = b.add(derivative, square).unwrap();
    let negative = if differing_coefficients {
        b.sub(square, derivative).unwrap()
    } else {
        b.sub(derivative, square).unwrap()
    };
    let selected = b.select(condition, positive, negative).unwrap();
    let domain = b.compare(ComparisonOp::NotEqual, state, zero).unwrap();
    let root = b.require(domain, selected).unwrap();
    (
        ScalarOperatorIr::lower(&b.finish([root]).unwrap()).unwrap(),
        x,
        rate,
    )
}

#[test]
fn independent_piecewise_forcing_preserves_the_rate_coefficient() {
    let (ir, x, rate) = fixture(false, false);
    assert_eq!(
        ir.constant_symbol_jacobian(&[rate]).unwrap().coefficients(),
        [1.]
    );
    // R = rate + x² for x>0 and rate-x² for x<0. Bind x, never rate.
    for (x_value, offset) in [(2., 4.), (-3., -9.)] {
        let affine = ir.bind_affine(&[rate], &[(x, x_value)]).unwrap();
        assert_eq!(affine.coefficients(), [1.]);
        assert_eq!(affine.offsets(), [offset]);
    }
    assert!(ir.bind_affine(&[rate], &[(x, 0.)]).is_err());
}

#[test]
fn variable_branch_coefficients_and_rate_dependent_switches_reject() {
    let (ir, _, rate) = fixture(false, true);
    assert!(matches!(
        ir.constant_symbol_jacobian(&[rate]),
        Err(SymbolicLinearityFailure::VariableCoefficient { .. })
    ));
    let (ir, _, rate) = fixture(true, false);
    assert!(matches!(
        ir.constant_symbol_jacobian(&[rate]),
        Err(SymbolicLinearityFailure::Nonlinear { .. })
    ));
}

#[test]
fn boolean_predicates_are_not_real_affine_outputs() {
    let mut b = ExprDagBuilder::new();
    let root = b
        .constant(eqiora_core::ValueLiteral::boolean(true))
        .unwrap();
    let ir = ScalarOperatorIr::lower(&b.finish([root]).unwrap()).unwrap();
    assert!(ir.constant_symbol_jacobian(&[]).is_err());
}

#[test]
fn typed_scalar_execution_reuses_demand_and_active_derivative_rules() {
    use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent};
    use eqiora_schema::kernel::{
        UnaryMathFunction,
        typing::{ExpressionType, RootContract, TypedResidual},
    };
    let x = SymbolRef::Field(Id::<kinds::Field>::new());
    let rate = SymbolRef::Derivative(Id::<kinds::Field>::new(), std::num::NonZeroU32::MIN);
    let mut b = ExprDagBuilder::new();
    let state = b.symbol(x).unwrap();
    let derivative = b.symbol(rate).unwrap();
    let zero = b
        .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
        .unwrap();
    let condition = b.compare(ComparisonOp::Greater, state, zero).unwrap();
    let positive = b.mul(state, state).unwrap();
    let negated = b.neg(state).unwrap();
    let negative = b.unary_math(UnaryMathFunction::Sqrt, negated).unwrap();
    let selected = b.select(condition, positive, negative).unwrap();
    let root = b.add(derivative, selected).unwrap();
    let typed = TypedResidual::<()>::infer(
        b.finish([root]).unwrap(),
        None,
        |_| None,
        RootContract::InitialResiduals,
        |_| Ok::<_, ()>(ExpressionType::scalar(DimExponents::DIMENSIONLESS, None)),
    )
    .unwrap();
    let ir = ScalarOperatorIr::lower_typed_scalar(&typed).unwrap();
    assert_eq!(
        ir.constant_symbol_jacobian(&[rate]).unwrap().coefficients(),
        [1.]
    );
    for (value, forcing, tangent) in [(2., 4., 4.), (-4., 2., -0.25)] {
        let point = ir
            .symbols()
            .iter()
            .map(|symbol| if *symbol == x { value } else { 3. })
            .collect::<Vec<_>>();
        assert_eq!(ir.evaluate(&point).unwrap(), [3. + forcing]);
        let roles = ir
            .symbols()
            .iter()
            .map(|symbol| {
                if *symbol == x {
                    DifferentiationRole::Unknown
                } else {
                    DifferentiationRole::Frozen
                }
            })
            .collect::<Vec<_>>();
        let linearized = ir.linearize(&point, &roles).unwrap();
        let mut actual = [0.];
        linearized
            .jvp(RelationTangent::Unknown(&[1.]), &mut actual)
            .unwrap();
        assert_eq!(actual, [tangent]);
    }
    // The forcing is defined at zero. Its x derivative is not; a rate derivative
    // holds x fixed and must not demand that unrelated derivative.
    let point = ir
        .symbols()
        .iter()
        .map(|symbol| if *symbol == x { 0. } else { 3. })
        .collect::<Vec<_>>();
    assert_eq!(ir.evaluate(&point).unwrap(), [3.]);
    for active in [x, rate] {
        let roles = ir
            .symbols()
            .iter()
            .map(|symbol| {
                if *symbol == active {
                    DifferentiationRole::Unknown
                } else {
                    DifferentiationRole::Frozen
                }
            })
            .collect::<Vec<_>>();
        if active == x {
            assert!(ir.linearize(&point, &roles).is_err());
        } else {
            let mut actual = [0.];
            ir.linearize(&point, &roles)
                .unwrap()
                .jvp(RelationTangent::Unknown(&[1.]), &mut actual)
                .unwrap();
            assert_eq!(actual, [1.]);
        }
    }
}
