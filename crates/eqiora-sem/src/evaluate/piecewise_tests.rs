use super::*;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, Id};
use eqiora_schema::kernel::{ComparisonOp, ExprDagBuilder, UnaryMathFunction};

#[test]
fn reference_select_and_require_do_not_evaluate_inactive_domains() {
    let owner = Id::<kinds::Relation>::new().erase();
    let parameter = Id::<kinds::Parameter>::new();
    let mut builder = ExprDagBuilder::new();
    let x = builder.symbol(SymbolRef::Parameter(parameter)).unwrap();
    let zero = builder
        .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
        .unwrap();
    let positive = builder.compare(ComparisonOp::Greater, x, zero).unwrap();
    let root = builder.unary_math(UnaryMathFunction::Sqrt, x).unwrap();
    let guarded = builder.select(positive, root, zero).unwrap();
    let false_condition = builder.constant(ValueLiteral::boolean(false)).unwrap();
    let failed = builder.require(false_condition, root).unwrap();
    let lazy = builder.select(positive, guarded, zero).unwrap();
    let hidden = builder.select(false_condition, failed, lazy).unwrap();
    let dag = builder.finish([hidden]).unwrap();
    for (input, expected) in [(-1., 0.), (4., 2.), (0., 0.)] {
        let values = evaluate_expression(owner, &dag, &mut |symbol| {
            (symbol == SymbolRef::Parameter(parameter)).then(|| {
                ValueLiteral::from_real(
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("valid scalar type"),
                    input,
                )
                .unwrap()
            })
        })
        .unwrap();
        assert_eq!(values[0].real_scalar_value().unwrap().value(), expected);
    }
    let error = evaluate_selected(owner, &dag, &[failed], &mut |_| None).unwrap_err();
    assert_eq!(error.code(), codes::EXPRESSION_DOMAIN_VIOLATION);
    assert!(error.message().contains("domain condition"));
}
