use super::*;
use eqiora_core::{Id, entity::kinds};
use eqiora_schema::kernel::{ExprDagBuilder, UnaryMathFunction};

#[test]
fn independent_division_preserves_degree_and_complex_antilinearity_rejects() {
    let field = Id::<kinds::Field>::new();
    let owner = Id::<kinds::Relation>::new().erase();
    for (trial_numerator, trial_denominator, conjugated, complex, expected) in [
        (false, false, false, false, Some(0)),
        (true, false, false, false, Some(1)),
        (false, true, false, false, Some(2)),
        (true, true, false, false, Some(2)),
        (true, false, true, false, Some(1)),
        (true, false, true, true, None),
    ] {
        let mut builder = ExprDagBuilder::new();
        let trial = builder.symbol(SymbolRef::Field(field)).unwrap();
        let coefficient = builder
            .symbol(SymbolRef::Parameter(Id::<kinds::Parameter>::new()))
            .unwrap();
        let mut numerator = if trial_numerator { trial } else { coefficient };
        if conjugated {
            numerator = builder
                .unary_math(UnaryMathFunction::Conj, numerator)
                .unwrap();
        }
        let denominator = if trial_denominator {
            trial
        } else {
            coefficient
        };
        let root = builder.div(numerator, denominator).unwrap();
        let dag = builder.finish([root]).unwrap();
        let degree = value_degree(&dag, root, field.erase(), owner, false, complex);
        match expected {
            Some(expected) => assert_eq!(degree.unwrap(), expected),
            None => assert!(degree.unwrap_err().message().contains("antilinear")),
        }
    }
}
