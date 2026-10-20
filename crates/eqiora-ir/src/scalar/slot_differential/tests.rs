//! Independent scalar calculus references and domain falsifiers.
use super::*;
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_schema::kernel::UnaryMathFunction;

fn partial(
    source: impl FnOnce(&mut ScalarInputIrBuilder, ScalarInputValueId) -> ScalarInputValueId,
    point: f64,
    selected: usize,
    order: usize,
) -> Result<f64, Diagnostic> {
    let dimension = DimExponents::DIMENSIONLESS;
    let mut scope = ScalarInputIrBuilder::new();
    let x = scope.scoped_input(dimension)?;
    let _y = scope.scoped_input(dimension)?;
    let mut root = source(&mut scope, x);
    let one = scope.constant(DynQuantity::new(1., dimension))?;
    let mut seeds = [None; 2];
    seeds[selected] = Some(one);
    for _ in 0..order {
        root = scope.formal_directional(root, &seeds, dimension, |_| Ok(()))?;
    }
    let mut destination = ScalarInputIrBuilder::new();
    let x = destination.constant(DynQuantity::new(point, dimension))?;
    let y = destination.constant(DynQuantity::new(0., dimension))?;
    let roots = destination.compose_scoped(
        scope,
        &[root],
        &[(x, dimension), (y, dimension)],
        |_| Ok(()),
    )?;
    Ok(destination.finish(roots)?.evaluate(&[])?[0])
}

#[test]
fn elementary_formal_partials_keep_independent_first_and_second_values() {
    use UnaryMathFunction::*;
    for (function, point, expected) in [
        (Sin, 0., [1., 0.]),
        (Cos, 0., [0., -1.]),
        (Exp, 0., [1., 1.]),
        (Log, 1., [1., -1.]),
        (Sqrt, 4., [0.25, -0.03125]),
    ] {
        for (index, expected) in expected.into_iter().enumerate() {
            let actual = partial(
                |builder, x| builder.unary_math(function, x).unwrap(),
                point,
                0,
                index + 1,
            )
            .unwrap();
            assert_eq!(actual, expected, "{function:?}, order {}", index + 1);
        }
    }
}

#[test]
fn independent_formal_partials_keep_intrinsic_domains_without_demanding_primal_overflow() {
    use UnaryMathFunction::*;
    for (function, point, valid) in [
        (Log, -1., false),
        (Log, 0., false),
        (Log, 1., true),
        (Sqrt, -1., false),
        (Sqrt, 0., true),
        (Exp, 1000., true),
    ] {
        for order in [1, 2] {
            let result = partial(
                |builder, x| builder.unary_math(function, x).unwrap(),
                point,
                1,
                order,
            );
            if valid {
                assert_eq!(result.unwrap(), 0.);
            } else {
                assert_eq!(
                    result.unwrap_err().code(),
                    eqiora_core::diagnostic::codes::EXPRESSION_DOMAIN_VIOLATION
                );
            }
        }
    }
    assert!(partial(|builder, x| builder.unary_math(Sqrt, x).unwrap(), 0., 0, 1).is_err());
    for (point, valid) in [(0., false), (1e-300, true)] {
        let result = partial(
            |builder, x| {
                let zero = builder
                    .constant(DynQuantity::new(0., DimExponents::DIMENSIONLESS))
                    .unwrap();
                builder.div(zero, x).unwrap()
            },
            point,
            1,
            1,
        );
        if valid {
            assert_eq!(result.unwrap(), 0.);
        } else {
            assert!(result.is_err());
        }
    }
}
