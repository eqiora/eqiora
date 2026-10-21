//! Independent scalar calculus references and domain falsifiers.
use super::*;
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_schema::kernel::UnaryMathFunction;

fn held_branch(point: [f64; 3], selected: Option<usize>, order: usize) -> Result<f64, Diagnostic> {
    let dimension = DimExponents::DIMENSIONLESS;
    let mut scope = ScalarInputIrBuilder::new();
    let x = scope.scoped_input(dimension)?;
    let selector = scope.scoped_input(dimension)?;
    let _independent = scope.scoped_input(dimension)?;
    let zero = scope.constant(DynQuantity::new(0., dimension))?;
    let condition = scope.compare(eqiora_schema::kernel::ComparisonOp::Greater, selector, zero)?;
    let positive = scope.unary_math(UnaryMathFunction::Log, x)?;
    let negative = scope.neg(x)?;
    let negative = scope.unary_math(UnaryMathFunction::Log, negative)?;
    let mut root = scope.select(condition, positive, negative)?;
    if let Some(selected) = selected {
        let one = scope.constant(DynQuantity::new(1., dimension))?;
        let mut seeds = [None; 3];
        seeds[selected] = Some(one);
        for _ in 0..order {
            root = scope.formal_directional(root, &seeds, dimension, |_| Ok(()))?;
        }
    }
    let mut destination = ScalarInputIrBuilder::new();
    let inputs = point
        .into_iter()
        .map(|value| {
            destination
                .constant(DynQuantity::new(value, dimension))
                .map(|id| (id, dimension))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let roots = destination.compose_scoped(scope, &[root], &inputs, |_| Ok(()))?;
    Ok(destination.finish(roots)?.evaluate(&[])?[0])
}

#[test]
fn held_branch_value_demands_only_the_selected_logarithm() {
    for (x, selector) in [(1., 1.), (-1., -1.), (-1., 0.)] {
        assert_eq!(held_branch([x, selector, 0.], None, 0).unwrap(), 0.);
    }
    for (x, selector) in [(-1., 1.), (1., -1.), (0., 1.)] {
        assert!(held_branch([x, selector, 0.], None, 0).is_err());
    }
}

#[test]
fn held_branch_formal_partials_preserve_selected_domains_and_analytic_values() {
    // With the selector held fixed, both logarithms have slope 1/x and
    // curvature -1/x² on their respective domains. The other arm is invalid.
    for (x, selector) in [(2., 1.), (-2., -1.), (-2., 0.)] {
        for (order, expected) in [(1, 1. / x), (2, -0.25)] {
            assert_eq!(
                held_branch([x, selector, 0.], Some(0), order).unwrap(),
                expected
            );
        }
    }
}

#[test]
fn held_branch_zero_partials_keep_only_demanded_source_validity() {
    for order in [1, 2] {
        for (x, selector) in [(2., 1.), (-2., -1.), (-2., 0.)] {
            assert_eq!(held_branch([x, selector, 0.], Some(2), order).unwrap(), 0.);
        }
        for (x, selector) in [(-2., 1.), (2., -1.), (0., 1.)] {
            let error = held_branch([x, selector, 0.], Some(2), order).unwrap_err();
            assert_eq!(
                error.code(),
                eqiora_core::diagnostic::codes::EXPRESSION_DOMAIN_VIOLATION
            );
        }
    }
}

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
