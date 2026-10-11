//! Closed input-unit catalog and the numerical conversion boundary.

use eqiora_core::{DimExponents, DynQuantity};
use eqiora_lang::{BinaryOp, Expr, ExprKind};

use crate::dimensions::rational_literal;

mod exact_time;
pub(crate) use exact_time::lower_clock;

/// The compiler's closed multiplicative input-unit vocabulary.
///
/// Source lowering and authoring clients share these symbols and prefix rules.
/// Scale evaluation remains at the compiler quantity boundary.
pub struct InputUnitCatalog;

impl InputUnitCatalog {
    /// Bare input symbols and whether one decimal prefix may be applied.
    pub fn symbols() -> impl Iterator<Item = (&'static str, bool)> {
        COHERENT_UNITS
            .iter()
            .map(|(name, _)| (*name, *name != "kg"))
            .chain([("g", true), ("1", false)])
    }

    /// Admitted case-sensitive prefixes, each applied at most once.
    pub fn prefixes() -> impl Iterator<Item = &'static str> {
        PREFIXES.iter().map(|(prefix, _)| *prefix)
    }
}

const COHERENT_UNITS: &[(&str, [i32; 7])] = &[
    ("kg", [1, 0, 0, 0, 0, 0, 0]),
    ("m", [0, 1, 0, 0, 0, 0, 0]),
    ("s", [0, 0, 1, 0, 0, 0, 0]),
    ("A", [0, 0, 0, 1, 0, 0, 0]),
    ("K", [0, 0, 0, 0, 1, 0, 0]),
    ("mol", [0, 0, 0, 0, 0, 1, 0]),
    ("cd", [0, 0, 0, 0, 0, 0, 1]),
    ("Hz", [0, 0, -1, 0, 0, 0, 0]),
    ("N", [1, 1, -2, 0, 0, 0, 0]),
    ("Pa", [1, -1, -2, 0, 0, 0, 0]),
    ("J", [1, 2, -2, 0, 0, 0, 0]),
    ("W", [1, 2, -3, 0, 0, 0, 0]),
    ("C", [0, 0, 1, 1, 0, 0, 0]),
    ("V", [1, 2, -3, -1, 0, 0, 0]),
    ("Ohm", [1, 2, -3, -2, 0, 0, 0]),
    ("S", [-1, -2, 3, 2, 0, 0, 0]),
    ("F", [-1, -2, 4, 2, 0, 0, 0]),
    ("H", [1, 2, -2, -2, 0, 0, 0]),
    ("Wb", [1, 2, -2, -1, 0, 0, 0]),
    ("T", [1, 0, -2, -1, 0, 0, 0]),
];

const PREFIXES: &[(&str, i32)] = &[
    ("n", -9),
    ("u", -6),
    ("m", -3),
    ("c", -2),
    ("k", 3),
    ("M", 6),
    ("G", 9),
];

pub(crate) fn coherent_dimension(name: &str) -> Option<DimExponents> {
    COHERENT_UNITS.iter().find_map(|(symbol, exponents)| {
        (*symbol == name)
            .then(|| DimExponents::from_integers(*exponents))
            .flatten()
    })
}

struct Unit {
    dimension: DimExponents,
    decimal_power: i32,
}

fn bare_unit(name: &str) -> Option<Unit> {
    if name == "g" {
        return Some(Unit {
            dimension: coherent_dimension("kg")?,
            decimal_power: -3,
        });
    }
    coherent_dimension(name).map(|dimension| Unit {
        dimension,
        decimal_power: 0,
    })
}

fn named_unit(name: &str) -> Option<Unit> {
    if let Some(unit) = bare_unit(name) {
        return Some(unit);
    }
    for &(prefix, power) in PREFIXES {
        if let Some(base) = name.strip_prefix(prefix).filter(|base| *base != "kg")
            && let Some(mut unit) = bare_unit(base)
        {
            unit.decimal_power += power;
            return Some(unit);
        }
    }
    None
}

fn lower_unit(expression: &Expr, depth: usize) -> Result<Unit, &'static str> {
    if depth > 256 {
        return Err("input-unit expression exceeds depth 256");
    }
    match expression.kind() {
        ExprKind::Number(value) if value.to_i64().ok() == Some(1) => Ok(Unit {
            dimension: DimExponents::DIMENSIONLESS,
            decimal_power: 0,
        }),
        ExprKind::Name(name) => named_unit(name).ok_or("unknown input-unit symbol"),
        ExprKind::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            let base = lower_unit(left, depth + 1)?;
            let (n, d) = rational_literal(right)
                .ok_or("input-unit exponent must be an exact bounded rational")?;
            let dimension = base
                .dimension
                .pow(n, d)
                .ok_or("input-unit dimension exceeds exponent bounds")?;
            let power = i64::from(base.decimal_power) * i64::from(n);
            if power % i64::from(d) != 0 {
                return Err("input-unit scale root is not an exact rational");
            }
            let decimal_power = i32::try_from(power / i64::from(d))
                .map_err(|_| "input-unit scale exceeds exponent bounds")?;
            Ok(Unit {
                dimension,
                decimal_power,
            })
        }
        ExprKind::Binary { op, left, right } if matches!(op, BinaryOp::Mul | BinaryOp::Div) => {
            let left = lower_unit(left, depth + 1)?;
            let right = lower_unit(right, depth + 1)?;
            let (dimension, power) = if *op == BinaryOp::Mul {
                (
                    left.dimension.mul(right.dimension),
                    left.decimal_power.checked_add(right.decimal_power),
                )
            } else {
                (
                    left.dimension.div(right.dimension),
                    left.decimal_power.checked_sub(right.decimal_power),
                )
            };
            Ok(Unit {
                dimension: dimension.ok_or("input-unit dimension exceeds exponent bounds")?,
                decimal_power: power.ok_or("input-unit scale exceeds exponent bounds")?,
            })
        }
        _ => Err("invalid input-unit expression"),
    }
}

pub(crate) fn exact_unit(expression: &Expr) -> Result<(DimExponents, i32), &'static str> {
    let unit = lower_unit(expression, 0)?;
    Ok((unit.dimension, unit.decimal_power))
}

pub(crate) fn quantity(
    value: &eqiora_lang::DecimalLiteral,
    expression: &Expr,
) -> Result<DynQuantity, &'static str> {
    let (dimension, decimal_power) = exact_unit(expression)?;
    let exponent = value
        .exponent10()
        .checked_add(i64::from(decimal_power))
        .ok_or("normalized quantity exceeds decimal exponent bounds")?;
    // Preserve the decimal coefficient and compose the exact power of ten.
    // Parsing this final decimal is the only binary64 rounding boundary.
    let normalized = format!(
        "{}{}e{exponent}",
        if value.is_negative() { "-" } else { "" },
        value.coefficient(),
    )
    .parse::<f64>()
    .map_err(|_| "normalized quantity cannot be represented")?;
    if !normalized.is_finite() {
        return Err("normalized quantity must be finite");
    }
    if !value.is_zero() && normalized == 0.0 {
        return Err("nonzero quantity underflows the normalized binary64 range");
    }
    Ok(DynQuantity::new(
        if normalized == 0.0 { 0.0 } else { normalized },
        dimension,
    ))
}

pub(crate) fn normalize_value(value: f64, scale: f64) -> Result<f64, &'static str> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("input-unit scale is outside the finite binary64 range");
    }
    let normalized = value * scale;
    if !normalized.is_finite() {
        return Err("normalized quantity must be finite");
    }
    if value != 0.0 && normalized == 0.0 {
        return Err("nonzero quantity underflows the normalized binary64 range");
    }
    Ok(if normalized == 0.0 { 0.0 } else { normalized })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_lang::{Item, parse};

    fn quantity(value: f64, expression: &Expr) -> Result<DynQuantity, &'static str> {
        super::quantity(
            &eqiora_lang::DecimalLiteral::from_f64(value).unwrap(),
            expression,
        )
    }

    fn unit(source: &str) -> Expr {
        let source = format!("model M() {{ let value = 1 [{source}]; }}");
        let document = parse("unit.eqi", &source).into_document().unwrap();
        let Item::Let(binding) = &document.models()[0].items()[0] else {
            panic!("let")
        };
        let ExprKind::Quantity { unit, .. } = binding.value().kind() else {
            panic!("quantity")
        };
        unit.as_ref().clone()
    }

    #[test]
    fn decimal_quantity_rounds_only_after_exact_unit_composition() {
        use eqiora_graph::Op;
        use eqiora_schema::kernel::KernelNode;

        // 0.1 nm = 1 / 10^10 m. The nearest binary64 is independently
        // specified by its bits, not by multiplying two rounded f64 values.
        let compiled = crate::compile(
            "single-rounding.eqi",
            "model M() { parameter length: m = 0.1[nm]; relation r { length - length = 0; } }",
        )
        .unwrap();
        let value = compiled[0]
            .transaction()
            .ops()
            .iter()
            .find_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Parameter(parameter),
                } => parameter.real_scalar_value(),
                _ => None,
            })
            .unwrap();
        assert_eq!(value.value().to_bits(), 0x3ddb_7cdf_d9d7_bdbb);
    }

    #[test]
    fn scaled_decimal_midpoints_round_to_even() {
        // Exact decimal spellings of (1 + 2^-53) * 1000 and
        // (1 + 3 * 2^-53) * 1000; ms contributes the exact 10^-3 factor.
        for (decimal, expected_bits) in [
            (
                "1000.00000000000011102230246251565404236316680908203125",
                0x3ff0_0000_0000_0000,
            ),
            (
                "1000.00000000000033306690738754696212708950042724609375",
                0x3ff0_0000_0000_0002,
            ),
        ] {
            let value = eqiora_lang::DecimalLiteral::parse(decimal).unwrap();
            let converted = super::quantity(&value, &unit("ms")).unwrap();
            assert_eq!(converted.value().to_bits(), expected_bits);
        }
    }

    #[test]
    fn complex_quantity_components_share_the_exact_decimal_boundary() {
        use eqiora_graph::Op;
        use eqiora_schema::kernel::KernelNode;

        let compiled = crate::compile(
            "complex-units.eqi",
            "model M() { parameter length: complex<m> = math.complex(0.1[nm], -0.1[nm]); relation r { length - length = 0; } }",
        ).unwrap();
        let value = compiled[0]
            .transaction()
            .ops()
            .iter()
            .find_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Parameter(parameter),
                } => Some(parameter.value()),
                _ => None,
            })
            .unwrap();
        let (real, imaginary) = value.component(0).unwrap();
        assert_eq!(real.to_bits(), 0x3ddb_7cdf_d9d7_bdbb);
        assert_eq!(imaginary.to_bits(), 0xbddb_7cdf_d9d7_bdbb);
    }

    #[test]
    fn decimal_quantity_checks_the_final_scaled_range() {
        for (literal, expected) in [
            ("1e-400[km ^ 100]", 1e-100_f64),
            ("1e400[nm ^ 100]", 0.0),
            ("1e400[nm ^ 20]", 1e220),
            ("0[nm ^ 100]", 0.0),
        ] {
            let source = format!("model M() {{ let value = {literal}; }}");
            let document = parse("scaled.eqi", &source).into_document().unwrap();
            let Item::Let(binding) = &document.models()[0].items()[0] else {
                panic!("let")
            };
            let ExprKind::Quantity { value, unit } = binding.value().kind() else {
                panic!("quantity")
            };
            let converted = super::quantity(value, unit);
            if literal == "1e400[nm ^ 100]" {
                assert!(converted.is_err(), "nonzero final underflow must reject");
            } else {
                assert_eq!(converted.unwrap().value().to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn catalog_scales_normalize_once_and_roots_remain_exact() {
        for (value, symbol, expected, dimension) in [
            (10.0, "ms", 0.01, "s"),
            (1.0, "kOhm", 1000.0, "Ohm"),
            (210.0, "GPa", 210_000_000_000.0, "Pa"),
            (1.0, "uF", 0.000001, "F"),
            (1.0, "mg", 0.000001, "kg"),
            (1.0, "(mm ^ 2) ^ (1 / 2)", 0.001, "m"),
        ] {
            let converted = quantity(value, &unit(symbol)).unwrap();
            assert_eq!(converted.value(), expected, "{symbol}");
            assert_eq!(converted.dim(), coherent_dimension(dimension).unwrap());
        }
        assert_eq!(quantity(1.0, &unit("km ^ (23 / 3)")).unwrap().value(), 1e23);
        let inverse_root_time = quantity(1.0, &unit("Hz ^ (-1 / 2)")).unwrap();
        assert_eq!(inverse_root_time.dim().exponents()[2], (1, 2));
    }

    #[test]
    fn source_quantities_reach_parameters_defaults_bindings_and_relations() {
        use eqiora_graph::Op;
        use eqiora_schema::kernel::KernelNode;

        let source = r#"
dimension Duration = s;
component Delay(parameter duration: Duration = 10 [ms]) {
  
  relation balance { duration - 0.01 [s] = 0; }
}
model Quantities() {
  parameter duration: Duration = -10[ms];
  let ms: m = 3[m];
  let positive: Duration = 10 [ms];
  variable elapsed: s; initial { elapsed = 0; }
  relation balance { elapsed - positive = 0; }
  instance defaulted: Delay();
  instance bound: Delay(duration = 10 [ms]);
}
"#;
        let compiled = crate::compile("quantities.eqi", source).unwrap();
        let values: Vec<_> = compiled[0]
            .transaction()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::DefineKernelNode {
                    node: KernelNode::Parameter(parameter),
                } => parameter.real_scalar_value(),
                _ => None,
            })
            .collect();
        assert!(values.iter().any(|value| value.value() == -0.01));
        for value in values {
            assert_eq!(value.dim(), coherent_dimension("s").unwrap());
            assert_eq!(value.value().abs(), 0.01);
        }
        for wrong in [
            source.replace("-10[ms]", "-10[m]"),
            source.replace("= 10 [ms];", "= 10 [m];"),
            source.replace("duration = 10 [ms]", "duration = 10 [m]"),
            source.replace("0.01 [s]", "0.01 [m]"),
            source.replace("10 [ms]", "10 [Duration]"),
        ] {
            assert!(crate::compile("wrong.eqi", &wrong).is_err(), "{wrong}");
        }
    }

    #[test]
    fn declaration_units_preserve_values_across_flat_and_component_initializers() {
        use eqiora_graph::Op;
        use eqiora_schema::kernel::KernelNode;

        let density = DimExponents::from_integers([1, -3, 0, 0, 0, 0, 0]).unwrap();
        for literal in ["1000", "1000[kg / m ^ 3]", "1[g / cm ^ 3]"] {
            let initial = if literal == "1000" {
                "1000[kg / m ^ 3]"
            } else {
                literal
            };
            let source = format!(
                "component C(parameter density: kg / m ^ 3 = {literal}) {{
                    
                    relation r {{ density - density = 0; }}
                }}
                model M() {{
                    parameter p: kg / m ^ 3 = {literal};
                    variable f: kg / m ^ 3; initial {{ f = {initial}; }}
                    let alias: kg / m ^ 3 = {literal};
                    instance defaulted: C();
                    instance bound: C(density = alias);
                    relation r {{ f - p = 0; }}
                }}"
            );
            let compiled = crate::compile("density.eqi", &source).unwrap();
            let values: Vec<_> = compiled[0]
                .transaction()
                .ops()
                .iter()
                .filter_map(|op| match op {
                    Op::DefineKernelNode {
                        node: KernelNode::Parameter(parameter),
                    } => parameter.real_scalar_value(),
                    _ => None,
                })
                .collect();
            // Parameter defaults remain values; initial equations own their constants.
            assert_eq!(values.len(), 1);
            for value in values {
                assert_eq!(value.dim(), density);
                assert_eq!(value.value(), 1000.0, "{literal}");
            }
            let component_constants: Vec<_> = compiled[0]
                .transaction()
                .ops()
                .iter()
                .filter_map(|op| match op {
                    Op::DefineKernelNode {
                        node: KernelNode::Relation(relation),
                    } => Some(relation),
                    _ => None,
                })
                .flat_map(|relation| relation.expression().nodes())
                .filter_map(|node| match node {
                    eqiora_schema::kernel::ExprNode::Constant(value)
                        if value.component(0).unwrap().0 != 0.0 =>
                    {
                        Some(value)
                    }
                    _ => None,
                })
                .collect();
            assert!(!component_constants.is_empty());
            for value in component_constants {
                assert_eq!(value.value_type().dimension(), density);
                assert_eq!(value.component(0).unwrap().0, 1000.0, "{literal}");
            }
            assert!(crate::compile("wrong-density.eqi", &source.replace(literal, "1[s]")).is_err());
        }
    }

    #[test]
    fn unit_errors_do_not_guess_symbols_or_approximate_scale_roots() {
        for symbol in [
            "mkg",
            "kkOhm",
            "KOhm",
            "Speed",
            "ms ^ (1 / 2)",
            "km ^ 2147483647",
        ] {
            assert!(quantity(1.0, &unit(symbol)).is_err(), "{symbol}");
        }
        assert!(quantity(f64::MAX, &unit("km")).is_err());
        assert!(
            parse("invalid-unit.eqi", "model M() { let value = 1 [µF]; }")
                .into_document()
                .is_err()
        );
        assert!(quantity(1.0, &unit("nm ^ 100")).is_err());
        assert_eq!(quantity(-0.0, &unit("ms")).unwrap().value().to_bits(), 0);
    }
}
