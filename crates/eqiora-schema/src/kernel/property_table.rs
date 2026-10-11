//! Exact real one-axis linear tables compiled to the shared scalar calculus.
//!
//! Artifact identity, axis roles and preprocessing belong to the property release.
//! This leaf performs no decoding, storage access or numerical evaluation.
use super::{
    ComparisonOp,
    pure_operator::{
        CalculusBuilder, CalculusNode, ExactRational, PureOperatorDefinition, PureOperatorError,
        PureValueClass,
    },
};
use eqiora_core::{DimExponents, ScalarDomain};

/// Maximum admitted abscissae, bounded by the shared calculus depth budget.
pub const MAX_TABLE_POINTS: usize = 128;

/// Failure to admit a real linear table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableError {
    /// Axis and sample lengths differ or are outside the admitted bounds.
    Shape,
    /// Declared validity is empty, reversed, or not covered by the array.
    Validity,
    /// Abscissae are not strictly increasing.
    Ordering,
    /// A binary64 datum is nonfinite or outside the exact rational representation.
    ExactValue,
    /// Shared calculus construction or exact arithmetic rejected the definition.
    Calculus(PureOperatorError),
}
impl From<PureOperatorError> for TableError {
    fn from(value: PureOperatorError) -> Self {
        Self::Calculus(value)
    }
}

/// Convert binary64 data without rounding or decimal reinterpretation.
///
/// # Errors
/// Rejects nonfinite values and exact values outside the portable rational bounds.
pub fn exact_binary64(value: f64) -> Result<ExactRational, TableError> {
    if !value.is_finite() {
        return Err(TableError::ExactValue);
    }
    if value == 0.0 {
        return Ok(ExactRational::integer(0));
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mut significand = bits & ((1_u64 << 52) - 1);
    let mut power = if exponent == 0 {
        -1074
    } else {
        exponent - 1023 - 52
    };
    if exponent != 0 {
        significand |= 1_u64 << 52;
    }
    let zeros = significand.trailing_zeros();
    significand >>= zeros;
    power += zeros as i32;
    let (magnitude, denominator) = if power >= 0 {
        if power > 63 {
            return Err(TableError::ExactValue);
        }
        let shift = u32::try_from(power).map_err(|_| TableError::ExactValue)?;
        let magnitude = u128::from(significand)
            .checked_shl(shift)
            .ok_or(TableError::ExactValue)?;
        (magnitude, 1)
    } else {
        let shift = power.unsigned_abs();
        let denominator = 1_u64.checked_shl(shift).ok_or(TableError::ExactValue)?;
        (u128::from(significand), denominator)
    };
    if magnitude > (1_u128 << 63) {
        return Err(TableError::ExactValue);
    }
    let signed = if bits >> 63 == 0 {
        magnitude as i128
    } else {
        -(magnitude as i128)
    };
    let numerator = i64::try_from(signed).map_err(|_| TableError::ExactValue)?;
    ExactRational::from_canonical_parts(numerator, denominator).map_err(|_| TableError::ExactValue)
}

/// Generate value (`derivative = false`) or first derivative calculus.
///
/// The value is `y_i + (x - x_i) * ((y_{i+1} - y_i)/(x_{i+1} - x_i))`.
/// Values admit the closed interval. First derivatives admit only open segment
/// interiors: every endpoint and internal knot rejects, including equal-slope
/// knots. Outside-domain evaluation always rejects; no extrapolation.
///
/// # Errors
/// Rejects malformed, unordered or excessive data and exact arithmetic overflow.
pub(crate) fn linear_table(
    axis: &[ExactRational],
    values: &[ExactRational],
    axis_dimension: DimExponents,
    value_dimension: DimExponents,
    derivative: bool,
    validity: [ExactRational; 2],
) -> Result<PureOperatorDefinition, TableError> {
    if axis.len() != values.len() || !(2..=MAX_TABLE_POINTS).contains(&axis.len()) {
        return Err(TableError::Shape);
    }
    let slope_dimension = value_dimension
        .div(axis_dimension)
        .ok_or(PureOperatorError::ResultDimensionOverflow)?;
    let mut slopes = Vec::with_capacity(axis.len() - 1);
    for (x, y) in axis.windows(2).zip(values.windows(2)) {
        if i128::from(x[0].numerator()) * i128::from(x[1].denominator())
            >= i128::from(x[1].numerator()) * i128::from(x[0].denominator())
        {
            return Err(TableError::Ordering);
        }
        let dx = x[1].checked_add(x[0].checked_neg()?)?;
        let dy = y[1].checked_add(y[0].checked_neg()?)?;
        let reciprocal = ExactRational::from_canonical_parts(
            i64::try_from(dx.denominator()).map_err(|_| PureOperatorError::RationalOverflow)?,
            u64::try_from(dx.numerator()).map_err(|_| PureOperatorError::RationalOverflow)?,
        )?;
        slopes.push(dy.checked_mul(reciprocal)?);
    }
    let compare = |a: ExactRational, b: ExactRational| {
        (i128::from(a.numerator()) * i128::from(b.denominator()))
            .cmp(&(i128::from(b.numerator()) * i128::from(a.denominator())))
    };
    if !compare(validity[0], validity[1]).is_lt()
        || compare(axis[0], validity[0]).is_gt()
        || compare(validity[1], axis[axis.len() - 1]).is_gt()
    {
        return Err(TableError::Validity);
    }
    let class = |dimension| {
        PureValueClass::invariant_scalar()
            .with_dimension(dimension)
            .with_scalar_domain(ScalarDomain::Real)
    };
    let mut builder = CalculusBuilder::new(
        [class(axis_dimension)?],
        class(if derivative {
            slope_dimension
        } else {
            value_dimension
        })?,
    )?;
    let input = builder.push(CalculusNode::FormalComponent {
        formal: 0,
        axes: Box::new([]),
    })?;
    let knots = axis
        .iter()
        .map(|&value| {
            builder.push(CalculusNode::Rational {
                value,
                dimension: axis_dimension,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bounds = validity.map(|value| {
        builder.push(CalculusNode::Rational {
            value,
            dimension: axis_dimension,
        })
    });
    let [low, high] = bounds;
    let low = low?;
    let high = high?;
    let lower = builder.push(CalculusNode::Compare(
        if derivative {
            ComparisonOp::Greater
        } else {
            ComparisonOp::GreaterEqual
        },
        input,
        low,
    ))?;
    let upper = builder.push(CalculusNode::Compare(
        if derivative {
            ComparisonOp::Less
        } else {
            ComparisonOp::LessEqual
        },
        input,
        high,
    ))?;
    let mut valid = builder.push(CalculusNode::And(lower, upper))?;
    if derivative {
        for &knot in &knots[1..knots.len() - 1] {
            let away = builder.push(CalculusNode::Compare(ComparisonOp::NotEqual, input, knot))?;
            valid = builder.push(CalculusNode::And(valid, away))?;
        }
    }
    let mut segments = Vec::with_capacity(slopes.len());
    for (i, &slope) in slopes.iter().enumerate() {
        let slope = builder.push(CalculusNode::Rational {
            value: slope,
            dimension: slope_dimension,
        })?;
        let segment = if derivative {
            slope
        } else {
            let negative = builder.push(CalculusNode::Neg(knots[i]))?;
            let offset = builder.push(CalculusNode::Add(input, negative))?;
            let delta = builder.push(CalculusNode::Mul(offset, slope))?;
            let origin = builder.push(CalculusNode::Rational {
                value: values[i],
                dimension: value_dimension,
            })?;
            builder.push(CalculusNode::Add(origin, delta))?
        };
        segments.push(segment);
    }
    let mut result = segments[segments.len() - 1];
    for i in (0..segments.len() - 1).rev() {
        let condition = builder.push(CalculusNode::Compare(
            ComparisonOp::Less,
            input,
            knots[i + 1],
        ))?;
        result = builder.push(CalculusNode::Select {
            condition,
            then_value: segments[i],
            else_value: result,
        })?;
    }
    let root = builder.push(CalculusNode::Require {
        condition: valid,
        value: result,
    })?;
    Ok(builder.finish(root)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    const D: DimExponents = DimExponents::DIMENSIONLESS;
    fn interval(low: i64, high: i64) -> [ExactRational; 2] {
        [ExactRational::integer(low), ExactRational::integer(high)]
    }
    fn ints(values: &[i64]) -> Vec<ExactRational> {
        values.iter().copied().map(ExactRational::integer).collect()
    }
    #[test]
    fn binary_data_is_exact_and_bounded() {
        assert_eq!(
            exact_binary64(0.1).unwrap(),
            ExactRational::from_canonical_parts(3602879701896397, 36028797018963968).unwrap()
        );
        assert_eq!(exact_binary64(-0.0).unwrap(), ExactRational::integer(0));
        assert_eq!(
            exact_binary64(-9223372036854775808.0).unwrap(),
            ExactRational::integer(i64::MIN)
        );
        assert_eq!(
            exact_binary64(2_f64.powi(-63)).unwrap().denominator(),
            1_u64 << 63
        );
        for value in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(1),
            2_f64.powi(-64),
            2_f64.powi(63),
            2_f64.powi(127),
        ] {
            assert_eq!(exact_binary64(value), Err(TableError::ExactValue));
        }
    }
    #[test]
    fn exact_slopes_and_domain_guards() {
        // (0,1), (2,5), (5,2): slopes are exactly 2 and -1.
        let definition = linear_table(
            &ints(&[0, 2, 5]),
            &ints(&[1, 5, 2]),
            D,
            D,
            true,
            interval(0, 5),
        )
        .unwrap();
        let nodes = definition.nodes();
        let CalculusNode::Require { value, .. } = nodes.last().unwrap() else {
            panic!("missing interval guard")
        };
        let CalculusNode::Select {
            condition,
            then_value,
            else_value,
        } = nodes[value.index() as usize]
        else {
            panic!("missing segment selection")
        };
        let CalculusNode::Compare(ComparisonOp::Less, _, knot) = nodes[condition.index() as usize]
        else {
            panic!("selection must use strict right endpoint")
        };
        assert!(
            matches!(nodes[knot.index() as usize], CalculusNode::Rational { value, .. } if value == ExactRational::integer(2))
        );
        assert!(
            matches!(nodes[then_value.index() as usize], CalculusNode::Rational { value, .. } if value == ExactRational::integer(2))
        );
        assert!(
            matches!(nodes[else_value.index() as usize], CalculusNode::Rational { value, .. } if value == ExactRational::integer(-1))
        );
        assert!(nodes.iter().any(|node| matches!(node, CalculusNode::Rational { value, .. } if *value == ExactRational::integer(-1))));
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node, CalculusNode::Compare(ComparisonOp::NotEqual, _, _)))
        );
        assert!(matches!(nodes.last(), Some(CalculusNode::Require { .. })));
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node, CalculusNode::Compare(ComparisonOp::Greater, _, _)))
        );
        assert!(
            nodes
                .iter()
                .any(|node| matches!(node, CalculusNode::Compare(ComparisonOp::Less, _, _)))
        );
        let smooth = linear_table(
            &ints(&[0, 2, 5]),
            &ints(&[1, 5, 11]),
            D,
            D,
            true,
            interval(0, 5),
        )
        .unwrap();
        assert!(
            smooth
                .nodes()
                .iter()
                .any(|node| matches!(node, CalculusNode::Compare(ComparisonOp::NotEqual, _, _)))
        );
    }
    #[test]
    fn malformed_and_excessive_data_reject() {
        for axis in [&[0, 0, 2][..], &[0, 2, 1][..]] {
            assert_eq!(
                linear_table(&ints(axis), &ints(&[1, 2, 3]), D, D, false, interval(0, 2)),
                Err(TableError::Ordering)
            );
        }
        assert_eq!(
            linear_table(&ints(&[0]), &ints(&[1]), D, D, false, interval(0, 2)),
            Err(TableError::Shape)
        );
        assert_eq!(
            linear_table(&ints(&[0, 1]), &ints(&[1]), D, D, false, interval(0, 2)),
            Err(TableError::Shape)
        );
        let axis = (0..=MAX_TABLE_POINTS)
            .map(|x| ExactRational::integer(x as i64))
            .collect::<Vec<_>>();
        assert_eq!(
            linear_table(&axis, &axis, D, D, false, interval(0, 2)),
            Err(TableError::Shape)
        );
        linear_table(
            &axis[..MAX_TABLE_POINTS],
            &axis[..MAX_TABLE_POINTS],
            D,
            D,
            false,
            interval(0, 2),
        )
        .unwrap();
    }
    #[test]
    fn dimensions_and_semantic_identity_are_retained() {
        let length = DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0]).unwrap();
        let value = linear_table(
            &ints(&[0, 2]),
            &ints(&[1, 5]),
            length,
            D,
            false,
            interval(0, 2),
        )
        .unwrap();
        let derivative = linear_table(
            &ints(&[0, 2]),
            &ints(&[1, 5]),
            length,
            D,
            true,
            interval(0, 2),
        )
        .unwrap();
        assert_eq!(derivative.result_rule().dimension(), D.div(length));
        assert_ne!(value.digest(), derivative.digest());
        let changed = linear_table(
            &ints(&[0, 2]),
            &ints(&[1, 6]),
            length,
            D,
            false,
            interval(0, 2),
        )
        .unwrap();
        assert_ne!(value.digest(), changed.digest());
    }
}
