//! Real one-axis linear tables compiled to the shared scalar calculus.
//!
//! Artifact identity, axis roles, and preprocessing belong to the property release.
//! This module does no decoding, storage access, or numerical evaluation.
use super::{
    ComparisonOp,
    pure_operator::{
        CalculusBuilder, CalculusNode, ExactRational, PureOperatorDefinition, PureOperatorError,
        PureValueClass,
    },
};
use eqiora_core::{DimExponents, ScalarDomain};

/// Maximum number of abscissae, limited by the shared calculus depth budget.
pub const MAX_TABLE_POINTS: usize = 128;

/// Failure to admit a real linear table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableError {
    /// Axis and sample lengths differ or are outside the allowed bounds.
    Shape,
    /// Declared validity is empty, reversed, or not covered by the array.
    Validity,
    /// Abscissae are not strictly increasing.
    Ordering,
    /// Shared calculus construction or exact arithmetic rejected the table.
    Calculus(PureOperatorError),
}
impl From<PureOperatorError> for TableError {
    fn from(value: PureOperatorError) -> Self {
        Self::Calculus(value)
    }
}

/// Generate the value calculus (`derivative = false`) or the first-derivative calculus.
///
/// The value is `y_i + (x - x_i) * ((y_{i+1} - y_i)/(x_{i+1} - x_i))`.
/// Values are defined on the closed interval. First derivatives are defined only
/// inside open segments: every endpoint and internal knot is rejected, even where
/// neighbouring slopes are equal. Points outside the domain are always rejected;
/// there is no extrapolation.
///
/// # Errors
/// Rejects malformed, unordered, or oversized data and exact arithmetic overflow.
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
        if x[0] >= x[1] {
            return Err(TableError::Ordering);
        }
        let dx = x[1].checked_add(x[0].checked_neg()?)?;
        let dy = y[1].checked_add(y[0].checked_neg()?)?;
        slopes.push(dy.checked_div(dx)?);
    }
    if validity[0] >= validity[1] || axis[0] > validity[0] || validity[1] > axis[axis.len() - 1] {
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
        .map(|value| {
            builder.push(CalculusNode::Rational {
                value: value.clone(),
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
    for (i, slope) in slopes.into_iter().enumerate() {
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
                value: values[i].clone(),
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
    if !derivative {
        // Interior knots select the following segment's origin. The final knot
        // has no following segment, so retain its sample explicitly rather than
        // reconstructing it from a rounded secant and losing it to cancellation.
        let last = knots.len() - 1;
        let condition = builder.push(CalculusNode::Compare(
            ComparisonOp::Equal,
            input,
            knots[last],
        ))?;
        let endpoint = builder.push(CalculusNode::Rational {
            value: values[last].clone(),
            dimension: value_dimension,
        })?;
        result = builder.push(CalculusNode::Select {
            condition,
            then_value: endpoint,
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
            matches!(nodes[knot.index() as usize], CalculusNode::Rational { ref value, .. } if *value == ExactRational::integer(2))
        );
        assert!(
            matches!(nodes[then_value.index() as usize], CalculusNode::Rational { ref value, .. } if *value == ExactRational::integer(2))
        );
        assert!(
            matches!(nodes[else_value.index() as usize], CalculusNode::Rational { ref value, .. } if *value == ExactRational::integer(-1))
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
