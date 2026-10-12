//! Exact closed binary64 rounding-cell certificates for retained facets.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct RationalInterval {
    lower: BigRational,
    upper: BigRational,
}

impl RationalInterval {
    fn rounding_cell(value: f64) -> Result<Self, Diagnostic> {
        let previous = next_down(value);
        let next = next_up(value);
        if !previous.is_finite() || !next.is_finite() {
            return Err(invalid_overlap(
                "retained-facet coordinate has no finite binary64 rounding cell",
            ));
        }
        let value =
            BigRational::from_float(value).expect("finite coordinate has an exact rational image");
        let two = BigRational::from_integer(2.into());
        Ok(Self {
            lower: (BigRational::from_float(previous)
                .expect("finite predecessor has an exact rational image")
                + &value)
                / &two,
            upper: (&value
                + BigRational::from_float(next)
                    .expect("finite successor has an exact rational image"))
                / two,
        })
    }

    fn intersect(self, other: Self) -> Option<Self> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        (lower <= upper).then_some(Self { lower, upper })
    }
}

pub(super) fn line_parameter_through_rounding_cell(
    line: &Segment2d,
    point: Point2d,
) -> Result<Option<RationalInterval>, Diagnostic> {
    let start = exact_point(line[0])?;
    let direction = exact_direction(line[0], line[1])?;
    let mut parameter = None;
    for axis in 0..DIMENSION {
        let cell = RationalInterval::rounding_cell(point[axis])?;
        if direction[axis].is_zero() {
            if start[axis] < cell.lower || start[axis] > cell.upper {
                return Ok(None);
            }
            continue;
        }
        let first = (&cell.lower - &start[axis]) / &direction[axis];
        let second = (&cell.upper - &start[axis]) / &direction[axis];
        let axis_parameter = RationalInterval {
            lower: first.clone().min(second.clone()),
            upper: first.max(second),
        };
        parameter = match parameter {
            None => Some(axis_parameter),
            Some(current) => current.intersect(axis_parameter),
        };
        if parameter.is_none() {
            return Ok(None);
        }
    }
    parameter
        .map(Some)
        .ok_or_else(|| invalid_overlap("retained facet has no nonzero exact line direction"))
}

/// Preserve the supplied binary64 point after proving exact closed-cell support.
/// A material trace is interpolated before X + d. Its rounded coordinate may
/// lie on either side of a shared cell boundary; rounding an arbitrary line
/// representative again would impose a different geometry construction.
pub(super) fn certified_line_point(
    line: &Segment2d,
    rounded: Point2d,
) -> Result<Point2d, Diagnostic> {
    line_parameter_through_rounding_cell(line, rounded)?.ok_or_else(|| {
        invalid_overlap("retained-facet endpoint is outside the canonical line rounding cell")
    })?;
    Ok(rounded.map(normalize_zero))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_cell_corner_preserves_the_supplied_trace_point() {
        // Write u = 2^-52. The x cell requires t <= 1/2, whereas the
        // y cell requires t >= 1/2. Their exact intersection is {1/2}.
        // Both supplied coordinates are odd mantissas: rounding the line
        // midpoint to nearest-even would change both, despite valid support.
        let u = f64::EPSILON;
        let source = [[1.0, 1.0], [1.0 + 3.0 * u, 1.0 + u]];
        let point = [1.0 + u, 1.0 + u];
        let half = BigRational::new(1.into(), 2.into());
        let interval = line_parameter_through_rounding_cell(&source, point)
            .unwrap()
            .unwrap();
        assert_eq!(interval.lower, half);
        assert_eq!(interval.upper, half);
        assert_eq!(certified_line_point(&source, point).unwrap(), point);
        assert_eq!(
            certified_line_point(&[source[1], source[0]], point).unwrap(),
            point
        );

        // Moving y one ULP upward requires t >= 3/2; x still requires
        // t <= 1/2. This is no longer a rounding-cell certificate.
        let outside = [point[0], next_up(point[1])];
        assert!(
            line_parameter_through_rounding_cell(&source, outside)
                .unwrap()
                .is_none()
        );
        assert!(certified_line_point(&source, outside).is_err());
    }
}
