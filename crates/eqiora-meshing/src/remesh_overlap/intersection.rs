//! Exact construction and canonical intersection of triangle and segment fragments.

use super::*;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ConstructedPoint2d {
    pub(super) exact: [BigRational; DIMENSION],
    pub(super) rounded: Point2d,
}

pub(super) fn intersect_triangles(
    source: &Triangle2d,
    target: &Triangle2d,
) -> Result<Vec<Triangle2d>, Diagnostic> {
    require_positive_triangle(source, "source")?;
    require_positive_triangle(target, "target")?;
    let mut candidates = Vec::with_capacity(12);
    for point in source
        .iter()
        .copied()
        .filter(|point| point_in_closed_triangle(*point, target))
    {
        candidates.push(ConstructedPoint2d::from_binary64(point)?);
    }
    for point in target
        .iter()
        .copied()
        .filter(|point| point_in_closed_triangle(*point, source))
    {
        candidates.push(ConstructedPoint2d::from_binary64(point)?);
    }
    for source_edge in triangle_edges(source) {
        for target_edge in triangle_edges(target) {
            if let Some(point) = proper_segment_intersection(source_edge, target_edge)? {
                candidates.push(point);
            }
        }
    }
    let polygon = convex_hull(candidates);
    if polygon.len() < 3 {
        return Ok(Vec::new());
    }
    let mut triangles = Vec::with_capacity(polygon.len() - 2);
    for index in 1..polygon.len() - 1 {
        let exact_sign = exact_orientation(
            &polygon[0].exact,
            &polygon[index].exact,
            &polygon[index + 1].exact,
        );
        if exact_sign.is_zero() {
            return Err(invalid_overlap(
                "exact common-refinement polygon triangulation is degenerate",
            ));
        }
        if exact_sign < BigRational::zero() {
            return Err(invalid_overlap(
                "canonical common-refinement polygon lost positive orientation",
            ));
        }
        let triangle = [
            polygon[0].rounded,
            polygon[index].rounded,
            polygon[index + 1].rounded,
        ];
        triangle_measure(&triangle).map_err(|_| {
            invalid_overlap(
                "exact positive common-refinement fragment collapses under canonical nearest-binary64 coordinates",
            )
        })?;
        triangles.push(triangle);
    }
    Ok(triangles)
}

fn point_in_closed_triangle(point: Point2d, triangle: &Triangle2d) -> bool {
    triangle_edges(triangle)
        .into_iter()
        .all(|edge| orientation(edge[0], edge[1], point) >= 0.0)
}

fn triangle_edges(triangle: &Triangle2d) -> [Segment2d; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
}

pub(super) fn proper_segment_intersection(
    source: Segment2d,
    target: Segment2d,
) -> Result<Option<ConstructedPoint2d>, Diagnostic> {
    let source_start = orientation(target[0], target[1], source[0]);
    let source_end = orientation(target[0], target[1], source[1]);
    let target_start = orientation(source[0], source[1], target[0]);
    let target_end = orientation(source[0], source[1], target[1]);
    if !opposite_sign(source_start, source_end) || !opposite_sign(target_start, target_end) {
        return Ok(None);
    }
    let source_start = exact_point(source[0])?;
    let source_direction = exact_direction(source[0], source[1])?;
    let target_start = exact_point(target[0])?;
    let target_direction = exact_direction(target[0], target[1])?;
    let offset = [
        &target_start[0] - &source_start[0],
        &target_start[1] - &source_start[1],
    ];
    let denominator = exact_cross(&source_direction, &target_direction);
    if denominator.is_zero() {
        return Err(invalid_overlap(
            "robust proper crossing has an exactly parallel construction denominator",
        ));
    }
    let source_parameter = exact_cross(&offset, &target_direction) / &denominator;
    let target_parameter = exact_cross(&offset, &source_direction) / denominator;
    if source_parameter <= BigRational::zero()
        || source_parameter >= BigRational::one()
        || target_parameter <= BigRational::zero()
        || target_parameter >= BigRational::one()
    {
        return Err(invalid_overlap(
            "robust proper crossing disagrees with exact binary-rational segment parameters",
        ));
    }
    let exact = [
        &source_start[0] + &source_parameter * &source_direction[0],
        &source_start[1] + source_parameter * &source_direction[1],
    ];
    let point = [
        round_exact_coordinate(&exact[0])?,
        round_exact_coordinate(&exact[1])?,
    ];
    Ok(Some(ConstructedPoint2d {
        exact,
        rounded: point.map(normalize_zero),
    }))
}

pub(super) fn exact_point(point: Point2d) -> Result<[BigRational; DIMENSION], Diagnostic> {
    point
        .map(|coordinate| {
            BigRational::from_float(coordinate).ok_or_else(|| {
                invalid_overlap("finite mesh coordinate has no exact binary-rational image")
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| invalid_overlap("exact point construction lost its fixed dimension"))
}

impl ConstructedPoint2d {
    pub(super) fn from_binary64(point: Point2d) -> Result<Self, Diagnostic> {
        Ok(Self {
            exact: exact_point(point)?,
            rounded: point.map(normalize_zero),
        })
    }
}

pub(super) fn exact_direction(
    start: Point2d,
    end: Point2d,
) -> Result<[BigRational; DIMENSION], Diagnostic> {
    let start = exact_point(start)?;
    let end = exact_point(end)?;
    Ok([&end[0] - &start[0], &end[1] - &start[1]])
}

fn exact_cross(left: &[BigRational; DIMENSION], right: &[BigRational; DIMENSION]) -> BigRational {
    &left[0] * &right[1] - &left[1] * &right[0]
}

fn exact_orientation(
    left: &[BigRational; DIMENSION],
    right: &[BigRational; DIMENSION],
    query: &[BigRational; DIMENSION],
) -> BigRational {
    let first = [&right[0] - &left[0], &right[1] - &left[1]];
    let second = [&query[0] - &left[0], &query[1] - &left[1]];
    exact_cross(&first, &second)
}

fn round_exact_coordinate(value: &BigRational) -> Result<f64, Diagnostic> {
    let rounded = value.to_f64().ok_or_else(|| {
        invalid_overlap("exact proper-intersection coordinate is outside binary64 range")
    })?;
    if !rounded.is_finite() {
        return Err(invalid_overlap(
            "exact proper-intersection coordinate rounded to a non-finite binary64 value",
        ));
    }
    let represented = BigRational::from_float(rounded)
        .expect("a finite binary64 coordinate always has an exact rational image");
    let certified = match value.cmp(&represented) {
        Ordering::Equal => true,
        Ordering::Greater => {
            BigRational::from_float(next_up(rounded)).is_some_and(|upper| value <= &upper)
        }
        Ordering::Less => {
            BigRational::from_float(next_down(rounded)).is_some_and(|lower| value >= &lower)
        }
    };
    if !certified {
        return Err(invalid_overlap(
            "exact proper-intersection coordinate did not round within one binary64 step",
        ));
    }
    Ok(rounded)
}

pub(super) fn next_up(value: f64) -> f64 {
    if value == f64::INFINITY {
        value
    } else if value == 0.0 {
        f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

pub(super) fn next_down(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        value
    } else if value == 0.0 {
        f64::from_bits((1_u64 << 63) | 1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

fn opposite_sign(left: f64, right: f64) -> bool {
    (left < 0.0 && right > 0.0) || (left > 0.0 && right < 0.0)
}

pub(super) fn convex_hull(mut points: Vec<ConstructedPoint2d>) -> Vec<ConstructedPoint2d> {
    points.sort_by(compare_constructed_points);
    points.dedup_by(|left, right| left.exact == right.exact);
    if points.len() < 3 {
        return points;
    }
    let mut lower: Vec<ConstructedPoint2d> = Vec::with_capacity(points.len());
    for point in &points {
        while lower.len() >= 2
            && exact_orientation(
                &lower[lower.len() - 2].exact,
                &lower[lower.len() - 1].exact,
                &point.exact,
            ) <= BigRational::zero()
        {
            lower.pop();
        }
        lower.push(point.clone());
    }
    let mut upper: Vec<ConstructedPoint2d> = Vec::with_capacity(points.len());
    for point in points.iter().rev() {
        while upper.len() >= 2
            && exact_orientation(
                &upper[upper.len() - 2].exact,
                &upper[upper.len() - 1].exact,
                &point.exact,
            ) <= BigRational::zero()
        {
            upper.pop();
        }
        upper.push(point.clone());
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn compare_constructed_points(left: &ConstructedPoint2d, right: &ConstructedPoint2d) -> Ordering {
    left.exact[0]
        .cmp(&right.exact[0])
        .then_with(|| left.exact[1].cmp(&right.exact[1]))
}

pub(super) fn collinear_segment_overlap(
    source: &Segment2d,
    target: &Segment2d,
) -> Result<Option<Segment2d>, Diagnostic> {
    let source_signs = [
        orientation(source[0], source[1], target[0]),
        orientation(source[0], source[1], target[1]),
    ];
    let target_signs = [
        orientation(target[0], target[1], source[0]),
        orientation(target[0], target[1], source[1]),
    ];
    if source_signs.iter().any(|sign| !sign.is_finite())
        || target_signs.iter().any(|sign| !sign.is_finite())
    {
        return Err(invalid_overlap(
            "retained-facet collinearity predicate is non-finite",
        ));
    }
    let exactly_collinear = source_signs == [0.0, 0.0] && target_signs == [0.0, 0.0];
    let representative = if compare_point_arrays(source, target).is_le() {
        source
    } else {
        target
    };
    let axis = dominant_axis(representative);
    let direction = (representative[1][axis] - representative[0][axis]).signum();
    let source_ordered = ordered_along(source, axis, direction);
    let target_ordered = ordered_along(target, axis, direction);
    let lower = if projected_coordinate(source_ordered[0], axis, direction)
        >= projected_coordinate(target_ordered[0], axis, direction)
    {
        source_ordered[0]
    } else {
        target_ordered[0]
    };
    let upper = if projected_coordinate(source_ordered[1], axis, direction)
        <= projected_coordinate(target_ordered[1], axis, direction)
    {
        source_ordered[1]
    } else {
        target_ordered[1]
    };
    if projected_coordinate(upper, axis, direction) <= projected_coordinate(lower, axis, direction)
    {
        return Ok(None);
    }
    if !exactly_collinear
        && ![lower, upper]
            .into_iter()
            .map(|point| {
                Ok(
                    line_parameter_through_rounding_cell(source, point)?.is_some()
                        && line_parameter_through_rounding_cell(target, point)?.is_some(),
                )
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?
            .into_iter()
            .all(|certified| certified)
    {
        return Ok(None);
    }
    let mut segment = [
        certified_line_point(representative, lower)?,
        certified_line_point(representative, upper)?,
    ];
    canonicalize_segment(&mut segment);
    let _ = segment_measure(&segment)?;
    Ok(Some(segment))
}

fn dominant_axis(segment: &Segment2d) -> usize {
    usize::from((segment[1][1] - segment[0][1]).abs() > (segment[1][0] - segment[0][0]).abs())
}

fn ordered_along(segment: &Segment2d, axis: usize, direction: f64) -> Segment2d {
    if projected_coordinate(segment[0], axis, direction)
        <= projected_coordinate(segment[1], axis, direction)
    {
        *segment
    } else {
        [segment[1], segment[0]]
    }
}

fn projected_coordinate(point: Point2d, axis: usize, direction: f64) -> f64 {
    direction * point[axis]
}
