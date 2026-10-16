//! Bidirectional measure and first-moment coverage using compensated accumulation.

use super::*;

const COVERAGE_ULPS: f64 = 16_384.0;

#[derive(Debug, Clone, Copy, Default)]
struct CoverageAccumulator {
    area_or_length: CompensatedSum,
    first_moment: [CompensatedSum; DIMENSION],
    term_count: usize,
}

impl CoverageAccumulator {
    fn add(&mut self, measure: f64, first_moment: Point2d) -> Result<(), Diagnostic> {
        self.area_or_length.add(measure)?;
        for (sum, value) in self.first_moment.iter_mut().zip(first_moment) {
            sum.add(value)?;
        }
        self.term_count = self
            .term_count
            .checked_add(1)
            .ok_or_else(|| invalid_overlap("common-refinement coverage count overflows usize"))?;
        Ok(())
    }

    fn require(self, measure: f64, first_moment: Point2d, label: &str) -> Result<(), Diagnostic> {
        require_close(
            self.area_or_length,
            measure,
            self.term_count,
            &format!("{label} measure"),
        )?;
        for (axis, (sum, expected)) in self.first_moment.into_iter().zip(first_moment).enumerate() {
            require_close(
                sum,
                expected,
                self.term_count,
                &format!("{label} first moment axis {axis}"),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
    absolute_sum: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) -> Result<(), Diagnostic> {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
        self.absolute_sum += value.abs();
        if !self.sum.is_finite() || !self.correction.is_finite() || !self.absolute_sum.is_finite() {
            return Err(invalid_overlap(
                "common-refinement coverage accumulation overflowed",
            ));
        }
        Ok(())
    }

    fn value(self) -> f64 {
        self.sum + self.correction
    }
}

pub(super) fn validate_cell_coverage(
    source_cells: &[CellId],
    source_triangles: &[Triangle2d],
    target_cells: &[CellId],
    target_triangles: &[Triangle2d],
    fragments: &[RevisionCellFragment2d],
) -> Result<(), Diagnostic> {
    let mut source = source_cells
        .iter()
        .copied()
        .map(|cell| (cell, CoverageAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    let mut target = target_cells
        .iter()
        .copied()
        .map(|cell| (cell, CoverageAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    for fragment in fragments {
        source
            .get_mut(&fragment.source_cell)
            .ok_or_else(|| invalid_overlap("cell fragment references an unselected source cell"))?
            .add(fragment.area, fragment.first_moment)?;
        target
            .get_mut(&fragment.target_cell)
            .ok_or_else(|| invalid_overlap("cell fragment references an unselected target cell"))?
            .add(fragment.area, fragment.first_moment)?;
    }
    for (cell, triangle) in source_cells.iter().zip(source_triangles) {
        let (area, moment) = triangle_measure(triangle)?;
        source[cell].require(area, moment, &format!("source cell {}", cell.index()))?;
    }
    for (cell, triangle) in target_cells.iter().zip(target_triangles) {
        let (area, moment) = triangle_measure(triangle)?;
        target[cell].require(area, moment, &format!("target cell {}", cell.index()))?;
    }
    Ok(())
}

pub(super) fn validate_facet_coverage(
    source_sides: &[RetainedFacetSide2d],
    source_geometry: &[SideGeometry2d],
    target_sides: &[RetainedFacetSide2d],
    target_geometry: &[SideGeometry2d],
    fragments: &[RevisionFacetFragment2d],
) -> Result<(), Diagnostic> {
    let mut source = source_sides
        .iter()
        .copied()
        .map(|side| (side, CoverageAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    let mut target = target_sides
        .iter()
        .copied()
        .map(|side| (side, CoverageAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    for fragment in fragments {
        source
            .get_mut(&fragment.source)
            .ok_or_else(|| invalid_overlap("facet fragment references an unselected source side"))?
            .add(fragment.length, fragment.first_moment)?;
        target
            .get_mut(&fragment.target)
            .ok_or_else(|| invalid_overlap("facet fragment references an unselected target side"))?
            .add(fragment.length, fragment.first_moment)?;
    }
    for (side, geometry) in source_sides.iter().zip(source_geometry) {
        source[side].require(
            geometry.length,
            geometry.first_moment,
            &format!("source facet {}", side.facet().index()),
        )?;
    }
    for (side, geometry) in target_sides.iter().zip(target_geometry) {
        target[side].require(
            geometry.length,
            geometry.first_moment,
            &format!("target facet {}", side.facet().index()),
        )?;
    }
    Ok(())
}

fn require_close(
    actual: CompensatedSum,
    expected: f64,
    term_count: usize,
    label: &str,
) -> Result<(), Diagnostic> {
    let actual_value = actual.value();
    let scale = expected
        .abs()
        .max(actual.absolute_sum)
        .max(f64::MIN_POSITIVE);
    let count = term_count.max(1) as f64;
    let tolerance = COVERAGE_ULPS * f64::EPSILON * count * scale;
    if !actual_value.is_finite()
        || !expected.is_finite()
        || !tolerance.is_finite()
        || (actual_value - expected).abs() > tolerance
    {
        return Err(invalid_overlap(format!(
            "common-refinement {label} is incomplete or multiply covered: expected {expected}, got {actual_value}",
        )));
    }
    Ok(())
}
