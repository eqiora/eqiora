//! Deterministic structural coloring for independent centered-Jacobian audits.
//!
//! This module is verification infrastructure. It derives a conservative
//! column-intersection graph from typed assembly closures, never from values in
//! the analytic Jacobian being checked. Each color requires two complete
//! residual evaluations and still reconstructs every individual column.

use std::collections::BTreeSet;

#[cfg(test)]
use std::cell::Cell;

use eqiora_assembly::{AssemblyMap, LocalUnknown};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

#[cfg(test)]
thread_local! {
    static CENTERED_RESIDUAL_ASSEMBLIES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_centered_residual_assembly_count() {
    CENTERED_RESIDUAL_ASSEMBLIES.set(0);
}

#[cfg(test)]
pub(crate) fn centered_residual_assembly_count() -> usize {
    CENTERED_RESIDUAL_ASSEMBLIES.get()
}

/// Conservative structural row support and deterministic coloring for one square audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StructuralJacobianPattern {
    row_count: usize,
    row_supports: Vec<Vec<usize>>,
    colors: Vec<Vec<usize>>,
    globally_coupled_singleton_count: usize,
}

impl StructuralJacobianPattern {
    pub(crate) fn colors(&self) -> &[Vec<usize>] {
        &self.colors
    }

    fn row_count(&self) -> usize {
        self.row_count
    }

    fn column_count(&self) -> usize {
        self.row_supports.len()
    }

    fn globally_coupled_singleton_count(&self) -> usize {
        self.globally_coupled_singleton_count
    }

    #[cfg(test)]
    pub(crate) fn is_singleton(&self, column: usize) -> bool {
        self.colors.iter().any(|color| color.as_slice() == [column])
    }
}

/// Ordered builder over the exact typed contribution inventory.
pub(crate) struct StructuralJacobianPatternBuilder {
    row_count: usize,
    row_supports: Vec<BTreeSet<usize>>,
    globally_coupled: Vec<bool>,
    expected_contributions: usize,
    next_contribution: usize,
}

impl StructuralJacobianPatternBuilder {
    pub(crate) fn new(
        row_count: usize,
        column_count: usize,
        expected_contributions: usize,
    ) -> Result<Self, Diagnostic> {
        if row_count == 0 || column_count == 0 || expected_contributions == 0 {
            return Err(invalid(
                "structural Jacobian pattern requires non-empty rows, columns, and contributions",
            ));
        }
        Ok(Self {
            row_count,
            row_supports: vec![BTreeSet::new(); column_count],
            globally_coupled: vec![false; column_count],
            expected_contributions,
            next_contribution: 0,
        })
    }

    /// Include one dense local dependency closure in canonical contribution order.
    ///
    /// Every free local unknown is conservatively connected to every retained
    /// local residual row. `expected_local_size` comes from the typed operator,
    /// independently of the assembly map presented here.
    pub(crate) fn include_dense_local(
        &mut self,
        contribution: usize,
        expected_local_size: usize,
        map: &AssemblyMap<f64>,
    ) -> Result<(), Diagnostic> {
        if contribution != self.next_contribution || contribution >= self.expected_contributions {
            return Err(invalid(
                "structural Jacobian contributions must be complete and canonically ordered",
            ));
        }
        if expected_local_size == 0
            || map.equations().len() != expected_local_size
            || map.unknowns().len() != expected_local_size
        {
            return Err(invalid(
                "structural Jacobian contribution underestimates its typed local closure",
            ));
        }
        let rows = map
            .equations()
            .iter()
            .filter_map(|equation| equation.map(|dof| dof.index()))
            .collect::<Vec<_>>();
        if rows.iter().any(|row| *row >= self.row_count) {
            return Err(invalid(
                "structural Jacobian contribution references a residual row outside the audit",
            ));
        }
        for unknown in map.unknowns() {
            let LocalUnknown::Free(dof) = unknown else {
                continue;
            };
            let support = self.row_supports.get_mut(dof.index()).ok_or_else(|| {
                invalid("structural Jacobian contribution references a column outside the audit")
            })?;
            support.extend(rows.iter().copied());
        }
        self.next_contribution += 1;
        Ok(())
    }

    /// Force one independently known global coupling to a singleton color.
    pub(crate) fn mark_globally_coupled(&mut self, column: usize) -> Result<(), Diagnostic> {
        let support = self
            .row_supports
            .get_mut(column)
            .ok_or_else(|| invalid("global Jacobian coupling references an unknown column"))?;
        support.extend(0..self.row_count);
        self.globally_coupled[column] = true;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<StructuralJacobianPattern, Diagnostic> {
        if self.next_contribution != self.expected_contributions {
            return Err(invalid(format!(
                "structural Jacobian pattern received {} of {} typed contributions",
                self.next_contribution, self.expected_contributions
            )));
        }
        if let Some(column) = self.row_supports.iter().position(BTreeSet::is_empty) {
            return Err(invalid(format!(
                "structural Jacobian column {column} has no proven residual dependency"
            )));
        }
        let row_supports = self
            .row_supports
            .into_iter()
            .map(|support| support.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let colors = deterministic_colors(&row_supports, &self.globally_coupled);
        validate_colors(&row_supports, &self.globally_coupled, &colors)?;
        Ok(StructuralJacobianPattern {
            row_count: self.row_count,
            row_supports,
            colors,
            globally_coupled_singleton_count: self
                .globally_coupled
                .into_iter()
                .filter(|coupled| *coupled)
                .count(),
        })
    }
}

/// Measurements from one explicitly requested centered-Jacobian verification.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CenteredJacobianVerification {
    column_count: usize,
    color_count: usize,
    globally_coupled_singleton_count: usize,
    residual_assembly_count: usize,
    maximum_error: f64,
}

impl CenteredJacobianVerification {
    pub(crate) fn new(
        colors: Vec<Vec<usize>>,
        globally_coupled_singleton_count: usize,
        maximum_error: f64,
    ) -> Result<Self, Diagnostic> {
        let residual_assembly_count = colors
            .len()
            .checked_mul(2)
            .ok_or_else(|| invalid("centered Jacobian residual-assembly count overflows usize"))?;
        if colors.is_empty()
            || colors.iter().any(Vec::is_empty)
            || !maximum_error.is_finite()
            || maximum_error < 0.0
        {
            return Err(invalid(
                "centered Jacobian evidence requires non-empty colors and a finite non-negative error",
            ));
        }
        let column_count = colors.iter().map(Vec::len).sum();
        let mut observed = vec![false; column_count];
        let mut previous_first = None;
        for color in &colors {
            if previous_first.is_some_and(|previous| previous >= color[0])
                || color.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(invalid(
                    "centered Jacobian evidence colors must retain deterministic column order",
                ));
            }
            previous_first = Some(color[0]);
            for &column in color {
                let slot = observed.get_mut(column).ok_or_else(|| {
                    invalid("centered Jacobian evidence contains a column outside its inventory")
                })?;
                if *slot {
                    return Err(invalid(
                        "centered Jacobian evidence contains a duplicate column",
                    ));
                }
                *slot = true;
            }
        }
        if observed.contains(&false) {
            return Err(invalid(
                "centered Jacobian evidence omits one or more audited columns",
            ));
        }
        if globally_coupled_singleton_count > colors.len() {
            return Err(invalid(
                "centered Jacobian evidence contains more global singleton columns than colors",
            ));
        }
        Ok(Self {
            column_count,
            color_count: colors.len(),
            globally_coupled_singleton_count,
            residual_assembly_count,
            maximum_error,
        })
    }

    #[must_use]
    pub(crate) const fn column_count(&self) -> usize {
        self.column_count
    }

    #[must_use]
    pub(crate) const fn color_count(&self) -> usize {
        self.color_count
    }

    #[must_use]
    pub(crate) const fn globally_coupled_singleton_count(&self) -> usize {
        self.globally_coupled_singleton_count
    }

    #[must_use]
    pub(crate) const fn residual_assembly_count(&self) -> usize {
        self.residual_assembly_count
    }

    #[must_use]
    pub(crate) const fn maximum_error(&self) -> f64 {
        self.maximum_error
    }
}

/// Check every analytic column using two residual evaluations per structural color.
pub(crate) fn audit_centered_jacobian<R, J>(
    point: &[f64],
    pattern: &StructuralJacobianPattern,
    tolerance_scale: f64,
    label: &'static str,
    mut residual: R,
    mut analytic_column: J,
) -> Result<CenteredJacobianVerification, Diagnostic>
where
    R: FnMut(&[f64]) -> Result<Vec<f64>, Diagnostic>,
    J: FnMut(usize, &mut [f64]) -> Result<(), Diagnostic>,
{
    let row_count = pattern.row_count();
    if point.len() != pattern.column_count()
        || row_count == 0
        || point.iter().any(|value| !value.is_finite())
        || !tolerance_scale.is_finite()
        || tolerance_scale <= 0.0
    {
        return Err(invalid(
            "centered Jacobian audit differs from its finite structural pattern",
        ));
    }

    let mut maximum_error = 0.0_f64;
    for color in pattern.colors() {
        let mut columns = Vec::with_capacity(color.len());
        let mut plus = point.to_vec();
        let mut minus = point.to_vec();
        for &column in color {
            let epsilon = f64::EPSILON.cbrt() * (1.0 + point[column].abs());
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(invalid(
                    "centered Jacobian audit produced an invalid column step",
                ));
            }
            plus[column] += epsilon;
            minus[column] -= epsilon;
            let mut analytic = vec![0.0; row_count];
            analytic_column(column, &mut analytic)?;
            if analytic.iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    "centered Jacobian audit received a non-finite analytic column",
                ));
            }
            let analytic_norm = euclidean_norm(&analytic);
            let tolerance = tolerance_scale * (1.0 + analytic_norm);
            columns.push((column, epsilon, analytic, tolerance));
        }

        #[cfg(test)]
        CENTERED_RESIDUAL_ASSEMBLIES.set(CENTERED_RESIDUAL_ASSEMBLIES.get() + 2);
        let plus_residual = residual(&plus)?;
        let minus_residual = residual(&minus)?;
        if plus_residual.len() != row_count
            || minus_residual.len() != row_count
            || plus_residual
                .iter()
                .chain(&minus_residual)
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "centered Jacobian audit residual shape or finiteness changed",
            ));
        }
        let half_difference = plus_residual
            .iter()
            .zip(minus_residual)
            .map(|(plus, minus)| (plus - minus) * 0.5)
            .collect::<Vec<_>>();

        // This aggregate comparison catches an omitted dependency even when
        // the omitted row lies outside every declared support in this color.
        let aggregate_error = half_difference
            .iter()
            .enumerate()
            .map(|(row, observed)| {
                let expected = columns
                    .iter()
                    .map(|(_, epsilon, analytic, _)| epsilon * analytic[row])
                    .sum::<f64>();
                (observed - expected).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        let aggregate_tolerance = columns
            .iter()
            .map(|(_, epsilon, _, tolerance)| (epsilon * tolerance).powi(2))
            .sum::<f64>()
            .sqrt();
        if !aggregate_error.is_finite() || aggregate_error > aggregate_tolerance {
            return Err(solve_failed(format!(
                "analytic {label} Jacobian column color beginning at {} combined error {aggregate_error:e} exceeds centered-difference tolerance {aggregate_tolerance:e}",
                color[0]
            )));
        }

        for (column, epsilon, analytic, tolerance) in columns {
            let support = &pattern.row_supports[column];
            let mut centered = vec![0.0; row_count];
            for &row in support {
                centered[row] = half_difference[row] / epsilon;
            }
            let error = centered
                .iter()
                .zip(analytic)
                .map(|(centered, analytic)| (centered - analytic).powi(2))
                .sum::<f64>()
                .sqrt();
            if !error.is_finite() || error > tolerance {
                return Err(solve_failed(format!(
                    "analytic {label} Jacobian column {column} error {error:e} exceeds centered-difference tolerance {tolerance:e}"
                )));
            }
            maximum_error = maximum_error.max(error);
        }
    }
    CenteredJacobianVerification::new(
        pattern.colors.clone(),
        pattern.globally_coupled_singleton_count(),
        maximum_error,
    )
}

fn deterministic_colors(row_supports: &[Vec<usize>], globally_coupled: &[bool]) -> Vec<Vec<usize>> {
    let mut colors: Vec<Vec<usize>> = Vec::new();
    for column in 0..row_supports.len() {
        if globally_coupled[column] {
            colors.push(vec![column]);
            continue;
        }
        let compatible = colors.iter().position(|color| {
            !color.iter().any(|member| globally_coupled[*member])
                && color
                    .iter()
                    .all(|member| !intersects(&row_supports[column], &row_supports[*member]))
        });
        match compatible {
            Some(color) => colors[color].push(column),
            None => colors.push(vec![column]),
        }
    }
    colors
}

fn validate_colors(
    row_supports: &[Vec<usize>],
    globally_coupled: &[bool],
    colors: &[Vec<usize>],
) -> Result<(), Diagnostic> {
    if row_supports.len() != globally_coupled.len() || colors.is_empty() {
        return Err(invalid(
            "structural Jacobian coloring differs from its column inventory",
        ));
    }
    let mut observed = vec![false; row_supports.len()];
    for color in colors {
        if color.is_empty() {
            return Err(invalid(
                "structural Jacobian coloring contains an empty color",
            ));
        }
        for (position, &column) in color.iter().enumerate() {
            if column >= row_supports.len()
                || observed[column]
                || position > 0 && color[position - 1] >= column
            {
                return Err(invalid(
                    "structural Jacobian coloring must cover columns once in deterministic order",
                ));
            }
            if globally_coupled[column] && color.len() != 1 {
                return Err(invalid(
                    "globally coupled Jacobian columns require singleton colors",
                ));
            }
            if color[..position]
                .iter()
                .any(|other| intersects(&row_supports[column], &row_supports[*other]))
            {
                return Err(invalid(
                    "structural Jacobian color contains colliding residual support",
                ));
            }
            observed[column] = true;
        }
    }
    if observed.contains(&false) {
        return Err(invalid(
            "structural Jacobian coloring does not cover every column",
        ));
    }
    if colors != deterministic_colors(row_supports, globally_coupled) {
        return Err(invalid(
            "structural Jacobian coloring differs from deterministic first-fit order",
        ));
    }
    Ok(())
}

fn intersects(left: &[usize], right: &[usize]) -> bool {
    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn euclidean_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_assembly::{AssemblyMap, DofId, LocalUnknown};

    fn map(rows: &[usize], columns: &[usize]) -> AssemblyMap<f64> {
        let width = rows.len().max(columns.len());
        let mut equations = rows
            .iter()
            .copied()
            .map(|row| Some(DofId::new(row)))
            .collect::<Vec<_>>();
        let mut unknowns = columns
            .iter()
            .copied()
            .map(|column| LocalUnknown::Free(DofId::new(column)))
            .collect::<Vec<_>>();
        equations.resize(width, None);
        unknowns.resize(width, LocalUnknown::Fixed(0.0));
        AssemblyMap::new(equations, unknowns).unwrap()
    }

    #[test]
    fn coloring_is_deterministic_and_first_fit_in_column_order() {
        let mut first = StructuralJacobianPatternBuilder::new(2, 3, 3).unwrap();
        first.include_dense_local(0, 1, &map(&[0], &[0])).unwrap();
        first.include_dense_local(1, 1, &map(&[1], &[1])).unwrap();
        first.include_dense_local(2, 1, &map(&[0], &[2])).unwrap();
        let first = first.finish().unwrap();

        let mut second = StructuralJacobianPatternBuilder::new(2, 3, 3).unwrap();
        second.include_dense_local(0, 1, &map(&[0], &[0])).unwrap();
        second.include_dense_local(1, 1, &map(&[1], &[1])).unwrap();
        second.include_dense_local(2, 1, &map(&[0], &[2])).unwrap();
        let second = second.finish().unwrap();

        assert_eq!(first, second);
        assert_eq!(first.colors(), &[vec![0, 1], vec![2]]);
    }

    #[test]
    fn missing_or_underwidth_typed_dependencies_fail_closed() {
        let mut missing = StructuralJacobianPatternBuilder::new(2, 2, 2).unwrap();
        missing.include_dense_local(0, 1, &map(&[0], &[0])).unwrap();
        assert!(missing.finish().is_err());

        let mut underwidth = StructuralJacobianPatternBuilder::new(2, 2, 1).unwrap();
        assert!(
            underwidth
                .include_dense_local(0, 2, &map(&[0], &[0]))
                .is_err()
        );
    }

    #[test]
    fn underestimated_support_is_rejected_by_the_independent_residual() {
        let mut builder = StructuralJacobianPatternBuilder::new(2, 2, 2).unwrap();
        builder.include_dense_local(0, 1, &map(&[0], &[0])).unwrap();
        builder.include_dense_local(1, 1, &map(&[1], &[1])).unwrap();
        let pattern = builder.finish().unwrap();
        let error = audit_centered_jacobian(
            &[0.25, -0.5],
            &pattern,
            1.0e-8,
            "mutant",
            |point| Ok(vec![point[0], point[0] + point[1]]),
            |column, output| {
                output.copy_from_slice(match column {
                    0 => &[1.0, 1.0],
                    1 => &[0.0, 1.0],
                    _ => unreachable!(),
                });
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.message().contains("Jacobian column"));
    }

    #[test]
    fn colliding_colors_and_global_coupling_fail_closed() {
        let supports = vec![vec![0], vec![0]];
        assert!(validate_colors(&supports, &[false, false], &[vec![0, 1]]).is_err());

        let disjoint = vec![vec![0, 1], vec![1]];
        assert!(validate_colors(&disjoint, &[true, false], &[vec![0, 1]]).is_err());

        let mut builder = StructuralJacobianPatternBuilder::new(2, 2, 2).unwrap();
        builder.include_dense_local(0, 1, &map(&[0], &[0])).unwrap();
        builder.include_dense_local(1, 1, &map(&[1], &[1])).unwrap();
        builder.mark_globally_coupled(0).unwrap();
        assert_eq!(builder.finish().unwrap().colors(), &[vec![0], vec![1]]);
    }

    #[test]
    fn omitted_global_mark_is_rejected_by_the_independent_residual() {
        let mut builder = StructuralJacobianPatternBuilder::new(3, 3, 3).unwrap();
        builder.include_dense_local(0, 1, &map(&[0], &[0])).unwrap();
        builder.include_dense_local(1, 1, &map(&[1], &[1])).unwrap();
        builder.include_dense_local(2, 1, &map(&[2], &[2])).unwrap();
        let pattern = builder.finish().unwrap();
        assert_eq!(pattern.colors(), &[vec![0, 1, 2]]);

        let error = audit_centered_jacobian(
            &[0.25, -0.5, 0.75],
            &pattern,
            1.0e-8,
            "omitted-global-mark mutant",
            |point| Ok(vec![point[0], point[0] + point[1], point[0] + point[2]]),
            |column, output| {
                output.copy_from_slice(match column {
                    0 => &[1.0, 1.0, 1.0],
                    1 => &[0.0, 1.0, 0.0],
                    2 => &[0.0, 0.0, 1.0],
                    _ => unreachable!(),
                });
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.message().contains("Jacobian column"));
    }

    #[test]
    fn audit_evidence_rejects_invalid_counts_and_errors() {
        assert!(CenteredJacobianVerification::new(Vec::new(), 0, 0.0).is_err());
        assert!(CenteredJacobianVerification::new(vec![vec![0]], 0, -1.0).is_err());
        assert!(CenteredJacobianVerification::new(vec![Vec::new()], 0, 0.0).is_err());
        assert!(CenteredJacobianVerification::new(vec![vec![0]], 2, 0.0).is_err());
    }
}
