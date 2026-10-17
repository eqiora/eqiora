//! Component coordinates share the scalar SSA affine proof without synthetic symbols.
use super::*;

impl ScalarInputOperatorIr {
    pub(crate) fn constant_coordinate_jacobian(
        &self,
        selected: &[ScalarSymbolCoordinate],
    ) -> Result<ConstantSymbolJacobian, Diagnostic> {
        let mut columns = HashMap::new();
        for (column, coordinate) in selected.iter().enumerate() {
            if columns.insert(coordinate, column).is_some() {
                return Err(ir_builder_error(
                    "constant Jacobian repeats a selected coordinate",
                ));
            }
        }
        let summaries =
            affine_analysis::summarize(&self.instructions, &[], selected.len(), |slot, index| {
                let source = self
                    .slots
                    .get(slot.0 as usize)
                    .ok_or(SymbolicLinearityFailure::InvalidProgram { instruction: index })?
                    .source();
                Ok(if let Some(&column) = columns.get(source) {
                    AffineSummary::variable(column, selected.len())
                } else {
                    AffineSummary::independent(selected.len())
                })
            })
            .map_err(|failure| match failure {
                affine_analysis::SummaryFailure::Symbolic(failure) => ir_builder_error(format!(
                    "cannot prove constant component Jacobian: {failure:?}"
                )),
                affine_analysis::SummaryFailure::Numerical { diagnostic, .. } => diagnostic,
            })?;
        let mut coefficients = Vec::new();
        for root in &self.roots {
            let summary = summaries
                .get(root.0 as usize)
                .ok_or_else(|| ir_builder_error("constant Jacobian root is unavailable"))?
                .scalar(self.instructions.len())
                .map_err(|error| {
                    ir_builder_error(format!("constant Jacobian root is not scalar: {error:?}"))
                })?;
            coefficients.extend(summary.coefficients.iter().map(|v| v.value));
        }
        Ok(ConstantSymbolJacobian {
            rows: self.roots.len(),
            columns: selected.len(),
            coefficients,
        })
    }

    pub(crate) fn bind_affine(
        &self,
        selected: &[ScalarSymbolCoordinate],
        bindings: &[(ScalarSymbolCoordinate, f64)],
    ) -> Result<BoundAffineScalarIr<ScalarSymbolCoordinate>, Diagnostic> {
        let bindings = bindings
            .iter()
            .map(|(c, v)| (c.clone(), AffineValue::constant(*v)))
            .collect::<Vec<_>>();
        self.bind_affine_values(selected, &bindings)
            .map(|rows| project(selected, &rows, |v| v.value))
    }

    pub(crate) fn bind_affine_direction(
        &self,
        selected: &[ScalarSymbolCoordinate],
        bindings: &[(ScalarSymbolCoordinate, f64, f64)],
    ) -> Result<
        (
            BoundAffineScalarIr<ScalarSymbolCoordinate>,
            BoundAffineScalarIr<ScalarSymbolCoordinate>,
        ),
        Diagnostic,
    > {
        let bindings = bindings
            .iter()
            .map(|(c, value, tangent)| {
                (
                    c.clone(),
                    AffineValue {
                        value: *value,
                        tangent: Some(*tangent),
                    },
                )
            })
            .collect::<Vec<_>>();
        let rows = self.bind_affine_values(selected, &bindings)?;
        Ok((
            project(selected, &rows, |v| v.value),
            project(selected, &rows, |v| v.tangent.unwrap_or(0.)),
        ))
    }

    fn bind_affine_values(
        &self,
        selected: &[ScalarSymbolCoordinate],
        bindings: &[(ScalarSymbolCoordinate, AffineValue)],
    ) -> Result<Vec<AffineSummary>, Diagnostic> {
        let mut columns = HashMap::with_capacity(selected.len());
        for (column, coordinate) in selected.iter().enumerate() {
            if columns.insert(coordinate, column).is_some() {
                return Err(ir_builder_error(
                    "affine input repeats a selected coordinate",
                ));
            }
        }
        let mut constants = HashMap::with_capacity(bindings.len());
        for (coordinate, value) in bindings {
            if !value.is_finite() {
                return Err(ir_builder_error("affine coordinate binding is nonfinite"));
            }
            if columns.contains_key(coordinate) {
                return Err(ir_builder_error(
                    "selected affine coordinate is bound as a constant",
                ));
            }
            if constants.insert(coordinate, *value).is_some() {
                return Err(ir_builder_error("affine coordinate binding is repeated"));
            }
        }
        for slot in &self.slots {
            if !columns.contains_key(slot.source()) && !constants.contains_key(slot.source()) {
                return Err(ir_builder_error("affine input coordinate is unbound"));
            }
        }
        let summaries =
            affine_analysis::summarize(&self.instructions, &[], selected.len(), |slot, index| {
                let coordinate = self
                    .slots
                    .get(slot.0 as usize)
                    .ok_or(SymbolicLinearityFailure::InvalidProgram { instruction: index })?
                    .source();
                Ok(if let Some(&column) = columns.get(coordinate) {
                    AffineSummary::variable(column, selected.len())
                } else {
                    AffineSummary::bound(constants[coordinate], selected.len())
                })
            })
            .map_err(|error| match error {
                affine_analysis::SummaryFailure::Symbolic(error) => {
                    ir_builder_error(format!("component affine proof failed: {error:?}"))
                }
                affine_analysis::SummaryFailure::Numerical { diagnostic, .. } => diagnostic,
            })?;
        self.roots
            .iter()
            .map(|root| {
                let summary = summaries
                    .get(root.0 as usize)
                    .ok_or_else(|| ir_builder_error("affine component root is unavailable"))?
                    .scalar(self.instructions.len())
                    .map_err(|error| {
                        ir_builder_error(format!("affine root is not scalar: {error:?}"))
                    })?;
                if summary.constant.is_none() {
                    return Err(ir_builder_error(
                        "affine component offset is not fully bound",
                    ));
                }
                Ok(summary.clone())
            })
            .collect()
    }
}

fn project(
    selected: &[ScalarSymbolCoordinate],
    rows: &[AffineSummary],
    value: impl Fn(AffineValue) -> f64,
) -> BoundAffineScalarIr<ScalarSymbolCoordinate> {
    BoundAffineScalarIr {
        selected_symbols: selected.to_vec(),
        residuals: rows.len(),
        coefficients: rows
            .iter()
            .flat_map(|row| row.coefficients.iter().map(|v| value(*v)))
            .collect(),
        offsets: rows
            .iter()
            .map(|row| value(row.constant.expect("validated bound offset")))
            .collect(),
    }
}
