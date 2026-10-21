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
        // Retained primal checks can read a coordinate absent from every
        // output. Prove that independence before analyzing affine coefficients:
        // a variable coefficient in a primal-only expression is not an output
        // coupling. Evaluation still owns all original domain obligations.
        let bindings = self
            .slots
            .iter()
            .map(|slot| {
                columns
                    .get(slot.source())
                    .map_or(InputBinding::Frozen, |&column| {
                        InputBinding::Unknown(column)
                    })
            })
            .collect::<Vec<_>>();
        let active = activity::numerical_activity(&self.instructions, &bindings)?;
        if self
            .roots
            .iter()
            .all(|root| active.get(root.0 as usize).is_some_and(|active| !active))
        {
            return Ok(ConstantSymbolJacobian {
                rows: self.roots.len(),
                columns: selected.len(),
                coefficients: vec![0.; self.roots.len() * selected.len()],
            });
        }
        let summaries = affine_analysis::summarize(
            &self.instructions,
            &self.typed_constants,
            selected.len(),
            |slot, index| {
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
            },
        )
        .map_err(|failure| failure.into_diagnostic("cannot prove constant component Jacobian"))?;
        let mut coefficients = Vec::new();
        for root in &self.roots {
            let summary = summaries
                .get(root.0 as usize)
                .ok_or_else(|| ir_builder_error("constant Jacobian root is unavailable"))?
                .scalar(self.instructions.len())
                .map_err(|failure| {
                    failure.into_diagnostic("constant Jacobian root is not scalar")
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
            .and_then(|rows| project(selected, &rows, |v| v.value))
    }

    pub(crate) fn bind_affine_direction(
        &self,
        selected: &[ScalarSymbolCoordinate],
        bindings: &[(ScalarSymbolCoordinate, f64, Option<f64>)],
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
                        tangent: tangent.map(Scaled::new),
                    },
                )
            })
            .collect::<Vec<_>>();
        let rows = self.bind_affine_values(selected, &bindings)?;
        Ok((
            project(selected, &rows, |v| v.value)?,
            project(selected, &rows, |v| {
                v.tangent.unwrap_or(Scaled::ZERO).value()
            })?,
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
        let summaries = affine_analysis::summarize(
            &self.instructions,
            &self.typed_constants,
            selected.len(),
            |slot, index| {
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
            },
        )
        .map_err(|failure| failure.into_diagnostic("component affine proof failed"))?;
        for root in &self.primal_checks {
            summaries[root.0 as usize]
                .scalar(self.instructions.len())
                .map_err(|failure| failure.into_diagnostic("affine primal domain is invalid"))?;
        }
        self.roots
            .iter()
            .map(|root| {
                let summary = summaries
                    .get(root.0 as usize)
                    .ok_or_else(|| ir_builder_error("affine component root is unavailable"))?
                    .scalar(self.instructions.len())
                    .map_err(|failure| failure.into_diagnostic("affine root is not scalar"))?;
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
) -> Result<BoundAffineScalarIr<ScalarSymbolCoordinate>, Diagnostic> {
    let result = BoundAffineScalarIr {
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
    };
    for (index, &value) in result
        .coefficients
        .iter()
        .chain(&result.offsets)
        .enumerate()
    {
        require_finite_value(value, "affine output", index)?;
    }
    Ok(result)
}
