//! Component coordinates share the scalar SSA affine proof without synthetic symbols.
use super::*;

impl ScalarInputOperatorIr {
    pub(crate) fn bind_affine(
        &self,
        selected: &[ScalarSymbolCoordinate],
        bindings: &[(ScalarSymbolCoordinate, f64)],
    ) -> Result<BoundAffineScalarIr<ScalarSymbolCoordinate>, Diagnostic> {
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
            affine_analysis::summarize(&self.instructions, selected.len(), |slot, index| {
                let coordinate = self
                    .slots
                    .get(slot.0 as usize)
                    .ok_or(SymbolicLinearityFailure::InvalidProgram { instruction: index })?
                    .source();
                Ok(if let Some(&column) = columns.get(coordinate) {
                    AffineSummary::variable(column, selected.len())
                } else {
                    AffineSummary::constant(constants[coordinate], selected.len())
                })
            })
            .map_err(|error| match error {
                affine_analysis::SummaryFailure::Symbolic(error) => {
                    ir_builder_error(format!("component affine proof failed: {error:?}"))
                }
                affine_analysis::SummaryFailure::Numerical { diagnostic, .. } => diagnostic,
            })?;
        let mut coefficients = Vec::new();
        let mut offsets = Vec::new();
        for root in &self.roots {
            let summary = summaries
                .get(root.0 as usize)
                .ok_or_else(|| ir_builder_error("affine component root is unavailable"))?;
            coefficients.extend_from_slice(&summary.coefficients);
            offsets.push(
                summary.constant.ok_or_else(|| {
                    ir_builder_error("affine component offset is not fully bound")
                })?,
            );
        }
        Ok(BoundAffineScalarIr {
            selected_symbols: selected.to_vec(),
            residuals: self.roots.len(),
            coefficients,
            offsets,
        })
    }
}
