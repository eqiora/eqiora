//! Bounded row-wise scalar property execution through the shared demand evaluator.
use super::*;
use eqiora_core::ValueLiteral;
use eqiora_schema::kernel::ExprId;

impl ScalarOperatorIr {
    /// Evaluate real scalar property rows in existing `symbols()` order.
    /// Each row has fresh memoization; total scalar work and retained outputs
    /// are each limited to one million before per-row projection/allocation.
    ///
    /// # Errors
    /// Rejects wrong row shapes/types, invalid active domains, and aggregate budgets.
    /// No partial batch result is returned on failure.
    pub fn evaluate_typed_batch(
        &self,
        roots: &[ExprId],
        inputs: &[Vec<ValueLiteral>],
    ) -> Result<Vec<Vec<ValueLiteral>>, Diagnostic> {
        let cost = self.projection_cost()?;
        check_budget(inputs.len(), cost, roots.len(), 1_000_000)?;
        if inputs.iter().any(|row| row.len() != self.symbols.len()) {
            return Err(ir_builder_error("typed batch has an incomplete row"));
        }
        let roots = roots
            .iter()
            .map(|id| {
                self.source_values
                    .get(id.index() as usize)
                    .copied()
                    .ok_or_else(|| ir_builder_error("batch output root is unavailable"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut results = Vec::with_capacity(inputs.len());
        for row in inputs {
            let projected = self.project_point(row, &roots)?;
            let values = self
                .symbols
                .iter()
                .copied()
                .zip(row)
                .collect::<HashMap<_, _>>();
            results.push(
                projected
                    .evaluate_value_trace(&projected.roots, &mut |symbol| {
                        values.get(&symbol).map(|value| (*value).clone())
                    })?
                    .0,
            );
        }
        Ok(results)
    }
}

fn check_budget(rows: usize, work: usize, outputs: usize, limit: usize) -> Result<(), Diagnostic> {
    if outputs > limit
        || rows.checked_mul(work.max(1)).is_none_or(|n| n > limit)
        || rows.checked_mul(outputs).is_none_or(|n| n > limit)
    {
        return Err(ir_builder_error(
            "typed batch exceeds its aggregate scalar work/output budget",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aggregate_batch_budgets_precede_row_allocation_and_check_overflow() {
        assert!(check_budget(2, 3, 4, 8).is_ok());
        assert!(check_budget(3, 3, 1, 8).is_err());
        assert!(check_budget(2, 1, 5, 8).is_err());
        assert!(check_budget(0, 1, 9, 8).is_err());
        assert!(check_budget(usize::MAX, 2, 1, usize::MAX).is_err());
    }
}
