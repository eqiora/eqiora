//! Physical candidate coordinates use the ordinary scalar IR primal and VJP.
use super::*;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationCotangent};

impl FieldComponent<f64> {
    pub(in crate::form_compiler) fn action(
        &self,
        physical: &[f64],
        values: &[f64],
        derivative: bool,
    ) -> Result<(f64, Vec<f64>), Diagnostic> {
        if self.rows.len() != 1
            || values.len() != self.columns.len()
            || values.iter().any(|value| !value.is_finite())
            || self
                .columns
                .iter()
                .any(|column| column.real.is_none() || column.imaginary.is_some())
        {
            return Err(super::super::super::invalid(
                "nonlinear component action requires exact finite real trial coordinates",
            ));
        }
        let mut selected = vec![0.; self.selected.len()];
        for (column, value) in self.columns.iter().zip(values) {
            selected[column.real.expect("checked real column")] = *value;
        }
        let row = &self.rows[0];
        let mut inputs = Vec::with_capacity(row.symbols().len());
        let mut roles = Vec::with_capacity(row.symbols().len());
        let mut unknowns = Vec::new();
        for coordinate in row.symbols() {
            if let Some(index) = self.selected.iter().position(|value| value == coordinate) {
                inputs.push(selected[index]);
                roles.push(DifferentiationRole::Unknown);
                unknowns.push(index);
            } else {
                let (_, input) = self
                    .bindings
                    .iter()
                    .find(|(value, _)| value == coordinate)
                    .ok_or_else(|| {
                        super::super::super::invalid("component input lacks its exact binding")
                    })?;
                inputs.push(input.evaluate(physical)?);
                roles.push(DifferentiationRole::Frozen);
            }
        }
        if !derivative {
            return row.evaluate(&inputs).map(|value| (value, Vec::new()));
        }
        let action = row.linearize(&inputs, &roles)?;
        let mut primal = [0.];
        action.primal(&mut primal)?;
        let mut gradient = vec![0.; unknowns.len()];
        action.vjp(&[1.], RelationCotangent::Unknown(&mut gradient))?;
        let mut selected_gradient = vec![0.; selected.len()];
        for (index, value) in unknowns.into_iter().zip(gradient) {
            selected_gradient[index] += value;
        }
        Ok((
            primal[0],
            self.columns
                .iter()
                .map(|column| selected_gradient[column.real.expect("checked real column")])
                .collect(),
        ))
    }
}
