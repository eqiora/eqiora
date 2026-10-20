//! Physical Field and Parameter products use the same bound scalar IR point.
use super::*;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};

impl FieldComponent<f64> {
    pub(in crate::form_compiler) fn action(
        &self,
        physical: &[f64],
        values: &[f64],
        derivative: bool,
    ) -> Result<(f64, Vec<f64>), Diagnostic> {
        self.product(physical, values, None, derivative)
    }

    /// Direct physical Parameter partial with all candidate Field coordinates fixed.
    pub(in crate::form_compiler) fn parameter_action(
        &self,
        physical: &[f64],
        values: &[f64],
        parameter: &ScalarSymbolCoordinate,
        derivative: bool,
    ) -> Result<(f64, Vec<f64>), Diagnostic> {
        self.product(physical, values, Some(parameter), derivative)
    }

    fn product(
        &self,
        physical: &[f64],
        values: &[f64],
        parameter: Option<&ScalarSymbolCoordinate>,
        derivative: bool,
    ) -> Result<(f64, Vec<f64>), Diagnostic> {
        let inputs = self.point(physical, values)?;
        let row = &self.rows[0];
        if parameter.is_none() && !derivative {
            return row.evaluate(&inputs).map(|value| (value, Vec::new()));
        }
        let direction =
            |coordinate: &ScalarSymbolCoordinate| f64::from(Some(coordinate) == parameter);
        let mut roles = Vec::with_capacity(inputs.len());
        let mut tangents = Vec::new();
        let mut unknowns = Vec::new();
        for coordinate in row.symbols() {
            if let Some(index) = self.selected.iter().position(|value| value == coordinate) {
                roles.push(if derivative {
                    unknowns.push(index);
                    DifferentiationRole::Unknown
                } else {
                    DifferentiationRole::Frozen
                });
                continue;
            }
            let input = self.binding(coordinate)?;
            if parameter.is_none() || input.constant_in_direction(coordinate, &direction)? {
                roles.push(DifferentiationRole::Frozen);
            } else {
                // Structural activity survives a zero sampled inner tangent.
                roles.push(DifferentiationRole::Parameter);
                tangents.push(
                    input
                        .derivative(coordinate, &direction, physical.len())?
                        .evaluate(physical)?,
                );
            }
        }
        let action = row.linearize(&inputs, &roles)?;
        let mut value = [0.];
        if parameter.is_some() {
            action.jvp(RelationTangent::Parameter(&tangents), &mut value)?;
        } else {
            action.primal(&mut value)?;
        }
        if !derivative {
            return Ok((value[0], Vec::new()));
        }
        let mut gradient = vec![0.; unknowns.len()];
        if parameter.is_some() {
            let mut field_direction = vec![0.; unknowns.len()];
            let mixed = vec![0.; tangents.len()];
            for (index, entry) in gradient.iter_mut().enumerate() {
                field_direction[index] = 1.;
                // Prescribed inputs do not depend on candidate Fields, so the
                // mixed input tangent is zero. The scalar IR owns the Hessian.
                let mut result = [0.];
                action.mixed_jvp(
                    RelationTangent::Unknown(&field_direction),
                    RelationTangent::Parameter(&tangents),
                    RelationTangent::Parameter(&mixed),
                    &mut result,
                )?;
                *entry = result[0];
                field_direction[index] = 0.;
            }
        } else {
            action.vjp(&[1.], RelationCotangent::Unknown(&mut gradient))?;
        }
        let mut selected_gradient = vec![0.; self.selected.len()];
        for (index, entry) in unknowns.into_iter().zip(gradient) {
            selected_gradient[index] += entry;
        }
        Ok((
            value[0],
            self.columns
                .iter()
                .map(|column| selected_gradient[column.real.expect("checked real column")])
                .collect(),
        ))
    }

    fn binding(&self, coordinate: &ScalarSymbolCoordinate) -> Result<&Input<f64>, Diagnostic> {
        self.bindings
            .iter()
            .find(|(value, _)| value == coordinate)
            .map(|(_, input)| input)
            .ok_or_else(|| super::super::super::invalid("component input lacks its exact binding"))
    }

    fn point(&self, physical: &[f64], values: &[f64]) -> Result<Vec<f64>, Diagnostic> {
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
        for coordinate in row.symbols() {
            if let Some(index) = self.selected.iter().position(|value| value == coordinate) {
                inputs.push(selected[index]);
            } else {
                inputs.push(self.binding(coordinate)?.evaluate(physical)?);
            }
        }
        Ok(inputs)
    }
}
