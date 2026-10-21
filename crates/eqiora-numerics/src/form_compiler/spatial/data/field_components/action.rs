//! Physical Field and Parameter products use the same bound scalar IR point.
use super::*;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};

impl<S: Coefficient> FieldComponent<S> {
    /// Each physical input column has independent real and imaginary derivatives.
    /// An absent scalar coordinate contributes zero; the output retains its domain.
    pub(in crate::form_compiler) fn action(
        &self,
        physical: &[f64],
        values: &[S],
        derivatives: Option<FieldDerivatives<'_>>,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        self.product(physical, values, None, derivatives)
    }

    /// Direct physical Parameter partial with all candidate Field coordinates fixed.
    pub(in crate::form_compiler) fn parameter_action(
        &self,
        physical: &[f64],
        values: &[S],
        parameter: &ScalarSymbolCoordinate,
        derivatives: Option<FieldDerivatives<'_>>,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        self.product(physical, values, Some(parameter), derivatives)
    }

    fn product(
        &self,
        physical: &[f64],
        values: &[S],
        parameter: Option<&ScalarSymbolCoordinate>,
        derivatives: Option<FieldDerivatives<'_>>,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        if derivatives.is_some_and(|parts| parts.0.len() != self.columns.len()) {
            return Err(super::super::super::invalid(
                "component derivative demand lacks exact column coverage",
            ));
        }
        if self.rows.is_empty() || self.rows.len() > 2 {
            return Err(super::super::super::invalid(
                "component action has no exact scalar rows",
            ));
        }
        let mut values_by_part = [0.; 2];
        let mut gradients = [Vec::new(), Vec::new()];
        for row in &self.rows {
            let part = usize::from(row.is_imaginary());
            let (value, gradient) =
                self.row_product(row, physical, values, parameter, derivatives)?;
            values_by_part[part] = value;
            gradients[part] = gradient;
        }
        let assemble = |parts: [f64; 2]| {
            S::from_parts(parts[0], parts[1]).ok_or_else(|| {
                super::super::super::invalid("component value exceeds its numeric scalar domain")
            })
        };
        let value = assemble(values_by_part)?;
        let mut columns = Vec::new();
        if derivatives.is_some() {
            for column in &self.columns {
                let coordinate = |index: Option<usize>| -> Result<S, Diagnostic> {
                    assemble(std::array::from_fn(|part| {
                        index
                            .and_then(|index| gradients[part].get(index))
                            .copied()
                            .unwrap_or(0.)
                    }))
                };
                columns.push([coordinate(column.real)?, coordinate(column.imaginary)?]);
            }
        }
        Ok((value, columns))
    }

    fn row_product(
        &self,
        row: &ComponentScalarRow,
        physical: &[f64],
        values: &[S],
        parameter: Option<&ScalarSymbolCoordinate>,
        derivatives: Option<FieldDerivatives<'_>>,
    ) -> Result<(f64, Vec<f64>), Diagnostic> {
        let inputs = self.point(row, physical, values)?;
        if parameter.is_none() && derivatives.is_none() {
            return row.evaluate(&inputs).map(|value| (value, Vec::new()));
        }
        let direction =
            |coordinate: &ScalarSymbolCoordinate| f64::from(Some(coordinate) == parameter);
        let mut roles = Vec::with_capacity(inputs.len());
        let mut tangents = Vec::new();
        let mut unknowns = Vec::new();
        for coordinate in row.symbols() {
            if let Some(index) = self.selected.iter().position(|value| value == coordinate) {
                roles.push(
                    if derivatives.is_some_and(|selection| selection.includes(&self.columns, index))
                    {
                        unknowns.push(index);
                        DifferentiationRole::Unknown
                    } else {
                        DifferentiationRole::Frozen
                    },
                );
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
        if derivatives.is_none() {
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
        Ok((value[0], selected_gradient))
    }

    fn binding(&self, coordinate: &ScalarSymbolCoordinate) -> Result<&Input<S>, Diagnostic> {
        self.bindings
            .iter()
            .find(|(value, _)| value == coordinate)
            .map(|(_, input)| input)
            .ok_or_else(|| super::super::super::invalid("component input lacks its exact binding"))
    }

    fn point(
        &self,
        row: &ComponentScalarRow,
        physical: &[f64],
        values: &[S],
    ) -> Result<Vec<f64>, Diagnostic> {
        if values.len() != self.columns.len()
            || values.iter().any(|value| !value.is_finite())
            || self.columns.iter().zip(values).any(|(column, value)| {
                column.scalar_domain == ScalarDomain::Real && value.im() != 0.
            })
        {
            return Err(super::super::super::invalid(
                "component action requires exact finite typed trial values",
            ));
        }
        let mut selected = vec![0.; self.selected.len()];
        for (column, value) in self.columns.iter().zip(values) {
            if let Some(index) = column.real {
                selected[index] = value.re();
            }
            if let Some(index) = column.imaginary {
                selected[index] = value.im();
            }
        }
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
