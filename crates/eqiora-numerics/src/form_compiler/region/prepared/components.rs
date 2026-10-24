//! Pointwise component differentials paired with the same physical basis samples.
use super::*;
use crate::form_compiler::bilinear::BasisComponent;
use crate::form_compiler::region::field_point::{
    BoundFieldComponent, FieldPoint, FieldPullback, FieldValues,
};
use crate::form_compiler::region::nonlinear::ComponentTerm;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Component<S: Coefficient> {
    row: usize,
    test: BasisComponent,
    value: BoundFieldComponent<S>,
    coefficient: usize,
    input_coefficient: crate::form_compiler::spatial::data::Data<S>,
    row_multiplier: f64,
    input_partial: Option<eqiora_ir::ScalarSymbolCoordinate>,
}

impl<S: Coefficient> Component<S> {
    pub(super) fn new(
        row: usize,
        term: &ComponentTerm<S>,
        fields: &[RegionFieldLayout],
        coefficient: usize,
        row_multiplier: f64,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            row,
            test: term.test,
            value: BoundFieldComponent::new(&term.value, fields, true)?,
            coefficient,
            input_coefficient: term.coefficient.clone(),
            row_multiplier,
            input_partial: term.input_partial.clone(),
        })
    }
}

impl<S: Coefficient, P: Projection<S, Scalar = f64>> PreparedRegionCell<S, P> {
    pub(super) fn add_component_action(
        &self,
        term: &Component<S>,
        sample: &Sample<S>,
        point: &FieldValues<'_, S>,
        pullback: Option<FieldPullback<'_>>,
        action: &mut RegionLinearization,
    ) -> Result<(), Diagnostic> {
        let basis = FieldPoint {
            coordinates: &self.coordinates,
            physical: &sample.physical,
            values: &sample.values,
            gradients: &sample.gradients,
        };
        let (value, gradient) = if let Some(input_partial) = &term.input_partial {
            term.value
                .input_action(&self.fields, &basis, point, input_partial, pullback)?
        } else {
            term.value.action(&self.fields, &basis, point, pullback)?
        };
        let test = &self.fields[term.row];
        let width = P::width(&test.value_type)?;
        let count = action.residual.len();
        for node in 0..sample.values[term.row].len() {
            let (component, test_basis) = basis.coordinate(term.row, node, term.test);
            let row =
                self.coordinates[term.row].start + (node * test.components + component) * width;
            let weight = sample.coefficients[term.coefficient] * test_basis;
            for test_coordinate in 0..width {
                let r = row + test_coordinate;
                action.residual[r] += P::component(weight * value, test_coordinate);
                if let Some(pullback) = pullback {
                    for ((column, component, derivative), partials) in
                        term.value.columns().iter().zip(&gradient)
                    {
                        let trial = &self.fields[*column];
                        let trial_width = P::width(&trial.value_type)?;
                        for node in 0..sample.values[*column].len() {
                            let (component, trial_basis) =
                                basis.coordinate(*column, node, *component);
                            let column = self.coordinates[*column].start
                                + (node * trial.components + component) * trial_width;
                            for (coordinate, partial) in
                                partials.iter().take(trial_width).enumerate()
                            {
                                let Some(direction) =
                                    pullback.weight(column + coordinate, *derivative)
                                else {
                                    continue;
                                };
                                action.jacobian[r * count + column + coordinate] +=
                                    P::component(weight * *partial, test_coordinate)
                                        * trial.scale
                                        * trial_basis
                                        * direction;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<S: Coefficient, P: Projection<S, Scalar = f64>> PreparedRegionCell<S, P> {
    pub(crate) fn component_input_actions(
        &self,
        state: &[f64],
        rates: &[f64],
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
    ) -> Result<Vec<crate::form_compiler::spatial::data::InputAction<S>>, Diagnostic> {
        use crate::form_compiler::spatial::data::InputAction;
        use num_rational::BigRational;
        use num_traits::Zero;
        if !self.independent_coordinates
            || state.len() != self.affine.rows()
            || rates.len() != state.len()
        {
            return Err(invalid(
                "input actions require exact independent State/rate coordinates",
            ));
        }
        let current = P::physical_point(&self.fields, &self.coordinates, state);
        let rates = P::physical_point(&self.fields, &self.coordinates, rates);
        let point = FieldValues {
            current: &current,
            rates: Some(&rates),
        };
        let mut actions = Vec::new();
        for sample in &self.samples {
            let basis = FieldPoint {
                coordinates: &self.coordinates,
                physical: &sample.physical,
                values: &sample.values,
                gradients: &sample.gradients,
            };
            for term in &self.components {
                if term.input_partial.is_some() {
                    return Err(invalid(
                        "projected input actions require the primal component",
                    ));
                }
                for (atom, coefficient) in term.value.input_atoms(
                    &self.fields,
                    &basis,
                    &point,
                    coordinate,
                    &term.input_coefficient,
                )? {
                    let mut coefficients = vec![BigRational::zero(); self.affine.rows()];
                    let test = &self.fields[term.row];
                    let width = P::width(&test.value_type)?;
                    for node in 0..sample.values[term.row].len() {
                        let (component, test_basis) = basis.coordinate(term.row, node, term.test);
                        let row = self.coordinates[term.row].start
                            + (node * test.components + component) * width;
                        let weight = <S as From<f64>>::from(
                            sample.weight * term.row_multiplier * test_basis,
                        );
                        for part in 0..width {
                            let real = BigRational::from_float(P::component(weight, part))
                                .ok_or_else(|| {
                                    invalid("input atom has a non-finite test action")
                                })?;
                            coefficients[row + part] += &coefficient.re * real;
                            if !coefficient.im.is_zero() {
                                let unit = S::imaginary_unit().ok_or_else(|| {
                                    invalid("input atom exceeds its scalar domain")
                                })?;
                                let imaginary =
                                    BigRational::from_float(P::component(weight * unit, part))
                                        .ok_or_else(|| {
                                            invalid("input atom has a non-finite test action")
                                        })?;
                                coefficients[row + part] += &coefficient.im * imaginary;
                            }
                        }
                    }
                    actions.push(InputAction { atom, coefficients });
                }
            }
        }
        Ok(actions)
    }
}
