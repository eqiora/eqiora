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
    input_partial: Option<eqiora_ir::ScalarSymbolCoordinate>,
}

impl<S: Coefficient> Component<S> {
    pub(super) fn new(
        row: usize,
        term: &ComponentTerm<S>,
        fields: &[RegionFieldLayout],
        coefficient: usize,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            row,
            test: term.test,
            value: BoundFieldComponent::new(&term.value, fields, true)?,
            coefficient,
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
