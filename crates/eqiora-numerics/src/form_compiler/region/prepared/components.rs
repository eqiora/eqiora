//! Pointwise component differentials paired with the same physical basis samples.
use super::*;
use crate::form_compiler::bilinear::BasisComponent;
use crate::form_compiler::region::field_point::{BoundFieldComponent, FieldPoint};
use crate::form_compiler::region::nonlinear::ComponentTerm;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Component<S: Coefficient> {
    row: usize,
    test: BasisComponent,
    value: BoundFieldComponent<S>,
    coefficient: usize,
    parameter: Option<eqiora_ir::ScalarSymbolCoordinate>,
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
            value: BoundFieldComponent::new(&term.value, fields)?,
            coefficient,
            parameter: term.parameter.clone(),
        })
    }
}

impl Component<f64> {
    pub(super) fn add_action(
        &self,
        fields: &[RegionFieldLayout],
        sample: &Sample<f64>,
        point: &[f64],
        derivative: bool,
        action: &mut RegionLinearization,
    ) -> Result<(), Diagnostic> {
        let basis = FieldPoint {
            physical: &sample.physical,
            values: &sample.values,
            gradients: &sample.gradients,
        };
        let (value, gradient) = if let Some(parameter) = &self.parameter {
            self.value
                .parameter_action(fields, &basis, point, parameter, derivative)?
        } else {
            self.value.action(fields, &basis, point, derivative)?
        };
        let test = &fields[self.row];
        let count = point.len();
        for node in 0..sample.values[self.row].len() {
            let (component, test_basis) = basis.coordinate(self.row, node, self.test);
            let r = test.range.start + node * test.components + component;
            let weight = test_basis * sample.coefficients[self.coefficient];
            action.residual[r] += weight * value;
            if derivative {
                for ((column, component), derivative) in self.value.columns().iter().zip(&gradient)
                {
                    let trial = &fields[*column];
                    for node in 0..sample.values[*column].len() {
                        let (component, trial_basis) = basis.coordinate(*column, node, *component);
                        let c = trial.range.start + node * trial.components + component;
                        action.jacobian[r * count + c] +=
                            weight * derivative * trial.scale * trial_basis;
                    }
                }
            }
        }
        Ok(())
    }
}
