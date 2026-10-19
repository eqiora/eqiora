//! Pointwise component differentials paired with the same physical basis samples.
use super::*;
use crate::form_compiler::bilinear::BasisComponent;
use crate::form_compiler::region::nonlinear::ComponentTerm;
use crate::form_compiler::spatial::data::field_components::FieldComponent;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Component<S: Coefficient> {
    row: usize,
    test: BasisComponent,
    value: Arc<FieldComponent<S>>,
    columns: Vec<(usize, BasisComponent)>,
    coefficient: usize,
}

impl<S: Coefficient> Component<S> {
    pub(super) fn new(
        row: usize,
        term: &ComponentTerm<S>,
        fields: &[RegionFieldLayout],
        coefficient: usize,
    ) -> Result<Self, Diagnostic> {
        let columns = term
            .value
            .columns()
            .map(|(id, component)| {
                fields
                    .iter()
                    .position(|field| field.field == id)
                    .map(|field| (field, component))
                    .ok_or_else(|| invalid("component trial lacks an algebraic Field binding"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            row,
            test: term.test,
            value: Arc::clone(&term.value),
            columns,
            coefficient,
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
        let trial_values = self
            .columns
            .iter()
            .map(|(column, component)| {
                let field = &fields[*column];
                sample.values[*column]
                    .iter()
                    .enumerate()
                    .map(|(node, _)| {
                        let (component, basis) = coordinate(sample, *column, node, *component);
                        field.scale
                            * point[field.range.start + node * field.components + component]
                            * basis
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let (value, gradient) = self
            .value
            .action(&sample.physical, &trial_values, derivative)?;
        let test = &fields[self.row];
        let count = point.len();
        for node in 0..sample.values[self.row].len() {
            let (component, basis) = coordinate(sample, self.row, node, self.test);
            let r = test.range.start + node * test.components + component;
            let weight = basis * sample.coefficients[self.coefficient];
            action.residual[r] += weight * value;
            if derivative {
                for ((column, component), derivative) in self.columns.iter().zip(&gradient) {
                    let trial = &fields[*column];
                    for node in 0..sample.values[*column].len() {
                        let (component, basis) = coordinate(sample, *column, node, *component);
                        let c = trial.range.start + node * trial.components + component;
                        action.jacobian[r * count + c] += weight * derivative * trial.scale * basis;
                    }
                }
            }
        }
        Ok(())
    }
}

fn coordinate<S: Coefficient>(
    sample: &Sample<S>,
    field: usize,
    node: usize,
    coordinate: BasisComponent,
) -> (usize, f64) {
    match coordinate {
        BasisComponent::Value(component) => (component, sample.values[field][node]),
        BasisComponent::Gradient { component, axis } => {
            (component, sample.gradients[field][node][axis])
        }
    }
}
