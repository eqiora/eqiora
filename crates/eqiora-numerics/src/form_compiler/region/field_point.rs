//! One physical Field interpolation for volume and constitutive trace actions.
use super::{RegionFieldLayout, invalid};
use crate::form_compiler::bilinear::BasisComponent;
use crate::form_compiler::spatial::data::field_components::FieldComponent;
use crate::spatial_expression::Coefficient;
use eqiora_core::Diagnostic;
use std::sync::Arc;

/// Parent-cell basis values and gradients at one method-owned physical point.
/// A vanishing value trace does not imply a vanishing normal derivative.
pub(super) struct FieldPoint<'a> {
    pub physical: &'a [f64],
    pub values: &'a [Vec<f64>],
    pub gradients: &'a [Vec<[f64; 3]>],
}

impl FieldPoint<'_> {
    pub(super) fn coordinate(
        &self,
        field: usize,
        node: usize,
        coordinate: BasisComponent,
    ) -> (usize, f64) {
        match coordinate {
            BasisComponent::Value(component) => (component, self.values[field][node]),
            BasisComponent::Gradient { component, axis } => {
                (component, self.gradients[field][node][axis])
            }
        }
    }

    pub(super) fn evaluate(
        &self,
        field: usize,
        layout: &RegionFieldLayout,
        coefficients: &[f64],
        coordinate: BasisComponent,
    ) -> f64 {
        self.values[field]
            .iter()
            .enumerate()
            .map(|(node, _)| {
                let (component, basis) = self.coordinate(field, node, coordinate);
                layout.scale
                    * coefficients[layout.range.start + node * layout.components + component]
                    * basis
            })
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct BoundFieldComponent<S: Coefficient> {
    value: Arc<FieldComponent<S>>,
    columns: Vec<(usize, BasisComponent)>,
}

impl<S: Coefficient> BoundFieldComponent<S> {
    pub(super) fn new(
        value: &Arc<FieldComponent<S>>,
        fields: &[RegionFieldLayout],
    ) -> Result<Self, Diagnostic> {
        let columns = value
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
            value: Arc::clone(value),
            columns,
        })
    }

    pub(super) fn columns(&self) -> &[(usize, BasisComponent)] {
        &self.columns
    }
}

impl BoundFieldComponent<f64> {
    pub(super) fn action(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        coefficients: &[f64],
        derivative: bool,
    ) -> Result<(f64, Vec<f64>), Diagnostic> {
        let inputs = self
            .columns
            .iter()
            .map(|(field, coordinate)| {
                sample.evaluate(*field, &fields[*field], coefficients, *coordinate)
            })
            .collect::<Vec<_>>();
        self.value.action(sample.physical, &inputs, derivative)
    }
}
