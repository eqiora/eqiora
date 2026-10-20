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

    pub(super) fn evaluate<S: Coefficient>(
        &self,
        field: usize,
        layout: &RegionFieldLayout,
        coefficients: &[S],
        coordinate: BasisComponent,
    ) -> S {
        self.values[field]
            .iter()
            .enumerate()
            .map(|(node, _)| {
                let (component, basis) = self.coordinate(field, node, coordinate);
                coefficients[layout.range.start + node * layout.components + component]
                    * (layout.scale * basis)
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

impl<S: Coefficient> BoundFieldComponent<S> {
    pub(super) fn action(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        coefficients: &[S],
        derivative: bool,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        self.value.action(
            sample.physical,
            &self.inputs(fields, sample, coefficients),
            derivative,
        )
    }

    pub(super) fn parameter_action(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        coefficients: &[S],
        parameter: &eqiora_ir::ScalarSymbolCoordinate,
        derivative: bool,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        self.value.parameter_action(
            sample.physical,
            &self.inputs(fields, sample, coefficients),
            parameter,
            derivative,
        )
    }

    fn inputs(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        coefficients: &[S],
    ) -> Vec<S> {
        self.columns
            .iter()
            .map(|(field, coordinate)| {
                sample.evaluate(*field, &fields[*field], coefficients, *coordinate)
            })
            .collect()
    }
}
