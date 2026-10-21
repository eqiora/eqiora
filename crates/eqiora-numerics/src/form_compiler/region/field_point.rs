//! One physical Field interpolation for volume and constitutive trace actions.
use super::{RegionFieldLayout, invalid};
use crate::form_compiler::bilinear::BasisComponent;
use crate::form_compiler::spatial::data::field_components::{FieldComponent, FieldDerivatives};
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
        scale: f64,
    ) -> S {
        self.values[field]
            .iter()
            .enumerate()
            .map(|(node, _)| {
                let (component, basis) = self.coordinate(field, node, coordinate);
                coefficients[layout.range.start + node * layout.components + component]
                    * (scale * basis)
            })
            .sum()
    }
}

/// Normalized State coefficients and independent physical rate coefficients.
pub(super) struct FieldValues<'a, S: Coefficient> {
    pub current: &'a [S],
    pub rates: Option<&'a [S]>,
}

/// Pull back independent physical State/rate partials into candidate coordinates.
/// `None` leaves a partial undemanded; a zero weight still demands that partial.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldPullback {
    pub current: Option<f64>,
    pub rate: Option<f64>,
}

impl FieldPullback {
    fn derivatives(self) -> FieldDerivatives {
        FieldDerivatives {
            current: self.current.is_some(),
            rate: self.rate.is_some(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct BoundFieldComponent<S: Coefficient> {
    value: Arc<FieldComponent<S>>,
    columns: Vec<(usize, BasisComponent, bool)>,
}

impl<S: Coefficient> BoundFieldComponent<S> {
    pub(super) fn new(
        value: &Arc<FieldComponent<S>>,
        fields: &[RegionFieldLayout],
        rates: bool,
    ) -> Result<Self, Diagnostic> {
        let columns: Vec<_> = value
            .columns()
            .map(|(id, component, derivative)| {
                if derivative && !rates {
                    return Err(invalid("component rate requires a bound temporal action"));
                }
                fields
                    .iter()
                    .position(|field| field.field == id)
                    .map(|field| (field, component, derivative))
                    .ok_or_else(|| invalid("component trial lacks an algebraic Field binding"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            value: Arc::clone(value),
            columns,
        })
    }

    pub(super) fn columns(&self) -> &[(usize, BasisComponent, bool)] {
        &self.columns
    }
}

impl<S: Coefficient> BoundFieldComponent<S> {
    pub(super) fn action(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        point: &FieldValues<'_, S>,
        pullback: Option<FieldPullback>,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        let (value, gradient) = self.value.action(
            sample.physical,
            &self.inputs(fields, sample, point)?,
            pullback.map(FieldPullback::derivatives),
        )?;
        Ok((value, self.pullback(gradient, pullback)))
    }

    pub(super) fn parameter_action(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        point: &FieldValues<'_, S>,
        parameter: &eqiora_ir::ScalarSymbolCoordinate,
        pullback: Option<FieldPullback>,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        let (value, gradient) = self.value.parameter_action(
            sample.physical,
            &self.inputs(fields, sample, point)?,
            parameter,
            pullback.map(FieldPullback::derivatives),
        )?;
        Ok((value, self.pullback(gradient, pullback)))
    }

    fn inputs(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        point: &FieldValues<'_, S>,
    ) -> Result<Vec<S>, Diagnostic> {
        self.columns
            .iter()
            .map(|(field, coordinate, derivative)| {
                let (coefficients, scale) = if *derivative {
                    (
                        point
                            .rates
                            .ok_or_else(|| invalid("component rate lacks consumed history"))?,
                        1.,
                    )
                } else {
                    (point.current, fields[*field].scale)
                };
                Ok(sample.evaluate(*field, &fields[*field], coefficients, *coordinate, scale))
            })
            .collect()
    }

    fn pullback(&self, mut gradient: Vec<[S; 2]>, pullback: Option<FieldPullback>) -> Vec<[S; 2]> {
        if let Some(pullback) = pullback {
            for ((_, _, derivative), parts) in self.columns.iter().zip(&mut gradient) {
                let scale = if *derivative {
                    pullback.rate
                } else {
                    pullback.current
                };
                if let Some(scale) = scale {
                    for part in parts {
                        *part = *part * scale;
                    }
                }
            }
        }
        gradient
    }
}
