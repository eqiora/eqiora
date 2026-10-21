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
    pub coordinates: &'a [std::ops::Range<usize>],
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
pub(crate) enum FieldPullback<'a> {
    Uniform {
        current: Option<f64>,
        rate: Option<f64>,
    },
    /// Numerical column order, with independent State and rate weights.
    Coordinates(&'a [[Option<f64>; 2]]),
}

impl FieldPullback<'_> {
    pub(crate) fn weight(self, column: usize, derivative: bool) -> Option<f64> {
        match self {
            Self::Uniform { current, rate } => {
                if derivative {
                    rate
                } else {
                    current
                }
            }
            Self::Coordinates(weights) => weights[column][usize::from(derivative)],
        }
    }

    pub(crate) fn validate(self, count: usize) -> Result<(), Diagnostic> {
        let valid = match self {
            Self::Uniform { current, rate } => {
                [current, rate].into_iter().flatten().all(f64::is_finite)
            }
            Self::Coordinates(weights) => {
                weights.len() == count
                    && weights
                        .iter()
                        .flatten()
                        .flatten()
                        .all(|weight| weight.is_finite())
            }
        };
        if !valid {
            return Err(invalid(
                "Field pullback requires exact finite coordinate weights",
            ));
        }
        Ok(())
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
        pullback: Option<FieldPullback<'_>>,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        let selected = self.derivatives(fields, sample, pullback);
        self.value.action(
            sample.physical,
            &self.inputs(fields, sample, point)?,
            selected.as_deref().map(FieldDerivatives),
        )
    }

    pub(super) fn input_action(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        point: &FieldValues<'_, S>,
        input_partial: &eqiora_ir::ScalarSymbolCoordinate,
        pullback: Option<FieldPullback<'_>>,
    ) -> Result<(S, Vec<[S; 2]>), Diagnostic> {
        let selected = self.derivatives(fields, sample, pullback);
        self.value.input_action(
            sample.physical,
            &self.inputs(fields, sample, point)?,
            input_partial,
            selected.as_deref().map(FieldDerivatives),
        )
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

    fn derivatives(
        &self,
        fields: &[RegionFieldLayout],
        sample: &FieldPoint<'_>,
        pullback: Option<FieldPullback<'_>>,
    ) -> Option<Vec<[bool; 2]>> {
        pullback.map(|pullback| {
            self.columns
                .iter()
                .map(|&(field, coordinate, derivative)| {
                    let layout = &fields[field];
                    let range = &sample.coordinates[field];
                    let width = range.len() / layout.range.len();
                    std::array::from_fn(|part| {
                        part < width
                            && sample.values[field].iter().enumerate().any(|(node, _)| {
                                let (component, basis) = sample.coordinate(field, node, coordinate);
                                let column = range.start
                                    + (node * layout.components + component) * width
                                    + part;
                                // The geometric basis owns dependency. A sampled zero Field
                                // value (or an explicit zero weight) does not freeze a partial.
                                basis != 0. && pullback.weight(column, derivative).is_some()
                            })
                    })
                })
                .collect()
        })
    }
}
