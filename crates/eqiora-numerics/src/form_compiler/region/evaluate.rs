use crate::form_compiler::region::integration::IntegralField;
use crate::spatial_expression::Coefficient;
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryLinearization, AffineGeometryMap, GeometryMap, QuadratureRule};

use super::binding::BoundRegionForm;
use super::invalid;

impl<S: Coefficient> BoundRegionForm<S> {
    /// Validate local inputs without performing quadrature or constructing contributions.
    pub(crate) fn validate_cell(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        previous: &BTreeMap<RawId, Vec<S>>,
    ) -> Result<(), Diagnostic> {
        self.validate_geometry(geometry, quadrature)?;
        if previous.len() != self.previous.len()
            || self.previous.iter().any(|(field, layout)| {
                previous.get(field).is_none_or(|values| {
                    values.len() != layout.range.len()
                        || values.iter().any(|value| !value.is_finite())
                })
            })
        {
            return Err(invalid(
                "previous coefficients require exact consumed Field coverage, shape and finite values",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_geometry(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<(), Diagnostic> {
        if geometry.reference_cell() != self.reference
            || quadrature.reference_cell() != self.reference
            || geometry.physical_dimension() != self.form.dimension
        {
            return Err(invalid(
                "region geometry and quadrature must match bound reference support",
            ));
        }
        if quadrature
            .polynomial_exactness()
            .is_none_or(|order| order < self.required_polynomial_exactness())
        {
            return Err(invalid(
                "quadrature does not cover region polynomial integrand degree",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare_affine_actions(
        &self,
        geometry: &AffineGeometryMap,
        direction: Option<&AffineGeometryLinearization>,
        quadrature: &QuadratureRule,
    ) -> Result<super::integration::IntegralActions<S>, Diagnostic> {
        self.validate_geometry(geometry, quadrature)?;
        let fields = self
            .fields
            .iter()
            .map(IntegralField::from)
            .collect::<Vec<_>>();
        let indices = self
            .fields
            .iter()
            .enumerate()
            .map(|(index, layout)| (layout.field, index))
            .collect::<BTreeMap<_, _>>();
        let mut integrals = Vec::new();
        let mut data = Vec::new();
        for (row_index, row) in self.form.rows.iter().enumerate() {
            for term in &row.terms {
                let eliminated = self.eliminations.get(&term.trial);
                let trial_field = eliminated.copied().unwrap_or(term.trial);
                let column = indices[&trial_field];
                let factor = match (eliminated, term.derivative) {
                    (Some(_), true) => 1.,
                    (Some(_), false) => self.time_step().expect("bound state step"),
                    (None, true) => self.time_step().expect("bound derivative step").recip(),
                    (None, false) => 1.,
                };
                integrals.push(super::integration::IntegralTerm {
                    test_coordinate: 0,
                    trial_coordinate: 0,
                    row: row_index,
                    column,
                    pairing: term.pairing,
                    trial_scale: factor * self.fields[column].scale,
                });
                data.push((row_index, term));
            }
        }
        super::integration::integrate_with_geometry_direction(
            self.reference,
            &fields,
            &integrals,
            geometry,
            direction,
            quadrature,
            |_,
             point,
             point_tangent,
             coefficients,
             forcing,
             flux_load,
             coefficient_tangents,
             forcing_tangents,
             flux_tangents| {
                let point_direction = direction.map(|_| point_tangent);
                let mut component_tangents = forcing_tangents.iter_mut();
                let mut components = forcing.iter_mut();
                let mut flux_offset = 0;
                for (index, row) in self.form.rows.iter().enumerate() {
                    for flux in &row.flux {
                        let Some((datum, entries)) =
                            flux.prescribed_components(self.form.dimension, row.forcing.len())?
                        else {
                            continue;
                        };
                        let (value, tangent) = evaluate_data(datum, point, point_direction)?;
                        let scale = <S as From<f64>>::from(self.row_multipliers[index]);
                        for entry in entries {
                            flux_load[flux_offset + entry] += scale * value;
                            if let Some(output) = flux_tangents.get_mut(flux_offset + entry) {
                                *output += scale * tangent;
                            }
                        }
                    }
                    flux_offset += row.forcing.len() * self.form.dimension;
                    for component in &row.forcing {
                        let (value, tangent) = evaluate_data(component, point, point_direction)?;
                        let scale = <S as From<f64>>::from(self.row_multipliers[index]);
                        *components.next().expect("exact row component inventory") = scale * value;
                        if let Some(output) = component_tangents.next() {
                            *output = scale * tangent;
                        }
                    }
                }
                for (index, (coefficient, (row, term))) in
                    coefficients.iter_mut().zip(&data).enumerate()
                {
                    let (value, tangent) =
                        evaluate_data(&term.coefficient, point, point_direction)?;
                    let value = term.require_coefficient(value)?;
                    let scale = <S as From<f64>>::from(self.row_multipliers[*row]);
                    *coefficient = scale * value;
                    if let Some(output) = coefficient_tangents.get_mut(index) {
                        *output = scale * tangent;
                    }
                }
                Ok(())
            },
        )
    }
}

/// Coordinate pullback with fixed physical Parameters. Coefficient derivative
/// admission stays with Data, including its exact product and chart rules.
pub(super) fn evaluate_data<S: Coefficient>(
    data: &super::Data<S>,
    point: &[f64],
    direction: Option<&[f64]>,
) -> Result<(S, S), Diagnostic> {
    let value = data.evaluate(point)?;
    let mut tangent = <S as From<f64>>::from(0.0);
    if let Some(direction) = direction {
        if direction.len() != point.len() || direction.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "coefficient direction differs from its physical point",
            ));
        }
        if data.spatial() {
            for (axis, entry) in direction.iter().enumerate() {
                if *entry != 0.0 {
                    tangent += <S as From<f64>>::from(*entry)
                        * data
                            .coordinate_derivative(axis, point.len())?
                            .evaluate(point)?;
                }
            }
        }
    }
    if !tangent.is_finite() {
        return Err(invalid("coefficient geometry tangent is non-finite"));
    }
    Ok((value, tangent))
}
