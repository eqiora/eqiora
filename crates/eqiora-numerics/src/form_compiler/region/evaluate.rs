use crate::form_compiler::region::integration::IntegralField;
use crate::spatial_expression::Coefficient;
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryLinearization, AffineGeometryMap, GeometryMap, QuadratureRule};

use super::binding::BoundRegionForm;
use super::invalid;

use super::coordinates::{Native, Projection, Real};

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
        self.prepare_projected_actions::<Native>(geometry, direction, quadrature)
    }

    fn prepare_projected_actions<P: Projection<S>>(
        &self,
        geometry: &AffineGeometryMap,
        direction: Option<&AffineGeometryLinearization>,
        quadrature: &QuadratureRule,
    ) -> Result<super::integration::IntegralActions<P::Scalar>, Diagnostic> {
        self.validate_geometry(geometry, quadrature)?;
        let fields = self
            .fields
            .iter()
            .map(|layout| {
                let mut field = IntegralField::from(layout);
                field.scalar_coordinates = P::width(&layout.value_type)?;
                Ok(field)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
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
                let columns = P::columns(
                    &term.coefficient,
                    self.form.dimension,
                    fields[column].scalar_coordinates,
                )?;
                for (trial_coordinate, source) in columns.into_iter().enumerate() {
                    for test_coordinate in 0..fields[row_index].scalar_coordinates {
                        integrals.push(super::integration::IntegralTerm {
                            row: row_index,
                            column,
                            pairing: term.pairing,
                            trial_scale: factor * self.fields[column].scale,
                            test_coordinate,
                            trial_coordinate,
                        });
                        data.push((row_index, term, source.clone(), test_coordinate));
                    }
                }
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
                    let width = fields[index].scalar_coordinates;
                    for flux in &row.flux {
                        let Some((datum, entries)) =
                            flux.prescribed_components(self.form.dimension, row.forcing.len())?
                        else {
                            continue;
                        };
                        let (value, tangent) = evaluate_data(datum, point, point_direction)?;
                        let scale = <S as From<f64>>::from(self.row_multipliers[index]);
                        for entry in entries {
                            for coordinate in 0..width {
                                let slot = flux_offset + entry * width + coordinate;
                                flux_load[slot] += P::component(scale * value, coordinate);
                                if let Some(output) = flux_tangents.get_mut(slot) {
                                    *output += P::component(scale * tangent, coordinate);
                                }
                            }
                        }
                    }
                    flux_offset += row.forcing.len() * self.form.dimension * width;
                    for component in &row.forcing {
                        let (value, tangent) = evaluate_data(component, point, point_direction)?;
                        let scale = <S as From<f64>>::from(self.row_multipliers[index]);
                        for coordinate in 0..width {
                            *components.next().expect("exact row component inventory") =
                                P::component(scale * value, coordinate);
                            if let Some(output) = component_tangents.next() {
                                *output = P::component(scale * tangent, coordinate);
                            }
                        }
                    }
                }
                for (index, (coefficient, (row, term, source, coordinate))) in
                    coefficients.iter_mut().zip(&data).enumerate()
                {
                    let (value, tangent) = evaluate_data(source, point, point_direction)?;
                    P::validate(term, value, point)?;
                    let scale = <S as From<f64>>::from(self.row_multipliers[*row]);
                    *coefficient = P::component(scale * value, *coordinate);
                    if let Some(output) = coefficient_tangents.get_mut(index) {
                        *output = P::component(scale * tangent, *coordinate);
                    }
                }
                Ok(())
            },
        )
    }
}

impl BoundRegionForm<num_complex::Complex64> {
    /// Integrate the full real-linear action while retaining physical component
    /// pairings and the exact nodal or moment basis. Numbering is Field-major,
    /// basis-major, real/imaginary-coordinate-minor.
    pub(crate) fn prepare_real_affine_actions(
        &self,
        geometry: &AffineGeometryMap,
        direction: Option<&AffineGeometryLinearization>,
        quadrature: &QuadratureRule,
    ) -> Result<super::integration::IntegralActions<f64>, Diagnostic> {
        if self.time.is_some() || !self.previous.is_empty() || self.form.is_nonlinear() {
            return Err(invalid(
                "real-coordinate local preparation requires static affine actions",
            ));
        }
        self.prepare_projected_actions::<Real>(geometry, direction, quadrature)
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
