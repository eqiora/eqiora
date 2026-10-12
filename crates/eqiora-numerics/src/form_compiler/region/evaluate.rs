use crate::spatial_expression::Coefficient;
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{
    AffineGeometryLinearization, AffineGeometryMap, GeometryMap, QuadratureRule,
    ReferenceCellFamily,
};
use eqiora_realization::SpaceFamily;

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
                "quadrature does not cover region basis-product degree",
            ));
        }
        Ok(())
    }

    /// Degree of the basis products in the bound weak residual and Jacobian.
    fn required_polynomial_exactness(&self) -> usize {
        let mut exactness = if self
            .fields
            .iter()
            .any(|layout| layout.space.family() == SpaceFamily::SimplexP1Bubble)
        {
            2 * (self.form.dimension + 1)
        } else if self.reference.family() == ReferenceCellFamily::Hypercube {
            3
        } else {
            2
        };
        if self.form.rows.iter().any(|row| !row.dyadics.is_empty()) {
            let degree = if self
                .fields
                .iter()
                .any(|layout| layout.space.family() == SpaceFamily::SimplexP1Bubble)
            {
                3 * (self.form.dimension + 1) - 1
            } else if self.reference.family() == ReferenceCellFamily::Hypercube {
                3 * self.form.dimension - 1
            } else {
                2
            };
            exactness = exactness.max(degree);
        }
        exactness
    }

    /// Gauss points per axis covering the same degree checked during preparation.
    pub(crate) fn minimum_gauss_points(&self) -> usize {
        // Tensor Gauss has degree 2n-1; the simplex Duffy Jacobian changes
        // the total-degree guarantee to 2n-d.
        let loss = if self.reference.family() == ReferenceCellFamily::Simplex {
            self.form.dimension
        } else {
            1
        };
        (self.required_polynomial_exactness() + loss).div_ceil(2)
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
            .map(|layout| (layout.space, layout.components))
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
             isotropic_flux,
             coefficient_tangents,
             forcing_tangents,
             flux_tangents| {
                let point_direction = direction.map(|_| point_tangent);
                let mut component_tangents = forcing_tangents.iter_mut();
                let mut components = forcing.iter_mut();
                for (index, row) in self.form.rows.iter().enumerate() {
                    isotropic_flux[index] = <S as From<f64>>::from(0.0);
                    for flux in &row.flux {
                        if let super::flux::FluxTerm::Isotropic(value) = flux {
                            let (value, tangent) = evaluate_data(value, point, point_direction)?;
                            let scale = <S as From<f64>>::from(self.row_multipliers[index]);
                            isotropic_flux[index] += scale * value;
                            if let Some(output) = flux_tangents.get_mut(index) {
                                *output += scale * tangent;
                            }
                        }
                    }
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
                    if term.positive_diffusion && (value.im() != 0.0 || value.re() <= 0.0) {
                        return Err(invalid(
                            "linear Q1 requires positive finite diffusion and finite reaction/forcing",
                        ));
                    }
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
