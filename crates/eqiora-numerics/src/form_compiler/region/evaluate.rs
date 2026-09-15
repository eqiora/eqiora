use std::collections::BTreeMap;

use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule, ReferenceCellFamily};
use eqiora_realization::SpaceFamily;

use super::binding::BoundRegionForm;
use super::invalid;

impl BoundRegionForm {
    /// Validate local inputs without performing quadrature or constructing contributions.
    pub(crate) fn validate_cell(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        previous: &BTreeMap<RawId, Vec<f64>>,
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

    fn validate_geometry(
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
        if quadrature
            .polynomial_exactness()
            .is_none_or(|order| order < exactness)
        {
            return Err(invalid(
                "quadrature does not cover region basis-product degree",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare_affine(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
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
                    (Some(_), false) => self.step.expect("bound state step"),
                    (None, true) => self.step.expect("bound derivative step").recip(),
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
        super::integration::integrate(
            self.reference,
            &fields,
            &integrals,
            geometry,
            quadrature,
            |point, coefficients, forcing, isotropic_flux| {
                let mut components = forcing.iter_mut();
                for (index, row) in self.form.rows.iter().enumerate() {
                    isotropic_flux[index] = 0.0;
                    for flux in &row.flux {
                        if let super::flux::FluxTerm::Isotropic(value) = flux {
                            isotropic_flux[index] +=
                                self.row_multipliers[index] * value.evaluate(point)?;
                        }
                    }
                    for component in &row.forcing {
                        *components.next().expect("exact row component inventory") =
                            self.row_multipliers[index] * component.evaluate(point)?;
                    }
                }
                for (coefficient, (row, term)) in coefficients.iter_mut().zip(&data) {
                    let value = term.coefficient.evaluate(point)?;
                    if term.positive_diffusion && value <= 0.0 {
                        return Err(invalid(
                            "linear Q1 requires positive finite diffusion and finite reaction/forcing",
                        ));
                    }
                    *coefficient = self.row_multipliers[*row] * value;
                }
                Ok(())
            },
        )
    }
}
