//! Prepared parent-cell trace actions, shared by affine and Newton assembly.
use super::{BoundRegionForm, RegionFieldLayout, RegionLinearization, basis, invalid};
use crate::spatial_expression::Coefficient;
use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{
    AffineGeometryMap, EntityIncidence, GeometryMap, QuadratureRule, ReferenceCellFamily,
};

#[derive(Debug, Clone, PartialEq)]
struct Sample<S: Coefficient> {
    values: Vec<Vec<f64>>,
    coefficients: Vec<S>,
}

/// Physical geometry, trace orientation and complete Field ranges remain bound
/// while candidate coefficients change. Global constraints belong to assembly.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedRegionFacet<S: Coefficient> {
    domain: RawId,
    parent: AffineGeometryMap,
    fields: Vec<RegionFieldLayout>,
    row: usize,
    columns: Vec<usize>,
    normal: Vec<f64>,
    affine: LocalContribution<S>,
    samples: Vec<Sample<S>>,
}

impl<S: Coefficient> BoundRegionForm<S> {
    pub(crate) fn has_nonlinear_flux(&self, field: RawId) -> bool {
        self.form
            .rows
            .iter()
            .any(|row| row.tested == field && !row.dyadics.is_empty())
    }

    /// Retain the established affine rule and cover every cubic trace product.
    pub(crate) fn natural_facet_quadrature(
        &self,
        field: RawId,
    ) -> Result<QuadratureRule, Diagnostic> {
        let degree = self.natural_trace_degree(field)?;
        let dimension = self.form.dimension - 1;
        if dimension == 0 {
            return Ok(QuadratureRule::point());
        }
        if self.reference.family() == ReferenceCellFamily::Hypercube {
            QuadratureRule::tensor_product_gauss_legendre(
                dimension,
                2.max((degree + 1).div_ceil(2)),
            )
        } else {
            eqiora_meshing::simplex_duffy_gauss_legendre(
                dimension,
                3.max((degree + dimension).div_ceil(2)),
            )
        }
    }

    fn natural_trace_degree(&self, field: RawId) -> Result<usize, Diagnostic> {
        let row = self
            .form
            .rows
            .iter()
            .find(|row| row.tested == field)
            .ok_or_else(|| invalid("natural flux has a foreign tested Field"))?;
        Ok(if row.dyadics.is_empty() {
            0
        } else if self.reference.family() == ReferenceCellFamily::Hypercube {
            3 * (self.form.dimension - 1)
        } else {
            3
        })
    }

    pub(crate) fn prepare_natural_facet(
        &self,
        field: RawId,
        cell: &AffineGeometryMap,
        facet: (&AffineGeometryMap, EntityIncidence, &[usize]),
        rule: &QuadratureRule,
        datum: impl Fn(&[f64], &[f64]) -> Result<Vec<S>, Diagnostic>,
    ) -> Result<PreparedRegionFacet<S>, Diagnostic> {
        let affine = self.evaluate_natural_facet(field, cell, facet, rule, datum)?;
        let (facet, incidence, parent_vertices) = facet;
        let dimension = self.form.dimension;
        let row = self
            .fields
            .iter()
            .position(|layout| layout.field == field)
            .expect("validated natural Field");
        let normal = crate::affine_fem::parent_outward_normal(cell, incidence)?;
        let terms = &self.form.rows[row].dyadics;
        let columns = terms
            .iter()
            .map(|term| {
                self.fields
                    .iter()
                    .position(|layout| layout.field == term.field)
                    .ok_or_else(|| invalid("natural dyadic trial lacks a binding"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut prepared = PreparedRegionFacet {
            domain: self.domain(),
            parent: cell.clone(),
            fields: self.fields.clone(),
            row,
            columns,
            normal,
            affine,
            samples: Vec::new(),
        };
        if terms.is_empty() {
            return Ok(prepared);
        }
        if dimension > 1
            && rule.polynomial_exactness().is_none_or(|order| {
                order < self.natural_trace_degree(field).expect("validated Field")
            })
        {
            return Err(invalid(
                "nonlinear natural facet requires exact cubic trace products",
            ));
        }
        let facet_space = if dimension == 1 {
            crate::discrete_space::DiscreteSpace::new(
                eqiora_realization::Space::cell_constant(),
                facet.reference_cell(),
            )?
        } else {
            basis(
                eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                facet.reference_cell(),
            )?
        };
        let spaces = self
            .fields
            .iter()
            .map(|layout| basis(layout.space, self.reference))
            .collect::<Result<Vec<_>, _>>()?;
        for sample in rule.points() {
            let facet_table = facet_space.tabulate(&sample.coordinates)?;
            let reference = crate::affine_fem::facet_reference_point(
                self.reference,
                parent_vertices,
                facet_table.values(),
            )?;
            let values = spaces
                .iter()
                .map(|space| {
                    let table = space.tabulate(&reference)?;
                    Ok(table
                        .values()
                        .iter()
                        .zip(space.local_dofs())
                        .map(|(value, dof)| {
                            if dof.entity_dimension() == dimension {
                                0.
                            } else {
                                *value
                            }
                        })
                        .collect())
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let mut physical = vec![0.; dimension];
            facet.map_point(&sample.coordinates, &mut physical)?;
            let coefficients = terms
                .iter()
                .map(|term| {
                    Ok(term.coefficient.evaluate(&physical)?
                        * <S as From<f64>>::from(
                            -self.row_multipliers[row]
                                * sample.weight
                                * facet.measure_scale()
                                * if term.split { 0.5 } else { 1. },
                        ))
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            prepared.samples.push(Sample {
                values,
                coefficients,
            });
        }
        Ok(prepared)
    }
}

impl<S: Coefficient> PreparedRegionFacet<S> {
    pub(crate) fn affine(&self) -> &LocalContribution<S> {
        &self.affine
    }

    pub(crate) fn validate_parent(
        &self,
        form: &BoundRegionForm<S>,
        geometry: &AffineGeometryMap,
    ) -> Result<(), Diagnostic> {
        if self.domain != form.domain() || self.fields != form.fields() || &self.parent != geometry
        {
            return Err(invalid(
                "prepared natural facet differs from its exact parent geometry or Field binding",
            ));
        }
        Ok(())
    }
}

impl PreparedRegionFacet<f64> {
    pub(crate) fn linearize(&self, point: &[f64]) -> Result<RegionLinearization, Diagnostic> {
        self.action(point, true)
    }

    pub(crate) fn residual(&self, point: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        Ok(self.action(point, false)?.residual)
    }

    pub(super) fn action(
        &self,
        point: &[f64],
        derivative: bool,
    ) -> Result<RegionLinearization, Diagnostic> {
        let count = self.affine.rows();
        if point.len() != count || point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "nonlinear natural facet requires an exact finite parent point",
            ));
        }
        let mut action = RegionLinearization {
            jacobian: if derivative {
                vec![0.; count * count]
            } else {
                Vec::new()
            },
            residual: self.affine.rhs().iter().map(|value| -*value).collect(),
        };
        let dimension = self.normal.len();
        let test = &self.fields[self.row];
        for sample in &self.samples {
            for (&column, &coefficient) in self.columns.iter().zip(&sample.coefficients) {
                let trial = &self.fields[column];
                let value = (0..dimension)
                    .map(|i| {
                        sample.values[column]
                            .iter()
                            .enumerate()
                            .map(|(b, shape)| {
                                shape * trial.scale * point[trial.range.start + b * dimension + i]
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let normal_value = value
                    .iter()
                    .zip(&self.normal)
                    .map(|(u, n)| u * n)
                    .sum::<f64>();
                for (a, test_value) in sample.values[self.row].iter().enumerate() {
                    for (i, component_value) in value.iter().enumerate() {
                        let r = test.range.start + a * dimension + i;
                        action.residual[r] +=
                            coefficient * test_value * normal_value * component_value;
                        if !derivative {
                            continue;
                        }
                        for (b, trial_value) in sample.values[column].iter().enumerate() {
                            for (k, normal_component) in self.normal.iter().enumerate() {
                                let c = trial.range.start + b * dimension + k;
                                action.jacobian[r * count + c] += coefficient
                                    * test_value
                                    * trial_value
                                    * trial.scale
                                    * (normal_component * component_value
                                        + if i == k { normal_value } else { 0. });
                            }
                        }
                    }
                }
            }
        }
        if action
            .residual
            .iter()
            .chain(&action.jacobian)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("nonlinear natural facet action is non-finite"));
        }
        Ok(action)
    }
}
