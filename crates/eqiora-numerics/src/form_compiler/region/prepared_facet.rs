//! Prepared parent-cell trace actions, shared by affine and Newton assembly.
use super::coordinates::{Native, Projection};
use super::field_point::{BoundFieldComponent, FieldPoint, FieldValues};
use super::flux::FluxTerm;
use super::{BoundRegionForm, RegionFieldLayout, RegionLinearization, invalid};
use crate::discrete_space::DiscreteSpace;
use crate::spatial_expression::Coefficient;
use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{
    AffineGeometryMap, EntityIncidence, GeometryMap, QuadratureRule, ReferenceCellFamily,
};

mod action;

#[derive(Debug, Clone, PartialEq)]
struct Sample<S: Coefficient> {
    physical: Vec<f64>,
    values: Vec<Vec<f64>>,
    gradients: Vec<Vec<[f64; 3]>>,
    coefficients: Vec<S>,
}

/// Physical geometry, trace orientation and complete Field ranges remain bound
/// while candidate coefficients change. Global constraints belong to assembly.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedRegionFacet<S: Coefficient, P: Projection<S> = Native> {
    domain: RawId,
    parent: AffineGeometryMap,
    assembly_points: Option<usize>,
    fields: Vec<RegionFieldLayout>,
    row: usize,
    columns: Vec<usize>,
    constitutive: Vec<BoundFieldComponent<S>>,
    transport_nodes: Vec<Vec<bool>>,
    normal: Vec<f64>,
    coordinates: Vec<std::ops::Range<usize>>,
    affine: LocalContribution<P::Scalar>,
    samples: Vec<Sample<S>>,
}

impl<S: Coefficient> BoundRegionForm<S> {
    pub(crate) fn requires_facet_evaluation(&self, field: RawId) -> bool {
        self.form
            .rows
            .iter()
            .any(|row| row.tested == field && row.has_boundary_flux())
    }

    /// Retain the established affine rule and cover every cubic trace product.
    pub(crate) fn natural_facet_quadrature(
        &self,
        field: RawId,
        datum_degree: Option<usize>,
    ) -> Result<QuadratureRule, Diagnostic> {
        let dimension = self.form.dimension - 1;
        if dimension == 0 {
            return Ok(QuadratureRule::point());
        }
        let minimum = self.minimum_natural_gauss_points(field, datum_degree)?;
        let points = self.assembly_points.unwrap_or(minimum);
        if points < minimum {
            return Err(invalid(
                "assembly quadrature underintegrates the prescribed polynomial facet datum",
            ));
        }
        if self.reference.family() == ReferenceCellFamily::Hypercube {
            QuadratureRule::tensor_product_gauss_legendre(dimension, points)
        } else {
            eqiora_meshing::simplex_duffy_gauss_legendre(dimension, points)
        }
    }

    pub(crate) fn minimum_natural_gauss_points(
        &self,
        field: RawId,
        datum_degree: Option<usize>,
    ) -> Result<usize, Diagnostic> {
        let degree = self.natural_trace_degree(field, datum_degree)?;
        let dimension = self.form.dimension - 1;
        Ok(
            if self.reference.family() == ReferenceCellFamily::Hypercube {
                2.max((degree + 1).div_ceil(2))
            } else {
                3.max((degree + dimension).div_ceil(2))
            },
        )
    }

    pub(super) fn natural_trace_degree(
        &self,
        field: RawId,
        datum_degree: Option<usize>,
    ) -> Result<usize, Diagnostic> {
        let row = self
            .form
            .rows
            .iter()
            .find(|row| row.tested == field)
            .ok_or_else(|| invalid("natural flux has a foreign tested Field"))?;
        let trace_basis = |field| {
            let layout = self
                .fields
                .iter()
                .find(|layout| layout.field == field)
                .expect("bound trace Field");
            if layout.space.family() == eqiora_realization::SpaceFamily::SimplexP1Bubble {
                1
            } else {
                self.field_basis_degree(field)
            }
        };
        let basis = trace_basis(field);
        let data = datum_degree.unwrap_or(0).saturating_add(basis);
        let nonlinear = row
            .dyadics
            .iter()
            .map(|term| basis + 2 * trace_basis(term.field))
            .max()
            .unwrap_or(0);
        // Retain the established conservative nonlinear first-order floor.
        let floor = if row.dyadics.is_empty() {
            0
        } else if self.reference.family() == ReferenceCellFamily::Hypercube {
            3 * (self.form.dimension - 1)
        } else {
            3
        };
        Ok(data.max(nonlinear).max(floor))
    }

    pub(crate) fn prepare_natural_facet(
        &self,
        field: RawId,
        cell: &AffineGeometryMap,
        facet: (&AffineGeometryMap, EntityIncidence, &[usize]),
        rule: &QuadratureRule,
        datum: impl Fn(&[f64], &[f64]) -> Result<Vec<S>, Diagnostic>,
        datum_degree: Option<usize>,
    ) -> Result<PreparedRegionFacet<S>, Diagnostic> {
        self.prepare_affine_facet(field, cell, facet, rule, datum, (datum_degree, None))
    }

    pub(crate) fn prepare_affine_facet(
        &self,
        field: RawId,
        cell: &AffineGeometryMap,
        facet: (&AffineGeometryMap, EntityIncidence, &[usize]),
        rule: &QuadratureRule,
        datum: impl Fn(&[f64], &[f64]) -> Result<Vec<S>, Diagnostic>,
        (datum_degree, robin): (Option<usize>, Option<&super::Data<S>>),
    ) -> Result<PreparedRegionFacet<S>, Diagnostic> {
        let affine =
            self.evaluate_affine_facet(field, cell, facet, rule, datum, (datum_degree, robin))?;
        let (facet, incidence, parent_vertices) = facet;
        let dimension = self.form.dimension;
        let row = self
            .fields
            .iter()
            .position(|layout| layout.field == field)
            .expect("validated natural Field");
        let normal = crate::affine_fem::parent_outward_normal(cell, incidence)?;
        let source = &self.form.rows[row];
        let terms = &source.dyadics;
        let columns = terms
            .iter()
            .map(|term| {
                self.fields
                    .iter()
                    .position(|layout| layout.field == term.field)
                    .ok_or_else(|| invalid("natural dyadic trial lacks a binding"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let constitutive = source
            .flux
            .iter()
            .filter_map(|term| match term {
                FluxTerm::Component(term) => {
                    Some(BoundFieldComponent::new(&term.value, &self.fields, false))
                }
                _ => None,
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut prepared = PreparedRegionFacet {
            domain: self.domain(),
            parent: cell.clone(),
            assembly_points: self.assembly_points,
            fields: self.fields.clone(),
            row,
            columns,
            constitutive,
            transport_nodes: Vec::new(),
            normal,
            coordinates: <Native as Projection<S>>::ranges(&self.fields)?,
            affine,
            samples: Vec::new(),
        };
        // Replacing traction by a datum does not erase the original
        // coefficients' domain obligations, including in affine execution.
        // Validate each physical scalar coordinate image just as volume
        // assembly does; a real-linear action need not be complex-linear.
        let coefficients = source
            .flux
            .iter()
            .map(|flux| flux.coefficient().coordinate_images(dimension))
            .collect::<Result<Vec<_>, _>>()?;
        for sample in rule.points() {
            let mut physical = vec![0.; dimension];
            facet.map_point(&sample.coordinates, &mut physical)?;
            for (coefficient, _) in coefficients.iter().flatten() {
                let _ = coefficient.evaluate(&physical)?;
            }
        }
        if terms.is_empty() && prepared.constitutive.is_empty() {
            return Ok(prepared);
        }
        if dimension > 1
            && rule.polynomial_exactness().is_none_or(|order| {
                order
                    < self
                        .natural_trace_degree(field, datum_degree)
                        .expect("validated Field")
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
            DiscreteSpace::new(
                eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                facet.reference_cell(),
            )?
        };
        let spaces = self
            .fields
            .iter()
            .map(|layout| DiscreteSpace::new(layout.space, self.reference))
            .collect::<Result<Vec<_>, _>>()?;
        prepared.transport_nodes = spaces
            .iter()
            .map(|space| {
                let trace = space.trace_dofs(incidence.local_ordinal)?;
                Ok((0..space.local_dofs().len())
                    .map(|node| trace.contains(&node))
                    .collect())
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let inverse = cell.inverse_jacobian()?;
        for sample in rule.points() {
            let facet_table = facet_space.tabulate(&sample.coordinates)?;
            let reference = crate::affine_fem::facet_reference_point(
                self.reference,
                parent_vertices,
                facet_table.values(),
            )?;
            let tables = spaces
                .iter()
                .map(|space| space.tabulate(&reference))
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let values = tables.iter().map(|table| table.values().to_vec()).collect();
            let gradients = tables
                .iter()
                .map(|table| {
                    (0..table.values().len())
                        .map(|node| {
                            let gradient = crate::affine_fem::physical_gradient(
                                table.gradient(node).expect("supported gradient"),
                                &inverse,
                                dimension,
                            );
                            let mut result = [0.; 3];
                            result[..dimension].copy_from_slice(&gradient);
                            result
                        })
                        .collect()
                })
                .collect();
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
                physical,
                values,
                gradients,
                coefficients,
            });
        }
        Ok(prepared)
    }
}

impl<S: Coefficient, P: Projection<S>> PreparedRegionFacet<S, P> {
    fn transport_shape(&self, sample: &Sample<S>, field: usize, node: usize) -> f64 {
        if self.transport_nodes[field][node] {
            sample.values[field][node]
        } else {
            0.
        }
    }

    pub(crate) fn affine(&self) -> &LocalContribution<P::Scalar> {
        &self.affine
    }

    pub(crate) fn validate_parent(
        &self,
        form: &BoundRegionForm<S>,
        geometry: &AffineGeometryMap,
    ) -> Result<(), Diagnostic> {
        if self.domain != form.domain()
            || self.fields != form.fields()
            || &self.parent != geometry
            || self.assembly_points != form.assembly_points
        {
            return Err(invalid(
                "prepared natural facet differs from its exact parent geometry, Field binding or assembly quadrature",
            ));
        }
        Ok(())
    }
}

impl<S: Coefficient> PreparedRegionFacet<S> {
    /// Reuse the physical trace samples and project only their numerical storage.
    pub(crate) fn project<P: Projection<S>>(self) -> Result<PreparedRegionFacet<S, P>, Diagnostic> {
        let mut rhs = Vec::new();
        for field in &self.fields {
            rhs.extend(super::coordinates::values::<S, P>(
                &field.value_type,
                &self.affine.rhs()[field.range.clone()],
            )?);
        }
        let count = rhs.len();
        let source_count = self.affine.rows();
        let mut matrix = vec![<P::Scalar as From<f64>>::from(0.); count * count];
        let ranges = P::ranges(&self.fields)?;
        for (r, row) in self.fields.iter().enumerate() {
            let rw = P::width(&row.value_type)?;
            for (c, column) in self.fields.iter().enumerate() {
                let cw = P::width(&column.value_type)?;
                for a in 0..row.range.len() {
                    for b in 0..column.range.len() {
                        let value = self.affine.matrix()
                            [(row.range.start + a) * source_count + column.range.start + b];
                        for i in 0..rw {
                            for j in 0..cw {
                                matrix[(ranges[r].start + a * rw + i) * count
                                    + ranges[c].start
                                    + b * cw
                                    + j] = P::component(value * P::direction(j), i);
                            }
                        }
                    }
                }
            }
        }
        let affine = LocalContribution::new(count, count, matrix, rhs)?;
        Ok(PreparedRegionFacet {
            domain: self.domain,
            parent: self.parent,
            assembly_points: self.assembly_points,
            coordinates: P::ranges(&self.fields)?,
            fields: self.fields,
            row: self.row,
            columns: self.columns,
            constitutive: self.constitutive,
            transport_nodes: self.transport_nodes,
            normal: self.normal,
            affine,
            samples: self.samples,
        })
    }
}
