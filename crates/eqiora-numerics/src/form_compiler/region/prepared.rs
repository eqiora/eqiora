//! Immutable local actions for one exact form, geometry and quadrature binding.

use super::coordinates::{Native, Projection};
use super::evaluate::AffineInput;
use num_complex::ComplexFloat;
use std::collections::BTreeMap;

use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryLinearization, AffineGeometryMap, GeometryMap, QuadratureRule};

use super::{BoundRegionForm, RegionFieldLayout, RegionLinearization, invalid};
use crate::affine_fem::physical_gradient;
use crate::discrete_space::DiscreteSpace;
use crate::spatial_expression::Coefficient;

mod action;
mod ale;
mod components;
mod dyadics;
use components::Component;

pub(crate) type CellActions<S, P = Native> =
    (PreparedRegionCell<S, P>, Option<PreparedRegionCell<S, P>>);

#[derive(Debug, Clone, PartialEq)]
struct Sample<S: Coefficient> {
    physical: Vec<f64>,
    values: Vec<Vec<f64>>,
    gradients: Vec<Vec<[f64; 3]>>,
    coefficients: Vec<S>,
}

#[derive(Debug, Clone, PartialEq)]
struct Dyadic {
    row: usize,
    column: usize,
    split: bool,
}

/// Local numbering contains no mesh indices, global maps or mutable coefficients.
/// Adaptation creates new geometry bindings and resolves its maps separately.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedRegionCell<S: Coefficient, P: Projection<S> = Native> {
    dimension: usize,
    fields: Vec<RegionFieldLayout>,
    coordinates: Vec<std::ops::Range<usize>>,
    affine: Box<LocalContribution<P::Scalar>>,
    rate: Option<Box<LocalContribution<P::Scalar>>>,
    step: Option<f64>,
    independent_coordinates: bool,
    history: Vec<(RawId, usize, Vec<P::Scalar>)>,
    dyadics: Vec<Dyadic>,
    components: Vec<Component<S>>,
    samples: Vec<Sample<S>>,
}

impl<S: Coefficient> BoundRegionForm<S> {
    pub(crate) fn prepare_cell(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<PreparedRegionCell<S>, Diagnostic> {
        self.prepare_cell_with_history_geometry(geometry, geometry, quadrature)
    }

    /// Integrate current operators and previous scalar inventory on their own
    /// geometries. The caller owns the consecutive topology/motion binding and
    /// the relative flux; this does not turn a physical flux into an ALE flux.
    pub(crate) fn prepare_cell_with_history_geometry(
        &self,
        geometry: &AffineGeometryMap,
        previous_geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<PreparedRegionCell<S>, Diagnostic> {
        self.prepare_cell_actions(geometry, previous_geometry, None, quadrature)
            .map(|(primal, _)| primal)
    }

    /// Prepare an action and its geometry derivative at fixed Field coefficients.
    /// Current and history maps have independent directions; in differential
    /// ALE both mass terms use the current map and the same direction.
    pub(crate) fn prepare_cell_actions(
        &self,
        geometry: &AffineGeometryMap,
        previous_geometry: &AffineGeometryMap,
        directions: Option<(&AffineGeometryLinearization, &AffineGeometryLinearization)>,
        quadrature: &QuadratureRule,
    ) -> Result<CellActions<S>, Diagnostic> {
        self.prepare_projected_cell_actions::<Native>(
            geometry,
            previous_geometry,
            directions,
            quadrature,
        )
    }

    pub(crate) fn prepare_projected_cell_actions<P: Projection<S>>(
        &self,
        geometry: &AffineGeometryMap,
        previous_geometry: &AffineGeometryMap,
        directions: Option<(&AffineGeometryLinearization, &AffineGeometryLinearization)>,
        quadrature: &QuadratureRule,
    ) -> Result<CellActions<S, P>, Diagnostic> {
        if directions.is_some_and(|(current, previous)| {
            current.map() != geometry || previous.map() != previous_geometry
        }) {
            return Err(invalid(
                "prepared geometry directions differ from their primal maps",
            ));
        }
        self.validate_geometry(previous_geometry, quadrature)?;
        if directions.is_some() && self.form.rows.iter().any(|row| !row.components.is_empty()) {
            return Err(invalid(
                "nonlinear component geometry actions require their mixed scalar IR products",
            ));
        }
        if geometry != previous_geometry
            && (self.previous.is_empty()
                || !self.eliminations.is_empty()
                || self.fields.iter().any(|field| field.components != 1)
                || self.form.rows.iter().any(|row| {
                    row.is_nonlinear()
                        || row.terms.iter().any(|term| {
                            term.derivative
                                && term.pairing != crate::form_compiler::bilinear::Pairing::Value
                        })
                }))
        {
            return Err(invalid(
                "distinct history geometry requires scalar inventory without eliminated states",
            ));
        }
        let coordinates = P::ranges(&self.fields)?;
        let (affine, affine_tangent) = self.prepare_affine_actions::<P>(
            geometry,
            directions.map(|(current, _)| current),
            quadrature,
            AffineInput::Current,
        )?;
        let (rate, rate_tangent) = if self.form.rows.iter().any(|row| {
            row.terms
                .iter()
                .any(|term| term.derivative && !self.eliminations.contains_key(&term.trial))
        }) {
            let (value, tangent) = self.prepare_affine_actions::<P>(
                geometry,
                directions.map(|(current, _)| current),
                quadrature,
                AffineInput::Rate,
            )?;
            (Some(Box::new(value)), tangent.map(Box::new))
        } else {
            (None, None)
        };
        let mut history = Vec::new();
        let mut history_tangent = Vec::new();
        for (field, layout) in &self.previous {
            let trial = self.eliminations.get(field).copied().unwrap_or(*field);
            let column = self
                .fields
                .iter()
                .position(|layout| layout.field == trial)
                .ok_or_else(|| invalid("history lacks an algebraic binding"))?;
            let (matrix, matrix_tangent) = self.prepare_affine_actions::<P>(
                previous_geometry,
                directions.map(|(_, previous)| previous),
                quadrature,
                AffineInput::Previous(*field),
            )?;
            let range = &coordinates[column];
            if self.fields[column].range.len() != layout.range.len() {
                return Err(invalid(
                    "prepared history basis differs from its algebraic binding",
                ));
            }
            let block = |matrix: &LocalContribution<P::Scalar>| {
                matrix
                    .matrix()
                    .chunks_exact(affine.rows())
                    .flat_map(|row| row[range.clone()].iter().copied())
                    .collect()
            };
            history.push((*field, range.len(), block(&matrix)));
            if let Some(matrix) = matrix_tangent {
                history_tangent.push((*field, range.len(), block(&matrix)));
            }
        }
        let mut dyadics = Vec::new();
        let mut data = Vec::new();
        for (row, source) in self.form.rows.iter().enumerate() {
            for term in &source.dyadics {
                let column = self
                    .fields
                    .iter()
                    .position(|field| field.field == term.field)
                    .ok_or_else(|| invalid("dyadic trial lacks an algebraic binding"))?;
                dyadics.push(Dyadic {
                    row,
                    column,
                    split: term.split,
                });
                data.push((row, &term.coefficient));
            }
        }
        let mut samples = Vec::new();
        let mut sample_tangents = Vec::new();
        let mut components = Vec::new();
        for (row, source) in self.form.rows.iter().enumerate() {
            for term in &source.components {
                components.push(Component::new(row, term, &self.fields, data.len())?);
                data.push((row, &term.coefficient));
            }
        }
        if !dyadics.is_empty() || !components.is_empty() {
            let spaces = self
                .fields
                .iter()
                .map(|field| DiscreteSpace::new(field.space, self.reference))
                .collect::<Result<Vec<_>, _>>()?;
            let inverse = geometry.inverse_jacobian()?;
            let inverse_tangent = directions
                .map(|(current, _)| current.inverse_jacobian_tangent())
                .transpose()?;
            for point in quadrature.points() {
                let mut physical = vec![0.; self.form.dimension];
                let mut physical_tangent = vec![0.; self.form.dimension];
                if let Some((current, _)) = directions {
                    current.map_point_jvp(
                        &point.coordinates,
                        &mut physical,
                        &mut physical_tangent,
                    )?;
                } else {
                    geometry.map_point(&point.coordinates, &mut physical)?;
                }
                let tables = spaces
                    .iter()
                    .map(|space| space.tabulate(&point.coordinates))
                    .collect::<Result<Vec<_>, _>>()?;
                let values: Vec<_> = tables.iter().map(|table| table.values().to_vec()).collect();
                let mapped_gradients = |inverse: &[f64]| {
                    tables
                        .iter()
                        .map(|table| {
                            (0..table.values().len())
                                .map(|node| {
                                    let gradient = physical_gradient(
                                        table.gradient(node).expect("supported gradient"),
                                        inverse,
                                        self.form.dimension,
                                    );
                                    let mut result = [0.; 3];
                                    result[..self.form.dimension].copy_from_slice(&gradient);
                                    result
                                })
                                .collect()
                        })
                        .collect::<Vec<Vec<_>>>()
                };
                let gradients = mapped_gradients(&inverse);
                let coefficient_values = data
                    .iter()
                    .map(|(row, coefficient)| {
                        let (value, tangent) = super::evaluate::evaluate_data(
                            coefficient,
                            &physical,
                            directions.map(|_| physical_tangent.as_slice()),
                        )?;
                        let tangent =
                            directions.map_or(<S as From<f64>>::from(0.), |(current, _)| {
                                <S as From<f64>>::from(point.weight * self.row_multipliers[*row])
                                    * (<S as From<f64>>::from(current.measure_scale_tangent())
                                        * value
                                        + <S as From<f64>>::from(geometry.measure_scale())
                                            * tangent)
                            });
                        Ok((
                            <S as From<f64>>::from(
                                point.weight
                                    * geometry.measure_scale()
                                    * self.row_multipliers[*row],
                            ) * value,
                            tangent,
                        ))
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                if coefficient_values
                    .iter()
                    .any(|(value, tangent)| !value.is_finite() || !tangent.is_finite())
                {
                    return Err(invalid("prepared nonlinear coefficient is non-finite"));
                }
                let coefficients = coefficient_values
                    .iter()
                    .map(|(value, _)| *value)
                    .collect::<Vec<_>>();
                if let Some(inverse_tangent) = &inverse_tangent {
                    // Every dyadic contraction has one gradient. Product-rule
                    // samples reuse that contraction, including its state Jacobian.
                    sample_tangents.push(Sample {
                        physical: physical.clone(),
                        values: values.clone(),
                        gradients: gradients.clone(),
                        coefficients: coefficient_values
                            .iter()
                            .map(|(_, tangent)| *tangent)
                            .collect(),
                    });
                    sample_tangents.push(Sample {
                        physical: physical.clone(),
                        values: values.clone(),
                        gradients: mapped_gradients(inverse_tangent),
                        coefficients: coefficients.clone(),
                    });
                }
                samples.push(Sample {
                    physical,
                    values,
                    gradients,
                    coefficients,
                });
            }
        }
        let tangent = affine_tangent.map(|affine| PreparedRegionCell {
            dimension: self.form.dimension,
            fields: self.fields.clone(),
            affine: Box::new(affine),
            rate: rate_tangent,
            step: self.time_step(),
            // A geometry tangent is not the original physical residual.
            independent_coordinates: false,
            coordinates: coordinates.clone(),
            history: history_tangent,
            dyadics: dyadics.clone(),
            components: components.clone(),
            samples: sample_tangents,
        });
        Ok((
            PreparedRegionCell {
                dimension: self.form.dimension,
                fields: self.fields.clone(),
                affine: Box::new(affine),
                rate,
                step: self.time_step(),
                independent_coordinates: self.eliminations.is_empty()
                    && geometry == previous_geometry,
                coordinates,
                history,
                dyadics,
                components,
                samples,
            },
            tangent,
        ))
    }
}

impl<S: Coefficient, P: Projection<S>> PreparedRegionCell<S, P> {
    pub(crate) fn add_load(&mut self, load: &[P::Scalar]) -> Result<(), Diagnostic> {
        if load.len() != self.affine.rows() || load.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "prepared load requires exact finite local equation coverage",
            ));
        }
        let rhs = self
            .affine
            .rhs()
            .iter()
            .zip(load)
            .map(|(a, b)| *a + *b)
            .collect();
        *self.affine = LocalContribution::new(
            self.affine.rows(),
            self.affine.columns(),
            self.affine.matrix().to_vec(),
            rhs,
        )?;
        Ok(())
    }
    fn validate_previous(
        &self,
        previous: &BTreeMap<RawId, Vec<P::Scalar>>,
    ) -> Result<(), Diagnostic> {
        if previous.len() != self.history.len()
            || self.history.iter().any(|(id, count, _)| {
                previous.get(id).is_none_or(|values| {
                    values.len() != *count || values.iter().any(|value| !value.is_finite())
                })
            })
        {
            return Err(invalid(
                "previous coefficients require exact consumed Field coverage, shape and finite values",
            ));
        }
        Ok(())
    }

    fn rhs(
        &self,
        previous: &BTreeMap<RawId, Vec<P::Scalar>>,
    ) -> Result<Vec<P::Scalar>, Diagnostic> {
        self.validate_previous(previous)?;
        let mut rhs = self.affine.rhs().to_vec();
        for (id, count, matrix) in &self.history {
            for (value, row) in rhs.iter_mut().zip(matrix.chunks_exact(*count)) {
                *value += row
                    .iter()
                    .zip(&previous[id])
                    .fold(<P::Scalar as From<f64>>::from(0.0), |sum, (a, b)| {
                        sum + *a * *b
                    });
            }
        }
        Ok(rhs)
    }

    pub(crate) fn evaluate(
        &self,
        previous: &BTreeMap<RawId, Vec<P::Scalar>>,
    ) -> Result<LocalContribution<P::Scalar>, Diagnostic> {
        if self.rate.is_some() && self.step.is_none() {
            return Err(invalid(
                "rate-dependent region evaluation requires an explicit State/rate point or time discretization",
            ));
        }
        if !self.dyadics.is_empty() || !self.components.is_empty() {
            return Err(invalid(
                "nonlinear region evaluation requires an explicit candidate point",
            ));
        }
        LocalContribution::new(
            self.affine.rows(),
            self.affine.columns(),
            self.affine_matrix(self.step_pullback()),
            self.rhs(previous)?,
        )
    }
}
