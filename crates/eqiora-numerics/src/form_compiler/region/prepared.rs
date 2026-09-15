//! Immutable local actions for one exact form, geometry and quadrature binding.

use std::collections::BTreeMap;

use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule};

use super::{BoundRegionForm, RegionFieldLayout, RegionLinearization, basis, invalid};
use crate::affine_fem::physical_gradient;

#[derive(Debug, Clone, PartialEq)]
struct Sample {
    values: Vec<Vec<f64>>,
    gradients: Vec<Vec<[f64; 3]>>,
    coefficients: Vec<f64>,
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
pub(crate) struct PreparedRegionCell {
    dimension: usize,
    fields: Vec<RegionFieldLayout>,
    affine: LocalContribution,
    history: Vec<(RawId, usize, Vec<f64>)>,
    dyadics: Vec<Dyadic>,
    samples: Vec<Sample>,
}

impl BoundRegionForm {
    pub(crate) fn prepare_cell(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<PreparedRegionCell, Diagnostic> {
        let zero = self
            .previous
            .iter()
            .map(|(id, layout)| (*id, vec![0.; layout.range.len()]))
            .collect();
        let affine = self.evaluate_affine(geometry, quadrature, &zero)?;
        let mut history = Vec::new();
        for (field, layout) in &self.previous {
            let (column, matrix) = self.history_matrix(*field, geometry, quadrature)?;
            let range = &self.fields[column].range;
            if range.len() != layout.range.len() {
                return Err(invalid(
                    "prepared history basis differs from its algebraic binding",
                ));
            }
            let block = matrix
                .matrix()
                .chunks_exact(affine.rows())
                .flat_map(|row| row[range.clone()].iter().copied())
                .collect();
            history.push((*field, range.len(), block));
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
        if !dyadics.is_empty() {
            let spaces = self
                .fields
                .iter()
                .map(|field| basis(field.space, self.reference))
                .collect::<Result<Vec<_>, _>>()?;
            let inverse = geometry.inverse_jacobian()?;
            for point in quadrature.points() {
                let mut physical = vec![0.; self.form.dimension];
                geometry.map_point(&point.coordinates, &mut physical)?;
                let tables = spaces
                    .iter()
                    .map(|space| space.tabulate(&point.coordinates))
                    .collect::<Result<Vec<_>, _>>()?;
                let values = tables.iter().map(|table| table.values().to_vec()).collect();
                let gradients = tables
                    .iter()
                    .map(|table| {
                        (0..table.values().len())
                            .map(|node| {
                                let gradient = physical_gradient(
                                    table.gradient(node).expect("supported gradient"),
                                    &inverse,
                                    self.form.dimension,
                                );
                                let mut result = [0.; 3];
                                result[..self.form.dimension].copy_from_slice(&gradient);
                                result
                            })
                            .collect()
                    })
                    .collect();
                let coefficients = data
                    .iter()
                    .map(|(row, coefficient)| {
                        Ok(point.weight
                            * geometry.measure_scale()
                            * self.row_multipliers[*row]
                            * coefficient.evaluate(&physical)?)
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                if coefficients.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("prepared nonlinear coefficient is non-finite"));
                }
                samples.push(Sample {
                    values,
                    gradients,
                    coefficients,
                });
            }
        }
        Ok(PreparedRegionCell {
            dimension: self.form.dimension,
            fields: self.fields.clone(),
            affine,
            history,
            dyadics,
            samples,
        })
    }

    fn history_matrix(
        &self,
        field: RawId,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<(usize, LocalContribution), Diagnostic> {
        let fields = self
            .fields
            .iter()
            .map(|field| (field.space, field.components))
            .collect::<Vec<_>>();
        let mut integrals = Vec::new();
        let mut data = Vec::new();
        let mut column = None;
        for (row, source) in self.form.rows.iter().enumerate() {
            for term in &source.terms {
                if term.trial != field {
                    continue;
                }
                let eliminated = self.eliminations.get(&term.trial);
                let scale = match (eliminated, term.derivative) {
                    (Some(_), false) => -1.,
                    (None, true) => self.step.expect("bound derivative step").recip(),
                    _ => continue,
                };
                let trial = eliminated.copied().unwrap_or(term.trial);
                let index = self
                    .fields
                    .iter()
                    .position(|layout| layout.field == trial)
                    .ok_or_else(|| invalid("history lacks an algebraic binding"))?;
                if column.is_some_and(|column| column != index) {
                    return Err(invalid("history has conflicting algebraic bindings"));
                }
                column = Some(index);
                integrals.push(super::integration::IntegralTerm {
                    row,
                    column: index,
                    pairing: term.pairing,
                    trial_scale: scale,
                    history: None,
                });
                data.push((row, &term.coefficient));
            }
        }
        let column =
            column.ok_or_else(|| invalid("consumed history lacks a mathematical action"))?;
        let matrix = super::integration::integrate(
            self.reference,
            &fields,
            &integrals,
            geometry,
            quadrature,
            |physical, coefficients, _, _| {
                for (value, (row, coefficient)) in coefficients.iter_mut().zip(&data) {
                    *value = self.row_multipliers[*row] * coefficient.evaluate(physical)?;
                }
                Ok(())
            },
        )?;
        Ok((column, matrix))
    }
}

impl PreparedRegionCell {
    pub(crate) fn add_load(&mut self, load: &[f64]) -> Result<(), Diagnostic> {
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
            .map(|(a, b)| a + b)
            .collect();
        self.affine = LocalContribution::new(
            self.affine.rows(),
            self.affine.columns(),
            self.affine.matrix().to_vec(),
            rhs,
        )?;
        Ok(())
    }
    fn rhs(&self, previous: &BTreeMap<RawId, Vec<f64>>) -> Result<Vec<f64>, Diagnostic> {
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
        let mut rhs = self.affine.rhs().to_vec();
        for (id, count, matrix) in &self.history {
            for (value, row) in rhs.iter_mut().zip(matrix.chunks_exact(*count)) {
                *value += row
                    .iter()
                    .zip(&previous[id])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            }
        }
        Ok(rhs)
    }

    pub(crate) fn evaluate(
        &self,
        previous: &BTreeMap<RawId, Vec<f64>>,
    ) -> Result<LocalContribution, Diagnostic> {
        if !self.dyadics.is_empty() {
            return Err(invalid(
                "nonlinear region evaluation requires an explicit candidate point",
            ));
        }
        LocalContribution::new(
            self.affine.rows(),
            self.affine.columns(),
            self.affine.matrix().to_vec(),
            self.rhs(previous)?,
        )
    }

    pub(crate) fn linearize(
        &self,
        previous: &BTreeMap<RawId, Vec<f64>>,
        point: &[f64],
    ) -> Result<RegionLinearization, Diagnostic> {
        self.action(previous, point, true)
    }

    pub(crate) fn residual(
        &self,
        previous: &BTreeMap<RawId, Vec<f64>>,
        point: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        Ok(self.action(previous, point, false)?.residual)
    }

    fn action(
        &self,
        previous: &BTreeMap<RawId, Vec<f64>>,
        point: &[f64],
        derivative: bool,
    ) -> Result<RegionLinearization, Diagnostic> {
        let count = self.affine.rows();
        if point.len() != count || point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "nonlinear region requires a finite exact candidate point",
            ));
        }
        let rhs = self.rhs(previous)?;
        let mut action = RegionLinearization {
            jacobian: if derivative {
                self.affine.matrix().to_vec()
            } else {
                Vec::new()
            },
            residual: self
                .affine
                .matrix()
                .chunks_exact(count)
                .zip(&rhs)
                .map(|(row, rhs)| row.iter().zip(point).map(|(a, b)| a * b).sum::<f64>() - rhs)
                .collect(),
        };
        let dimension = self.dimension;
        for sample in &self.samples {
            for (term, coefficient) in self.dyadics.iter().zip(&sample.coefficients) {
                let trial = &self.fields[term.column];
                let test = &self.fields[term.row];
                let mut value = [0.; 3];
                let mut gradient = [[0.; 3]; 3];
                for (node, shape) in sample.values[term.column].iter().enumerate() {
                    for i in 0..dimension {
                        let u = trial.scale * point[trial.range.start + node * dimension + i];
                        value[i] += shape * u;
                        for (entry, shape_gradient) in gradient[i][..dimension]
                            .iter_mut()
                            .zip(&sample.gradients[term.column][node][..dimension])
                        {
                            *entry += shape_gradient * u;
                        }
                    }
                }
                for (a, test_value) in sample.values[term.row].iter().enumerate() {
                    let test_gradient = &sample.gradients[term.row][a];
                    let directional_test = dot(&value, test_gradient, dimension);
                    for i in 0..dimension {
                        let r = test.range.start + a * dimension + i;
                        let advective = dot(&value, &gradient[i], dimension);
                        action.residual[r] += if term.split {
                            -0.5 * coefficient
                                * (advective * test_value - value[i] * directional_test)
                        } else {
                            coefficient * value[i] * directional_test
                        };
                        if !derivative {
                            continue;
                        }
                        for (b, trial_value) in sample.values[term.column].iter().enumerate() {
                            let directional_trial =
                                dot(&value, &sample.gradients[term.column][b], dimension);
                            for k in 0..dimension {
                                let c = trial.range.start + b * dimension + k;
                                let identity = if i == k { 1. } else { 0. };
                                let entry = if term.split {
                                    -0.5 * coefficient
                                        * trial.scale
                                        * ((trial_value * gradient[i][k]
                                            + identity * directional_trial)
                                            * test_value
                                            - trial_value
                                                * (identity * directional_test
                                                    + value[i] * test_gradient[k]))
                                } else {
                                    coefficient
                                        * trial.scale
                                        * trial_value
                                        * (identity * directional_test
                                            + value[i] * test_gradient[k])
                                };
                                action.jacobian[r * count + c] += entry;
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
            return Err(invalid("nonlinear region action is non-finite"));
        }
        Ok(action)
    }
}

fn dot(left: &[f64; 3], right: &[f64; 3], dimension: usize) -> f64 {
    left[..dimension]
        .iter()
        .zip(&right[..dimension])
        .map(|(a, b)| a * b)
        .sum()
}
