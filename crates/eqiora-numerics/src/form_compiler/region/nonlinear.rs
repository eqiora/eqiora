//! Equation-derived nonlinear flux actions, independent of physical field names.

use std::collections::BTreeMap;

use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule};

use super::{BoundRegionForm, Data, binding::basis, invalid};
use crate::affine_fem::physical_gradient;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DyadicTerm {
    pub field: RawId,
    pub coefficient: Data,
    /// A retained div(field)=0 constraint permits the skew split by parts.
    /// This is a mathematical condition, not a velocity or fluid role.
    pub split: bool,
}

pub(crate) struct RegionLinearization {
    pub(crate) jacobian: Vec<f64>,
    pub(crate) residual: Vec<f64>,
}

impl RegionLinearization {
    pub(crate) fn into_contribution(self, point: &[f64]) -> Result<LocalContribution, Diagnostic> {
        let count = self.residual.len();
        if point.len() != count {
            return Err(invalid("region linearization point shape mismatch"));
        }
        let rhs = self
            .jacobian
            .chunks_exact(count)
            .zip(&self.residual)
            .map(|(row, residual)| {
                row.iter().zip(point).map(|(j, u)| j * u).sum::<f64>() - residual
            })
            .collect();
        LocalContribution::new(count, count, self.jacobian, rhs)
    }
}

impl BoundRegionForm {
    /// Anonymous source/test pairing, shared by caller-provided mathematical loads.
    pub(crate) fn pair_load(
        &self,
        field: RawId,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        datum: impl Fn(&[f64]) -> Result<Vec<f64>, Diagnostic>,
    ) -> Result<LocalContribution, Diagnostic> {
        let row = self
            .fields
            .iter()
            .position(|layout| layout.field == field)
            .ok_or_else(|| invalid("source pairing has a foreign tested Field"))?;
        let offset = self.fields[..row]
            .iter()
            .map(|layout| layout.components)
            .sum::<usize>();
        super::integration::integrate(
            self.reference,
            &self
                .fields
                .iter()
                .map(|layout| (layout.space, layout.components))
                .collect::<Vec<_>>(),
            &[],
            geometry,
            quadrature,
            |point, _, forcing, _| {
                let values = datum(point)?;
                if values.len() != self.fields[row].components {
                    return Err(invalid("source pairing component mismatch"));
                }
                forcing[offset..offset + values.len()].copy_from_slice(&values);
                Ok(())
            },
        )
    }

    /// Complete a nonlinear divergence on an authored natural-flux boundary.
    /// A divergence-free skew split retains half the physical dyadic flux;
    /// conservative integration by parts retains its complete flux.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linearize_natural_facet(
        &self,
        field: RawId,
        cell: &AffineGeometryMap,
        facet: (
            &AffineGeometryMap,
            eqiora_meshing::EntityIncidence,
            &[usize],
        ),
        rule: &QuadratureRule,
        point: &[f64],
        datum: impl Fn(&[f64], &[f64]) -> Result<Vec<f64>, Diagnostic>,
    ) -> Result<RegionLinearization, Diagnostic> {
        let traction = self.evaluate_natural_facet(field, cell, facet, rule, datum)?;
        let count = traction.rows();
        if point.len() != count || point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "nonlinear natural facet requires an exact finite parent point",
            ));
        }
        let mut action = RegionLinearization {
            jacobian: vec![0.0; count * count],
            residual: traction.rhs().iter().map(|value| -*value).collect(),
        };
        let (facet, incidence, parent_vertices) = facet;
        let dimension = self.form.dimension;
        let row_index = self
            .fields
            .iter()
            .position(|layout| layout.field == field)
            .expect("validated natural Field");
        let test = &self.fields[row_index];
        let normal = super::boundary_integral::parent_outward_normal(cell, incidence)?;
        let facet_space: Box<dyn crate::discrete_space::DiscreteSpace> = if dimension == 1 {
            Box::new(crate::discrete_space::CellConstantSpace::new(
                facet.reference_cell(),
            ))
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
        let required_degree =
            if facet.reference_cell().family() == eqiora_meshing::ReferenceCellFamily::Hypercube {
                3 * (dimension - 1)
            } else {
                3
            };
        if dimension > 1
            && rule
                .polynomial_exactness()
                .is_none_or(|order| order < required_degree)
        {
            return Err(invalid(
                "nonlinear natural facet requires exact cubic trace products",
            ));
        }
        for sample in rule.points() {
            let facet_table = facet_space.tabulate(&sample.coordinates)?;
            let mut reference = vec![0.0; dimension];
            for (node, vertex) in parent_vertices.iter().enumerate() {
                for (axis, coordinate) in reference.iter_mut().enumerate() {
                    let value = match self.reference.family() {
                        eqiora_meshing::ReferenceCellFamily::Simplex => {
                            if *vertex == axis + 1 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        eqiora_meshing::ReferenceCellFamily::Hypercube => {
                            2.0 * ((vertex >> axis) & 1) as f64 - 1.0
                        }
                        _ => {
                            return Err(invalid(
                                "nonlinear flux requires positive-dimensional parent",
                            ));
                        }
                    };
                    *coordinate += facet_table.values()[node] * value;
                }
            }
            let tables = spaces
                .iter()
                .map(|space| {
                    let table = space.tabulate(&reference)?;
                    Ok(table
                        .values()
                        .iter()
                        .zip(space.local_dofs())
                        .map(|(value, dof)| {
                            if dof.entity_dimension() == dimension {
                                0.0
                            } else {
                                *value
                            }
                        })
                        .collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut physical = vec![0.0; dimension];
            facet.map_point(&sample.coordinates, &mut physical)?;
            for term in &self.form.rows[row_index].dyadics {
                let column = self
                    .fields
                    .iter()
                    .position(|layout| layout.field == term.field)
                    .ok_or_else(|| invalid("natural dyadic trial lacks a binding"))?;
                let trial = &self.fields[column];
                let value = (0..dimension)
                    .map(|i| {
                        tables[column]
                            .iter()
                            .enumerate()
                            .map(|(b, shape)| {
                                shape * trial.scale * point[trial.range.start + b * dimension + i]
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let normal_value = value.iter().zip(&normal).map(|(u, n)| u * n).sum::<f64>();
                let coefficient = -term.coefficient.evaluate(&physical)?
                    * self.row_multipliers[row_index]
                    * sample.weight
                    * facet.measure_scale()
                    * if term.split { 0.5 } else { 1.0 };
                for (a, test_value) in tables[row_index].iter().enumerate() {
                    for (i, component_value) in value.iter().enumerate() {
                        let r = test.range.start + a * dimension + i;
                        action.residual[r] +=
                            coefficient * test_value * normal_value * component_value;
                        for (b, trial_value) in tables[column].iter().enumerate() {
                            for (k, normal_component) in normal.iter().enumerate() {
                                let c = trial.range.start + b * dimension + k;
                                action.jacobian[r * count + c] += coefficient
                                    * test_value
                                    * trial_value
                                    * trial.scale
                                    * (normal_component * component_value
                                        + if i == k { normal_value } else { 0.0 });
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

    /// Residual and exact derivative of the authored region at one algebraic point.
    pub(crate) fn linearize(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        previous: &BTreeMap<RawId, Vec<f64>>,
        point: &[f64],
    ) -> Result<RegionLinearization, Diagnostic> {
        let affine = self.evaluate_affine(geometry, quadrature, previous)?;
        let count = affine.rows();
        if point.len() != count || point.iter().any(|u| !u.is_finite()) {
            return Err(invalid(
                "nonlinear region requires a finite exact candidate point",
            ));
        }
        let mut action = RegionLinearization {
            jacobian: affine.matrix().to_vec(),
            residual: affine
                .matrix()
                .chunks_exact(count)
                .zip(affine.rhs())
                .map(|(row, rhs)| row.iter().zip(point).map(|(j, u)| j * u).sum::<f64>() - rhs)
                .collect(),
        };
        let dimension = self.form.dimension;
        let inverse = geometry.inverse_jacobian()?;
        let spaces = self
            .fields
            .iter()
            .map(|layout| basis(layout.space, self.reference))
            .collect::<Result<Vec<_>, _>>()?;
        for sample in quadrature.points() {
            let mut physical = vec![0.0; dimension];
            geometry.map_point(&sample.coordinates, &mut physical)?;
            let tables = spaces
                .iter()
                .map(|space| space.tabulate(&sample.coordinates))
                .collect::<Result<Vec<_>, _>>()?;
            let gradients = tables
                .iter()
                .map(|table| {
                    (0..table.values().len())
                        .map(|node| {
                            physical_gradient(
                                table.gradient(node).expect("supported gradient"),
                                &inverse,
                                dimension,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let weight = sample.weight * geometry.measure_scale();
            for (row_index, row) in self.form.rows.iter().enumerate() {
                let test = &self.fields[row_index];
                for term in &row.dyadics {
                    let column = self
                        .fields
                        .iter()
                        .position(|layout| layout.field == term.field)
                        .ok_or_else(|| invalid("dyadic trial lacks an algebraic binding"))?;
                    let trial = &self.fields[column];
                    let mut value = vec![0.0; dimension];
                    let mut gradient = vec![vec![0.0; dimension]; dimension];
                    for (node, basis_value) in tables[column].values().iter().enumerate() {
                        for i in 0..dimension {
                            let u = trial.scale * point[trial.range.start + node * dimension + i];
                            value[i] += basis_value * u;
                            for j in 0..dimension {
                                gradient[i][j] += gradients[column][node][j] * u;
                            }
                        }
                    }
                    let coefficient = weight
                        * self.row_multipliers[row_index]
                        * term.coefficient.evaluate(&physical)?;
                    for (a, test_value) in tables[row_index].values().iter().enumerate() {
                        let test_gradient = &gradients[row_index][a];
                        let directional_test = value
                            .iter()
                            .zip(test_gradient)
                            .map(|(u, g)| u * g)
                            .sum::<f64>();
                        for i in 0..dimension {
                            let r = test.range.start + a * dimension + i;
                            let advective = value
                                .iter()
                                .zip(&gradient[i])
                                .map(|(u, g)| u * g)
                                .sum::<f64>();
                            action.residual[r] += if term.split {
                                -0.5 * coefficient
                                    * (advective * test_value - value[i] * directional_test)
                            } else {
                                coefficient * value[i] * directional_test
                            };
                            for (b, trial_value) in tables[column].values().iter().enumerate() {
                                let directional_trial = value
                                    .iter()
                                    .zip(&gradients[column][b])
                                    .map(|(u, g)| u * g)
                                    .sum::<f64>();
                                for k in 0..dimension {
                                    let c = trial.range.start + b * dimension + k;
                                    let identity = if i == k { 1.0 } else { 0.0 };
                                    let derivative = if term.split {
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
                                    action.jacobian[r * count + c] += derivative;
                                }
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
            .any(|v| !v.is_finite())
        {
            return Err(invalid("nonlinear region action is non-finite"));
        }
        Ok(action)
    }
}
