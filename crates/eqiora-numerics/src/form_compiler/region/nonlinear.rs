//! Equation-derived nonlinear flux actions, independent of physical field names.

use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule};

use super::{BoundRegionForm, Data, binding::basis, invalid};

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
    pub(crate) fn natural_facet_action(
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
        derivative: bool,
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
            jacobian: if derivative {
                vec![0.0; count * count]
            } else {
                Vec::new()
            },
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
                        if !derivative {
                            continue;
                        }
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
}
