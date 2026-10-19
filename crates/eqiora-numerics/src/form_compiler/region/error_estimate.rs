//! Residual indicators consume the admitted equation, not an algebraic residual.

use super::*;
use crate::affine_fem::{simplex_p1_physical_gradients, weighted_gradient};
use eqiora_meshing::{
    GeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, ReferenceCell,
    SimplicialMesh,
};

impl BoundRegionForm<f64> {
    /// Squared H1-seminorm indicators in canonical triangle order. The caller
    /// proves homogeneous essential boundary coverage on the complete mesh.
    pub(crate) fn p1_residual_indicators(
        &self,
        mesh: &SimplicialMesh,
        values: &[f64],
        quadrature: &QuadratureRule,
    ) -> Result<Vec<f64>, Diagnostic> {
        let triangle = ReferenceCell::simplex(2)?;
        let [row] = self.form.rows.as_slice() else {
            return Err(invalid(
                "residual estimation requires one scalar diffusion equation",
            ));
        };
        let [field] = self.fields.as_slice() else {
            return Err(invalid("residual estimation requires one scalar P1 Field"));
        };
        if mesh.topological_dimension() != 2
            || self.reference != triangle
            || quadrature.reference_cell() != triangle
            || quadrature.points().iter().any(|point| point.weight <= 0.0)
            || field.space
                != eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN)
            || field.components != 1
            || field.scale != 1.0
            || self.time.is_some()
            || row.is_nonlinear()
            || values.len() != mesh.vertices().len()
            || values.iter().any(|value| !value.is_finite())
            || row.terms.is_empty()
            || row.terms.iter().any(|term| {
                term.trial != field.field
                    || term.derivative
                    || term.pairing != Pairing::Gradient
                    || term.coefficient.spatial()
            })
            || row
                .flux
                .iter()
                .any(|flux| !matches!(flux, flux::FluxTerm::Trial(_)))
        {
            return Err(invalid(
                "residual estimation requires real planar P1 constant diffusion without storage, reaction or transport",
            ));
        }
        if row
            .forcing
            .iter()
            .filter_map(Data::polynomial_degree)
            .any(|degree| {
                quadrature
                    .polynomial_exactness()
                    .is_none_or(|exactness| exactness < degree.saturating_mul(2))
            })
        {
            return Err(invalid(
                "residual quadrature does not cover the squared polynomial source degree",
            ));
        }
        let diffusion = row.terms.iter().try_fold(0.0, |sum, term| {
            term.coefficient
                .evaluate(&[0.0, 0.0])
                .map(|value| sum + value)
        })?;
        if !diffusion.is_finite() || diffusion <= 0.0 {
            return Err(invalid(
                "residual estimation requires finite positive diffusion",
            ));
        }
        let mut indicators = Vec::with_capacity(mesh.cells().len());
        let mut gradients = Vec::with_capacity(mesh.cells().len());
        let mut edge_cells = vec![Vec::new(); mesh.entity_count(1).expect("triangle edges")];
        for (index, vertices) in mesh.cells().iter().enumerate() {
            let geometry = mesh
                .geometry_map(MeshEntity::new(2, index))
                .expect("accepted triangle");
            let coefficients = vertices
                .iter()
                .map(|&vertex| values[vertex])
                .collect::<Vec<_>>();
            gradients.push(weighted_gradient(
                &coefficients,
                &simplex_p1_physical_gradients::<2>(&geometry)?,
            ));
            let mut diameter_squared = 0.0_f64;
            for edge in mesh
                .incidence(MeshEntity::new(2, index), 1)
                .expect("triangle edge closure")
            {
                let endpoints = mesh.entity_vertices(edge.entity).expect("edge endpoints");
                let p = &mesh.vertices()[endpoints[0].index()];
                let q = &mesh.vertices()[endpoints[1].index()];
                diameter_squared =
                    diameter_squared.max((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2));
                edge_cells[edge.entity.index()].push(index);
            }
            let mut integral = 0.0;
            let mut point = [0.0; 2];
            for sample in quadrature.points() {
                geometry.map_point(&sample.coordinates, &mut point)?;
                let forcing = row.forcing.iter().try_fold(0.0, |sum, datum| {
                    datum.evaluate(&point).map(|value| sum + value)
                })? / diffusion;
                integral += sample.weight * geometry.measure_scale() * forcing.powi(2);
            }
            indicators.push(diameter_squared * integral);
        }
        for (edge, cells) in edge_cells.iter().enumerate() {
            let [left, right] = cells.as_slice() else {
                continue;
            };
            let endpoints = mesh
                .entity_vertices(MeshEntity::new(1, edge))
                .expect("edge endpoints");
            let p = &mesh.vertices()[endpoints[0].index()];
            let q = &mesh.vertices()[endpoints[1].index()];
            // h_E * integral_E jump(grad u . n)^2 ds. For a straight
            // edge and P1 gradients, its length cancels the unit normal's
            // denominator. Share each interior-edge contribution equally.
            let normal = [q[1] - p[1], p[0] - q[0]];
            let jump = (gradients[*left][0] - gradients[*right][0]) * normal[0]
                + (gradients[*left][1] - gradients[*right][1]) * normal[1];
            let contribution = 0.5 * jump.powi(2);
            indicators[*left] += contribution;
            indicators[*right] += contribution;
        }
        if indicators
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid("residual indicators overflowed or became negative"));
        }
        Ok(indicators)
    }
}
