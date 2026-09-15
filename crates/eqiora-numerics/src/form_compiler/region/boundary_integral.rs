use eqiora_assembly::LocalContribution;
use eqiora_meshing::{
    AffineGeometryMap, EntityIncidence, GeometryMap, QuadratureRule, ReferenceCellFamily,
    ReferenceTopology,
};

use super::*;
use crate::discrete_space::{CellConstantSpace, DiscreteSpace, HypercubeQ1Space, SimplexP1Space};

impl BoundRegionForm {
    /// Integrate physical parent-outward flux into the complete parent-cell map.
    /// The caller owns the authenticated facet-to-cell incidence and Domain binding.
    pub(crate) fn evaluate_natural_facet(
        &self,
        field: RawId,
        cell: &AffineGeometryMap,
        facet: (&AffineGeometryMap, EntityIncidence, &[usize]),
        rule: &QuadratureRule,
        datum: impl Fn(&[f64], &[f64]) -> Result<Vec<f64>, Diagnostic>,
    ) -> Result<LocalContribution, Diagnostic> {
        let (facet, incidence, parent_vertices) = facet;
        let dimension = self.form.dimension;
        if cell.reference_cell() != self.reference
            || cell.physical_dimension() != dimension
            || facet.physical_dimension() != dimension
            || facet.reference_cell().dimension() + 1 != dimension
            || facet.reference_cell() != rule.reference_cell()
        {
            return Err(invalid(
                "natural flux requires a matching parent-cell and facet geometry",
            ));
        }
        let row = self
            .fields
            .iter()
            .position(|layout| layout.field == field)
            .ok_or_else(|| invalid("natural flux has a foreign tested Field"))?;
        let layout = &self.fields[row];
        let space = super::binding::basis(layout.space, self.reference)?;
        let topology = ReferenceTopology::new(self.reference)?;
        let expected = topology
            .entity(dimension - 1, incidence.local_ordinal)
            .ok_or_else(|| invalid("natural flux has a foreign parent facet ordinal"))?
            .vertex_ordinals();
        let mut supplied = parent_vertices.to_vec();
        supplied.sort_unstable();
        if supplied != expected {
            return Err(invalid(
                "natural flux vertex embedding differs from its exact parent facet",
            ));
        }
        let facet_space: Box<dyn DiscreteSpace> = match facet.reference_cell().family() {
            ReferenceCellFamily::Point => Box::new(CellConstantSpace::new(facet.reference_cell())),
            ReferenceCellFamily::Simplex => Box::new(SimplexP1Space::new(dimension - 1)?),
            ReferenceCellFamily::Hypercube => Box::new(HypercubeQ1Space::new(dimension - 1)?),
        };
        if facet_space.local_dofs().len() != parent_vertices.len() {
            return Err(invalid(
                "natural flux facet basis and vertex embedding differ",
            ));
        }
        let count = self.fields.last().expect("nonempty bound form").range.end;
        let entries = count
            .checked_mul(count)
            .ok_or_else(|| invalid("natural flux local matrix size overflow"))?;
        let mut rhs = vec![0.0; count];
        let normal = parent_outward_normal(cell, incidence)?;
        for point in rule.points() {
            let mut physical = vec![0.0; dimension];
            facet.map_point(&point.coordinates, &mut physical)?;
            let test = facet_space.tabulate(&point.coordinates)?;
            let value = datum(&physical, &normal)?;
            if value.len() != layout.components || value.iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    "natural flux datum differs from the exact Field component shape",
                ));
            }
            let weight = point.weight * facet.measure_scale() * self.row_multipliers[row];
            // P1/Q1 vertex functions restrict to the intrinsic facet basis.
            // Off-facet vertex functions and the admitted MINI interior bubble
            // have identically zero trace. No physical inverse or clipping enters.
            for (node, dof) in space.local_dofs().iter().enumerate() {
                if dof.entity_dimension() != 0 {
                    continue;
                }
                let Some(facet_vertex) = parent_vertices
                    .iter()
                    .position(|vertex| *vertex == dof.entity_ordinal())
                else {
                    continue;
                };
                let test = test.values()[facet_vertex];
                for (component, value) in value.iter().enumerate() {
                    rhs[layout.range.start + node * layout.components + component] +=
                        weight * test * value;
                }
            }
        }
        LocalContribution::new(count, count, vec![0.0; entries], rhs)
    }
}

pub(super) fn parent_outward_normal(
    cell: &AffineGeometryMap,
    incidence: EntityIncidence,
) -> Result<Vec<f64>, Diagnostic> {
    let dimension = cell.physical_dimension();
    if dimension == 0 || incidence.entity.dimension() != dimension {
        return Err(invalid(
            "natural flux requires an exact positive-dimensional parent incidence",
        ));
    }
    let topology = ReferenceTopology::new(cell.reference_cell())?;
    let facet = topology
        .entity(dimension - 1, incidence.local_ordinal)
        .ok_or_else(|| invalid("natural flux has a foreign parent facet ordinal"))?;
    let vertices = facet.vertex_ordinals();
    let mut reference_normal = vec![0.0; dimension];
    match cell.reference_cell().family() {
        ReferenceCellFamily::Simplex => {
            let missing = (0..=dimension)
                .find(|vertex| !vertices.contains(vertex))
                .ok_or_else(|| invalid("simplex facet has no opposite parent vertex"))?;
            if missing == 0 {
                reference_normal.fill(1.0);
            } else {
                reference_normal[missing - 1] = -1.0;
            }
        }
        ReferenceCellFamily::Hypercube => {
            let axis = (0..dimension)
                .find(|axis| {
                    vertices
                        .iter()
                        .all(|vertex| (vertex >> axis) & 1 == (vertices[0] >> axis) & 1)
                })
                .ok_or_else(|| invalid("hypercube facet has no fixed parent axis"))?;
            reference_normal[axis] = if (vertices[0] >> axis) & 1 == 0 {
                -1.0
            } else {
                1.0
            };
        }
        ReferenceCellFamily::Point => {
            return Err(invalid(
                "natural flux requires a positive-dimensional parent",
            ));
        }
    }
    // Covectors transform by J^{-T}; the exact reference facet owns the sign.
    let inverse = cell.inverse_jacobian()?;
    let mut normal = (0..dimension)
        .map(|physical| {
            (0..dimension)
                .map(|reference| {
                    inverse[reference * dimension + physical] * reference_normal[reference]
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let norm = normal
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value));
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid("natural flux facet has no finite nonzero normal"));
    }
    for component in &mut normal {
        *component /= norm;
    }
    Ok(normal)
}
