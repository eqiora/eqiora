use eqiora_assembly::LocalContribution;
use eqiora_meshing::{
    AffineGeometryMap, EntityIncidence, GeometryMap, QuadratureRule, ReferenceCellFamily,
    ReferenceTopology,
};

use super::*;
use crate::affine_fem::parent_outward_normal;
use crate::discrete_space::DiscreteSpace;

impl<S: Coefficient> BoundRegionForm<S> {
    /// Integrate physical parent-outward flux into the complete parent-cell map.
    /// The caller owns the authenticated facet-to-cell incidence and Domain binding.
    pub(crate) fn evaluate_natural_facet(
        &self,
        field: RawId,
        cell: &AffineGeometryMap,
        facet: (&AffineGeometryMap, EntityIncidence, &[usize]),
        rule: &QuadratureRule,
        datum: impl Fn(&[f64], &[f64]) -> Result<Vec<S>, Diagnostic>,
    ) -> Result<LocalContribution<S>, Diagnostic> {
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
        if matches!(
            layout.space.family(),
            eqiora_realization::SpaceFamily::TetrahedralEdge
                | eqiora_realization::SpaceFamily::TetrahedralFace
        ) {
            return Err(invalid(
                "moment spaces require tangential/normal trace admission before natural facet assembly",
            ));
        }
        let space = crate::discrete_space::DiscreteSpace::new(layout.space, self.reference)?;
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
        let facet_reference = facet.reference_cell();
        let facet_space = match facet_reference.family() {
            ReferenceCellFamily::Point => {
                DiscreteSpace::new(eqiora_realization::Space::cell_constant(), facet_reference)?
            }
            ReferenceCellFamily::Simplex | ReferenceCellFamily::Hypercube => {
                let space =
                    eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN);
                DiscreteSpace::new(space, facet_reference)?
            }
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
        let mut rhs = vec![<S as From<f64>>::from(0.0); count];
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
                        <S as From<f64>>::from(weight * test) * *value;
                }
            }
        }
        LocalContribution::new(
            count,
            count,
            vec![<S as From<f64>>::from(0.0); entries],
            rhs,
        )
    }
}
