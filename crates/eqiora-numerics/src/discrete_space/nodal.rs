//! Entity-supported nodal interpolation and exact reference-facet closure.
use super::*;

pub(super) fn quadratic_dofs(cell: ReferenceCell) -> Result<Vec<LocalDof>, Diagnostic> {
    let topology = ReferenceTopology::new(cell)?;
    let last = if cell.family() == ReferenceCellFamily::Simplex {
        1
    } else {
        cell.dimension()
    };
    Ok((0..=last)
        .flat_map(|dimension| {
            (0..topology.entity_count(dimension).expect("reference stratum")).map(move |ordinal| {
                LocalDof {
                    entity_dimension: dimension,
                    entity_ordinal: ordinal,
                    slot: 0,
                }
            })
        })
        .collect())
}

fn vertex_point(cell: ReferenceCell, vertex: usize) -> Vec<f64> {
    (0..cell.dimension())
        .map(|axis| match cell.family() {
            ReferenceCellFamily::Simplex => {
                if vertex == axis + 1 {
                    1.0
                } else {
                    0.0
                }
            }
            ReferenceCellFamily::Hypercube => {
                if vertex & (1 << axis) == 0 {
                    -1.0
                } else {
                    1.0
                }
            }
            ReferenceCellFamily::Point => 0.0,
        })
        .collect()
}

impl DiscreteSpace {
    /// Reference interpolation point of one nodal coefficient. Geometry nodes
    /// remain independent; hierarchical bubbles and integral moments are not nodal.
    pub(crate) fn interpolation_point(&self, local: usize) -> Result<Vec<f64>, Diagnostic> {
        let dof = self
            .local_dofs
            .get(local)
            .ok_or_else(|| invalid_space("foreign interpolation coefficient"))?;
        if self.vector_element()
            || (self.space.family() == SpaceFamily::SimplexP1Bubble && dof.entity_dimension != 0)
        {
            return Err(invalid_space(
                "coefficient functional is not a nodal point evaluation",
            ));
        }
        let topology = ReferenceTopology::new(self.cell)?;
        let vertices = topology
            .entity(dof.entity_dimension, dof.entity_ordinal)
            .ok_or_else(|| {
                invalid_space("interpolation support is absent from reference topology")
            })?
            .vertex_ordinals();
        let mut point = vec![0.0; self.cell.dimension()];
        for &vertex in vertices {
            for (value, coordinate) in point.iter_mut().zip(vertex_point(self.cell, vertex)) {
                *value += coordinate / vertices.len() as f64;
            }
        }
        Ok(point)
    }

    /// Parent basis ordinals whose restrictions can be nonzero on this facet.
    /// P0 is constant on every facet; MINI/Q2 interior functions have zero trace.
    pub(crate) fn trace_dofs(&self, facet: usize) -> Result<Vec<usize>, Diagnostic> {
        if self.cell.dimension() == 0 || self.vector_element() {
            return Err(invalid_space(
                "nodal trace requires a positive-dimensional scalar cell",
            ));
        }
        let topology = ReferenceTopology::new(self.cell)?;
        let vertices = topology
            .entity(self.cell.dimension() - 1, facet)
            .ok_or_else(|| invalid_space("foreign reference facet"))?
            .vertex_ordinals();
        if self.space.family() == SpaceFamily::CellConstant {
            return Ok(vec![0]);
        }
        Ok(self
            .local_dofs
            .iter()
            .enumerate()
            .filter_map(|(index, dof)| {
                let support = topology
                    .entity(dof.entity_dimension, dof.entity_ordinal)
                    .expect("validated nodal entity")
                    .vertex_ordinals();
                support
                    .iter()
                    .all(|v| vertices.contains(v))
                    .then_some(index)
            })
            .collect())
    }
}

pub(super) fn check_topology_permutation(
    cell: ReferenceCell,
    permutation: &VertexPermutation,
) -> Result<(), Diagnostic> {
    if cell.family() != ReferenceCellFamily::Hypercube {
        return Ok(());
    }
    let topology = ReferenceTopology::new(cell)?;
    for dimension in 1..cell.dimension() {
        for index in 0..topology.entity_count(dimension).expect("reference stratum") {
            let mut images = topology
                .entity(dimension, index)
                .expect("reference entity")
                .vertex_ordinals()
                .iter()
                .map(|v| permutation.images()[*v])
                .collect::<Vec<_>>();
            images.sort_unstable();
            if !(0..topology.entity_count(dimension).expect("reference stratum")).any(|other| {
                topology
                    .entity(dimension, other)
                    .expect("reference entity")
                    .vertex_ordinals()
                    == images
            }) {
                return Err(invalid_space(
                    "vertex permutation does not preserve reference cube topology",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn permuted_dofs(
    space: &DiscreteSpace,
    permutation: &VertexPermutation,
) -> Result<Vec<(usize, i8)>, Diagnostic> {
    let topology = ReferenceTopology::new(space.cell)?;
    space
        .local_dofs
        .iter()
        .map(|dof| {
            let mut images = topology
                .entity(dof.entity_dimension, dof.entity_ordinal)
                .expect("validated nodal entity")
                .vertex_ordinals()
                .iter()
                .map(|v| permutation.images()[*v])
                .collect::<Vec<_>>();
            images.sort_unstable();
            let destination = space
                .local_dofs
                .iter()
                .position(|candidate| {
                    candidate.entity_dimension == dof.entity_dimension
                        && topology
                            .entity(candidate.entity_dimension, candidate.entity_ordinal)
                            .expect("validated nodal entity")
                            .vertex_ordinals()
                            == images
                })
                .ok_or_else(|| invalid_space("permutation does not preserve nodal support"))?;
            Ok((destination, 1))
        })
        .collect()
}
