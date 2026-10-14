use std::collections::BTreeSet;

use eqiora_core::{Diagnostic, Id, entity::kinds};
use eqiora_meshing::{FixedTopologyGeometryState, SimplicialMesh};

use crate::simplicial_ale_fsi::P1HarmonicMeshMotionAction;
use crate::simplicial_fsi::FixedReferenceFsiPartition;

use super::{COMPONENTS, ProjectionFields};

pub(super) fn derive_target_geometry(
    fields: ProjectionFields,
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    motion: &P1HarmonicMeshMotionAction<2>,
    solid_displacement: &[[f64; COMPONENTS]],
) -> Result<FixedTopologyGeometryState<2>, Diagnostic> {
    motion.validate_reference(reference, partition)?;
    let field = fields.displacement;
    let values = partition
        .domain_vertices(fields.solid_domain)
        .ok_or_else(|| super::super::invalid("remesh driver Domain is absent"))?
        .iter()
        .map(|&vertex| (vertex, solid_displacement[vertex.index()]))
        .collect();
    let displacement = motion.apply([(field, &values)])?;
    let coordinates = reference
        .vertices()
        .iter()
        .zip(displacement)
        .map(|(reference, displacement)| {
            let coordinate = vec![
                reference[0] + displacement[0],
                reference[1] + displacement[1],
            ];
            coordinate
                .iter()
                .all(|value| value.is_finite())
                .then_some(coordinate)
                .ok_or_else(|| {
                    super::super::invalid("ALE FSI remesh target coordinates overflowed")
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    FixedTopologyGeometryState::<2>::new(reference, coordinates)
}

pub(super) fn interface_facet_indices(
    connection: Id<kinds::Connection>,
    partition: &FixedReferenceFsiPartition<2>,
) -> BTreeSet<usize> {
    partition
        .traces()
        .iter()
        .filter(|trace| {
            trace.quotient.source()
                == eqiora_realization::ConformingTraceSource::ConservingConnection(connection)
        })
        .flat_map(|trace| trace.facets.iter().map(|witness| witness.facet.index()))
        .collect()
}
