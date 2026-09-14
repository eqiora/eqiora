use eqiora_core::Diagnostic;
use eqiora_meshing::{FixedTopologyGeometryState2d, SimplicialMesh};
use eqiora_realization::P1HarmonicMeshMotionPolicy;

use crate::simplicial_ale_fsi::P1HarmonicMeshMotionAction;
use crate::simplicial_fsi::FixedReferenceFsiPartition;

use super::COMPONENTS;

pub(super) fn derive_target_geometry(
    policy: P1HarmonicMeshMotionPolicy,
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    motion: &P1HarmonicMeshMotionAction<2>,
    solid_displacement: &[[f64; COMPONENTS]],
) -> Result<FixedTopologyGeometryState2d, Diagnostic> {
    motion.validate_reference(reference, partition)?;
    let field = policy.solid_displacement();
    let values = partition
        .domain_vertices(policy.solid_domain())
        .ok_or_else(|| super::super::invalid("remesh driver Domain is absent"))?
        .iter()
        .map(|&vertex| (vertex, solid_displacement[vertex.index()]))
        .collect();
    let displacement = motion.apply(field, &values)?;
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
    FixedTopologyGeometryState2d::new(reference, coordinates)
}
