//! Exact correspondence for preaccepted simplices on rectangle Geometry v2.
//! No source rewrite or geometric classification tolerance is introduced.
use super::super::{
    cartesian_facet_roles, entity_coordinates, point_inside_cartesian, validate_parent_outward_cell,
};
use super::*;
use eqiora_schema::kernel::BoundarySide;

impl GeometryMeshCorrespondenceEnvelopeV1 {
    /// Bind an existing triangle mesh to the original classification-free rectangle.
    /// Boundary coordinates and side coverage must match exactly; no tolerance is inferred.
    pub fn from_planar_rectangle_v2_simplicial(
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let bounds = require_planar_rectangle_v2(geometry)?;
        if mesh.dimension() != 2 {
            return Err(invalid_artifact(
                "rectangle correspondence requires a triangle mesh",
            ));
        }
        let native = mesh.mesh();
        let cell_count = native.cells().len();
        let facet_count = native.entity_count(1).unwrap_or(0);
        let boundary_count = (0..facet_count)
            .filter(|&i| native.is_boundary_entity(MeshEntity::new(1, i)) == Some(true))
            .count();
        let limits = GeometryDecoderLimits::default();
        if cell_count
            .checked_add(boundary_count)
            .is_none_or(|count| count > limits.max_geometry_mesh_memberships)
            || SOURCE_EDGE_COUNT + 1 > limits.max_geometry_entities
        {
            return Err(invalid_artifact(
                "rectangle imported correspondence exceeds membership limits",
            ));
        }
        native.validate_signed_boundary_cancellation()?;
        let axis_bounds = bounds
            .iter()
            .map(|&[lower, upper]| (lower, upper))
            .collect::<Vec<_>>();
        if native
            .vertices()
            .iter()
            .any(|point| !point_inside_cartesian(point, &axis_bounds, 0.))
        {
            return Err(invalid_artifact(
                "imported mesh vertex lies outside exact rectangle bounds",
            ));
        }
        let mut sides: [Vec<(usize, [f64; 2])>; 4] = std::array::from_fn(|_| Vec::new());
        for index in 0..facet_count {
            let facet = MeshEntity::new(1, index);
            if native.is_boundary_entity(facet) != Some(true) {
                continue;
            }
            let points = entity_coordinates(native, facet)?;
            let roles = cartesian_facet_roles(&points, &axis_bounds, 0.);
            let [(axis, side)] = roles.as_slice() else {
                return Err(invalid_artifact(
                    "imported boundary edge must lie on exactly one exact rectangle side",
                ));
            };
            let adjacent = native
                .incidence(facet, 2)
                .ok_or_else(|| invalid_artifact("rectangle boundary omitted parent incidence"))?;
            let [parent] = adjacent.as_slice() else {
                return Err(invalid_artifact(
                    "rectangle boundary requires one parent cell",
                ));
            };
            validate_parent_outward_cell(native, parent.entity, *axis, *side, &axis_bounds)?;
            let side_index = 2 * axis + usize::from(*side == BoundarySide::Upper);
            let mut interval = [points[0][1 - axis], points[1][1 - axis]];
            interval.sort_by(f64::total_cmp);
            sides[side_index].push((index, interval));
        }
        let mut frontiers = Vec::with_capacity(4);
        for (side, mut intervals) in sides.into_iter().enumerate() {
            validate_exact_side_coverage(&mut intervals, bounds[1 - side / 2])?;
            let mut facet_indices = intervals
                .iter()
                .map(|(index, _)| portable(*index, "mesh facet"))
                .collect::<Result<Vec<_>, _>>()?;
            facet_indices.sort_unstable();
            frontiers.push(WireFrontierAssignment {
                parent_face: 0,
                geometry_edge: portable(side, "source edge")?,
                facet_indices,
            });
        }
        let correspondence = Self {
            wire: WireCorrespondenceV1::PlanarRectangleV2(WirePlanarRectangleV2CorrespondenceV1 {
                schema: CORRESPONDENCE_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                source: SIMPLICIAL_IMPORT_SOURCE.to_owned(),
                geometry_sha256: ArtifactDigest::from_sha256(geometry.digest_bytes()).to_string(),
                mesh_sha256: mesh.digest()?.to_string(),
                dimension: 2,
                face: WireFaceAssignment {
                    geometry_face: 0,
                    cell_indices: (0..cell_count)
                        .map(|cell| portable(cell, "mesh cell"))
                        .collect::<Result<Vec<_>, _>>()?,
                },
                frontiers,
            }),
        };
        correspondence.validate_local(limits)?;
        Ok(correspondence)
    }

    /// Reconstruct the complete exact rectangle-to-imported-mesh correspondence.
    pub fn validate_against_planar_rectangle_v2_simplicial(
        &self,
        geometry: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
    ) -> Result<(), Diagnostic> {
        if self != &Self::from_planar_rectangle_v2_simplicial(geometry, mesh)? {
            return Err(invalid_artifact(
                "rectangle imported correspondence differs from full resource replay",
            ));
        }
        Ok(())
    }
}

fn validate_exact_side_coverage(
    intervals: &mut [(usize, [f64; 2])],
    bounds: [f64; 2],
) -> Result<(), Diagnostic> {
    intervals.sort_by(|left, right| left.1[0].total_cmp(&right.1[0]));
    if intervals.is_empty()
        || intervals.iter().any(|(_, range)| range[0] >= range[1])
        || intervals[0].1[0] != bounds[0]
        || intervals.last().expect("nonempty intervals").1[1] != bounds[1]
        || intervals
            .windows(2)
            .any(|pair| pair[0].1[1] != pair[1].1[0])
    {
        return Err(invalid_artifact(
            "imported mesh boundary intervals must cover each rectangle side exactly once without gaps or overlaps",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[test]
fn exact_side_coverage_rejects_gaps_overlaps_duplicates_and_zero_length() {
    for mut intervals in [
        vec![(0, [0., 0.4]), (1, [0.6, 1.])],
        vec![(0, [0., 0.6]), (1, [0.4, 1.])],
        vec![(0, [0., 1.]), (1, [0., 1.])],
        vec![(0, [0., 0.]), (1, [0., 1.])],
        vec![(0, [0.00000000000001, 1.])],
    ] {
        assert!(validate_exact_side_coverage(&mut intervals, [0., 1.]).is_err());
    }
    assert!(validate_exact_side_coverage(&mut [(1, [0.5, 1.]), (0, [0., 0.5])], [0., 1.]).is_ok());
}
