//! Expectations authored before the import implementation: two square triangles
//! and one unit tetrahedron, with analytic boundaries and unchanged input cells.
use super::*;
use eqiora_artifact::{MshCoordinateUnit, MshImportPolicyV1};
use eqiora_geometry::{PlanarFace, PlanarRegion};

const SQUARE: &[u8] = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n$Elements\n1 2 1 2\n2 1 2 2\n1 1 2 3\n2 1 3 4\n$EndElements\n";
const TETRAHEDRON: &[u8] = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4\n$EndElements\n";

fn square(width: f64) -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_region(&PlanarRegion::new(
        vec![[0., 0.], [width, 0.], [width, 1.], [0., 1.]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
        vec![NamedEntitySet::new("body", 2, vec![0]), NamedEntitySet::new("outer", 1, vec![0, 1, 2, 3])],
        1e-12,
    ).unwrap()).unwrap()
}

fn tetrahedron() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_convex_polyhedra(
        vec![[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]],
        vec![vec![vec![0,2,1], vec![0,1,3], vec![0,3,2], vec![1,2,3]]],
        vec![NamedEntitySet::new("body", 3, vec![0]), NamedEntitySet::new("outer", 2, vec![0,1,2,3])],
        1e-12,
    ).unwrap()
}

fn policy(facets: usize, bytes: usize) -> MshImportPolicyV1 {
    MshImportPolicyV1::new(MshCoordinateUnit::Metres, 0.01, facets, bytes).unwrap()
}

#[test]
fn external_msh_retains_two_triangles_or_one_tetrahedron_and_replays_honest_origin() {
    for (geometry, source, dimension, cells) in [
        (square(1.), SQUARE, 2, 2), (tetrahedron(), TETRAHEDRON, 3, 1),
    ] {
        let owner = AuthenticatedCommonMesh::import_msh41(geometry.clone(), policy(4, 4096), source.to_vec()).unwrap();
        assert_eq!(owner.geometry(), Some(&geometry));
        let mesh = owner.simplicial_mesh().unwrap();
        assert_eq!(mesh.dimension(), dimension);
        assert_eq!(mesh.mesh().entity_count(0), Some(4));
        assert_eq!(mesh.mesh().entity_count(dimension), Some(cells));
        let expected_cells = if dimension == 2 { vec![vec![0,1,2], vec![0,2,3]] } else { vec![vec![0,1,2,3]] };
        assert_eq!(mesh.mesh().cells(), expected_cells);
        assert_eq!((0..mesh.mesh().entity_count(dimension-1).unwrap()).filter(|&i| mesh.mesh().is_boundary_entity(MeshEntity::new(dimension-1,i)) == Some(true)).count(), 4);
        assert_eq!(owner.production().unwrap().provider_identity(), "eqiora.external-msh");
        assert_eq!(owner.production().unwrap().provider_version(), "4.1");
        assert!(owner.production().unwrap().gmsh_mesh_policy().is_none());
        assert!(owner.gmsh_provider_output().is_none());
        let bytes = owner.to_bytes().unwrap();
        let replay = AuthenticatedCommonMesh::from_bytes(&bytes).unwrap();
        assert_eq!(owner, replay);
        assert_eq!(owner.digest().unwrap(), replay.digest().unwrap());
        let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        wire["resources"]["mesh"]["kind"] = "gmsh4152".into();
        assert!(AuthenticatedCommonMesh::from_bytes(&serde_json::to_vec(&wire).unwrap()).is_err());
    }
}

#[test]
fn external_msh_rejects_foreign_geometry_and_both_dimension_work_limits() {
    assert!(AuthenticatedCommonMesh::import_msh41(square(2.), policy(4,4096), SQUARE.to_vec()).is_err());
    for (geometry, source) in [(square(1.),SQUARE), (tetrahedron(),TETRAHEDRON)] {
        assert!(AuthenticatedCommonMesh::import_msh41(geometry.clone(), policy(3,4096), source.to_vec()).is_err());
        assert!(AuthenticatedCommonMesh::import_msh41(geometry, policy(4,source.len()-1), source.to_vec()).is_err());
    }
}
