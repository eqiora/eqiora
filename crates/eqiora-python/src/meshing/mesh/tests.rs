use std::collections::BTreeMap;

use eqiora::artifact::{AffineTriangleMeshCellsV1, GeometryMeshCorrespondenceEnvelopeV1};
use eqiora::geometry::{CanonicalGeometryV1, GeometryGraph};

use super::*;

fn rectangle(xmax: f64) -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, xmax], [-1.0, 2.0]).unwrap();
    let edges = rectangle.boundaries();
    graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("region".to_owned(), vec![rectangle.region().into()]),
                ("left".to_owned(), vec![edges[0].into()]),
                ("right".to_owned(), vec![edges[1].into()]),
                ("bottom".to_owned(), vec![edges[2].into()]),
                ("top".to_owned(), vec![edges[3].into()]),
            ]),
        )
        .unwrap()
}

#[test]
fn affine_triangle_native_resource_view_reauthenticates_every_bound_resource() {
    Python::initialize();
    Python::attach(|py| {
        let source = rectangle(2.0);
        let policy = AffineTriangleMeshCellsV1::new([2, 3]).unwrap();
        let (mesh, correspondence) =
            GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
                &source,
                policy.cells(),
            )
            .unwrap();
        let production =
            MeshProductionLineageEnvelopeV1::from_affine_triangle_rectangle_v1_resources(
                policy,
                &source,
                &mesh,
                &correspondence,
            )
            .unwrap();
        let publish = || {
            PyMesh::from_source_owned_affine_triangle(
                py,
                &source,
                &mesh,
                &correspondence,
                &production,
            )
            .unwrap()
        };
        let published = publish();
        let authenticated = published
            .authenticated_affine_triangle_resources()
            .unwrap()
            .unwrap();
        assert_eq!(authenticated.geometry, &source);
        assert_eq!(authenticated.mesh, &mesh);
        assert_eq!(authenticated.correspondence, &correspondence);
        assert_eq!(authenticated.production, &production);
        assert!(published.authenticated_common_mesh().unwrap().is_some());

        let foreign_geometry = rectangle(3.0);
        let mut geometry_crosswire = publish();
        let AcceptedMeshSource::SourceOwned { geometry, .. } = &mut geometry_crosswire.source
        else {
            unreachable!()
        };
        **geometry = foreign_geometry;
        assert!(
            geometry_crosswire
                .authenticated_affine_triangle_resources()
                .is_err()
        );
        assert!(geometry_crosswire.authenticated_common_mesh().is_err());

        let alternate_policy = AffineTriangleMeshCellsV1::new([3, 2]).unwrap();
        let (alternate_mesh, alternate_correspondence) =
            GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
                &source,
                alternate_policy.cells(),
            )
            .unwrap();
        let alternate_production =
            MeshProductionLineageEnvelopeV1::from_affine_triangle_rectangle_v1_resources(
                alternate_policy,
                &source,
                &alternate_mesh,
                &alternate_correspondence,
            )
            .unwrap();

        let mut mesh_crosswire = publish();
        let AcceptedMeshSource::SourceOwned { mesh, .. } = &mut mesh_crosswire.source else {
            unreachable!()
        };
        **mesh = alternate_mesh;
        assert!(
            mesh_crosswire
                .authenticated_affine_triangle_resources()
                .is_err()
        );
        assert!(mesh_crosswire.authenticated_common_mesh().is_err());

        let mut correspondence_crosswire = publish();
        let AcceptedMeshSource::SourceOwned { correspondence, .. } =
            &mut correspondence_crosswire.source
        else {
            unreachable!()
        };
        **correspondence = alternate_correspondence;
        assert!(
            correspondence_crosswire
                .authenticated_affine_triangle_resources()
                .is_err()
        );
        assert!(
            correspondence_crosswire
                .authenticated_common_mesh()
                .is_err()
        );

        let mut production_crosswire = publish();
        let AcceptedMeshSource::SourceOwned { production, .. } = &mut production_crosswire.source
        else {
            unreachable!()
        };
        **production = alternate_production;
        assert!(
            production_crosswire
                .authenticated_affine_triangle_resources()
                .is_err()
        );
        assert!(production_crosswire.authenticated_common_mesh().is_err());
    });
}
