use super::*;

fn geometry() -> CanonicalGeometryV1 {
    let vertices = (0..8)
        .map(|i| {
            [
                2. * (i & 1) as f64,
                3. * ((i >> 1) & 1) as f64,
                4. * ((i >> 2) & 1) as f64,
            ]
        })
        .collect();
    CanonicalGeometryV1::from_convex_polyhedra(
        vertices,
        vec![vec![
            vec![0, 2, 3, 1],
            vec![4, 5, 7, 6],
            vec![0, 1, 5, 4],
            vec![2, 6, 7, 3],
            vec![0, 4, 6, 2],
            vec![1, 3, 7, 5],
        ]],
        vec![
            NamedEntitySet::new("body", 3, vec![0]),
            NamedEntitySet::new("outer", 2, (0..6).collect()),
        ],
        1e-12,
    )
    .unwrap()
}

// A bounded MSH fixture, not evidence that a Gmsh binary was executed.
fn observation(permuted: bool) -> Vec<u8> {
    let mut text = String::from(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 8 1 8\n3 1 0 8\n1\n2\n3\n4\n5\n6\n7\n8\n",
    );
    for i in 0..8 {
        text += &format!(
            "{} {} {}\n",
            2 * (i & 1),
            3 * ((i >> 1) & 1),
            4 * ((i >> 2) & 1)
        );
    }
    text += "$EndNodes\n$Elements\n1 6 1 6\n3 1 4 6\n";
    for (i, mut cell) in [
        [1, 2, 4, 8],
        [1, 4, 3, 8],
        [1, 3, 7, 8],
        [1, 7, 5, 8],
        [1, 5, 6, 8],
        [1, 6, 2, 8],
    ]
    .into_iter()
    .enumerate()
    {
        if permuted {
            cell.swap(0, 1);
            cell.swap(2, 3);
        }
        text += &format!(
            "{} {} {} {} {}\n",
            i + 1,
            cell[0],
            cell[1],
            cell[2],
            cell[3]
        );
    }
    text += "$EndElements\n";
    text.into_bytes()
}

fn policy(facets: usize) -> eqiora_artifact::GmshMeshPolicyV1 {
    eqiora_artifact::GmshMeshPolicyV1::explicit(1e-12, 0.01, facets, 5.).unwrap()
}

#[test]
fn polyhedral_common_mesh_replays_exact_tetrahedra_and_frontiers() {
    let geometry = geometry();
    let mut owners = Vec::new();
    for permuted in [false, true] {
        let output = observation(permuted);
        let owner =
            AuthenticatedCommonMesh::gmsh_4152(geometry.clone(), policy(12), output.clone())
                .unwrap();
        super::super::super::native::validate_simplicial_resources(&owner.resources).unwrap();
        let mesh = owner.simplicial_mesh().unwrap();
        assert_eq!(mesh.dimension(), 3);
        assert_eq!(
            (
                mesh.mesh().entity_count(1),
                mesh.mesh().entity_count(2),
                mesh.mesh().entity_count(3)
            ),
            (Some(19), Some(18), Some(6))
        );
        let definition = eqiora_artifact::GeometryDefinitionV1::from_canonical(&geometry).unwrap();
        let correspondence = owner.correspondence().unwrap();
        correspondence
            .validate_against_polyhedra(&definition, mesh)
            .unwrap();
        let cells = correspondence
            .polyhedral_entity_set_entities(&definition, "body")
            .unwrap();
        assert_eq!(
            cells,
            (0..6).map(|i| MeshEntity::new(3, i)).collect::<Vec<_>>()
        );
        let outer = correspondence
            .polyhedral_entity_set_entities(&definition, "outer")
            .unwrap();
        assert_eq!(outer.len(), 12);
        assert!(
            outer
                .iter()
                .all(|&facet| mesh.mesh().is_boundary_entity(facet) == Some(true))
        );
        assert_eq!(owner.gmsh_provider_output(), Some(output.as_slice()));
        let bytes = owner.to_bytes().unwrap();
        let replay = AuthenticatedCommonMesh::from_bytes(&bytes).unwrap();
        assert_eq!(replay, owner);
        assert_eq!(replay.digest().unwrap(), owner.digest().unwrap());
        owners.push(owner);
    }
    assert_ne!(owners[0].digest().unwrap(), owners[1].digest().unwrap());
    // Resource and raw observation identities are independently retained.
    for field in [
        "mesh_base64",
        "correspondence_base64",
        "provider_output_base64",
        "production_base64",
    ] {
        let wire: serde_json::Value =
            serde_json::from_slice(&owners[0].to_bytes().unwrap()).unwrap();
        let other: serde_json::Value =
            serde_json::from_slice(&owners[1].to_bytes().unwrap()).unwrap();
        let original = String::from_utf8(owners[0].to_bytes().unwrap()).unwrap();
        let before = format!("\"{field}\":{}", wire["resources"]["mesh"][field]);
        let after = format!("\"{field}\":{}", other["resources"]["mesh"][field]);
        assert_ne!(before, after);
        let changed = original.replace(&before, &after);
        assert_ne!(changed, original);
        assert!(
            AuthenticatedCommonMesh::from_bytes(changed.as_bytes()).is_err(),
            "accepted stale {field}"
        );
    }
    let scales = IncompressibleFlowScaleProfile2d::new(
        DynQuantity::new(
            1.,
            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
        ),
        DynQuantity::new(
            1.,
            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap(),
        ),
        DynQuantity::new(
            1.,
            DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).unwrap(),
        ),
    )
    .unwrap();
    for spatial in [
        NativeSpatialPolicy::StokesMiniP1(scales),
        NativeSpatialPolicy::TransientMiniP1(scales),
    ] {
        let error = super::super::super::native::validate_resources(spatial, &owners[0].resources)
            .unwrap_err();
        assert!(error.message().contains("two-dimensional Geometry"));
    }
}

#[test]
fn polyhedral_mesh_rejects_outside_cells_missing_coverage_and_excess_facets() {
    let geometry = geometry();
    let original = String::from_utf8(observation(false)).unwrap();
    let outside = original.replace("2 3 4\n$EndNodes", "2 3 5\n$EndNodes");
    assert!(
        AuthenticatedCommonMesh::gmsh_4152(geometry.clone(), policy(12), outside.into_bytes())
            .is_err()
    );
    let missing = original
        .replace("1 6 1 6\n3 1 4 6", "1 5 1 5\n3 1 4 5")
        .replace("6 1 6 2 8\n", "");
    assert!(
        AuthenticatedCommonMesh::gmsh_4152(geometry.clone(), policy(20), missing.into_bytes())
            .is_err()
    );
    let error =
        AuthenticatedCommonMesh::gmsh_4152(geometry, policy(8), observation(false)).unwrap_err();
    assert!(error.message().contains("boundary facet budget"));
}
