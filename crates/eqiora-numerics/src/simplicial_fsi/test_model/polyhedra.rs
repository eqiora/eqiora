//! Authored outer bodies of the retained three-dimensional numerical fixtures.

use super::*;

pub(crate) fn tetrahedral_geometry() -> CanonicalGeometryV1 {
    named_geometry(
        vec![
            [-1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [1., 0., 0.],
        ],
        vec![
            vec![vec![1, 3, 2], vec![0, 3, 1], vec![0, 2, 3], vec![0, 1, 2]],
            vec![vec![1, 2, 3], vec![4, 1, 3], vec![4, 3, 2], vec![4, 2, 1]],
        ],
        [-1., 0., 0.],
    )
}

pub(crate) fn z_tetrahedral_geometry() -> CanonicalGeometryV1 {
    // Fluid above the unit xy triangle, solid below it. This is the element
    // fixture's exact authored shell, with no extra refinement vertices.
    named_geometry(
        vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
        ],
        vec![
            vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![2, 0, 3]],
            vec![vec![0, 1, 2], vec![0, 4, 1], vec![1, 4, 2], vec![2, 4, 0]],
        ],
        [0., 0., 1.],
    )
}

fn named_geometry(
    vertices: Vec<[f64; 3]>,
    shells: Vec<Vec<Vec<usize>>>,
    fluid_apex: [f64; 3],
) -> CanonicalGeometryV1 {
    let geometry =
        CanonicalGeometryV1::from_convex_polyhedra(vertices.clone(), shells.clone(), vec![], 1e-12)
            .unwrap();
    let canonical_vertices = geometry.polyhedral_vertices().unwrap();
    let fluid = (0..2)
        .find(|&volume| {
            geometry
                .polyhedral_volume_facets(volume)
                .unwrap()
                .iter()
                .any(|&facet| {
                    geometry
                        .polyhedral_outward_facet(facet, volume)
                        .unwrap()
                        .iter()
                        .any(|&vertex| canonical_vertices[vertex] == fluid_apex)
                })
        })
        .unwrap();
    let solid = 1 - fluid;
    let facets = [fluid, solid].map(|volume| {
        geometry
            .polyhedral_volume_facets(volume)
            .unwrap()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
    });
    let common = facets[0]
        .intersection(&facets[1])
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(common.len(), 1);
    CanonicalGeometryV1::from_convex_polyhedra(
        vertices,
        shells,
        vec![
            NamedEntitySet::new("fluid", 3, vec![fluid]),
            NamedEntitySet::new("solid", 3, vec![solid]),
            NamedEntitySet::new("fluid_contact", 2, common.clone()),
            NamedEntitySet::new("solid_contact", 2, common),
            NamedEntitySet::new(
                "fluid_outer",
                2,
                facets[0].difference(&facets[1]).copied().collect(),
            ),
            NamedEntitySet::new(
                "solid_outer",
                2,
                facets[1].difference(&facets[0]).copied().collect(),
            ),
        ],
        1e-12,
    )
    .unwrap()
}

pub(crate) fn polyhedral_model(
    geometry: &CanonicalGeometryV1,
    mesh: &eqiora_meshing::SimplicialMesh,
    config: FixedReferenceFsiStepConfig<3>,
    solver: SolverPlan,
    ale: bool,
) -> super::AuthoredFsiModel {
    use eqiora_artifact::SimplicialMeshEnvelopeV1;
    let artifact = SimplicialMeshEnvelopeV1::from_mesh(mesh).unwrap();
    authored_model(
        geometry,
        [
            [
                geometry.entity_set("fluid").unwrap(),
                geometry.entity_set("solid").unwrap(),
            ],
            [
                geometry.entity_set("fluid_outer").unwrap(),
                geometry.entity_set("solid_outer").unwrap(),
            ],
            [
                geometry.entity_set("fluid_contact").unwrap(),
                geometry.entity_set("solid_contact").unwrap(),
            ],
        ],
        MeshArtifactReference::from_sha256(artifact.digest().unwrap().sha256_bytes()),
        config,
        solver,
        ale,
    )
}

/// The exact outer shells of the shared 15-vertex, 28-tetrahedron fixture.
pub(crate) fn adjacent_boxes() -> CanonicalGeometryV1 {
    let vertices = (0..3)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| [x as f64, y as f64, z as f64])))
        .collect();
    let shells = (0..2)
        .map(|x| {
            let indices = (0..8)
                .map(|i| (x + (i & 1)) * 4 + ((i >> 1) & 1) * 2 + ((i >> 2) & 1))
                .collect::<Vec<_>>();
            [
                [0, 2, 3, 1],
                [4, 5, 7, 6],
                [0, 1, 5, 4],
                [2, 6, 7, 3],
                [0, 4, 6, 2],
                [1, 3, 7, 5],
            ]
            .map(|face| face.map(|i| indices[i]).to_vec())
            .to_vec()
        })
        .collect();
    named_geometry(vertices, shells, [0., 0., 0.])
}
