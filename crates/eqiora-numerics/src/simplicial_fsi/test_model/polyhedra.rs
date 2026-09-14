//! Authored outer bodies of the retained three-dimensional numerical fixtures.

use super::*;

pub(crate) fn bipyramid_geometry() -> CanonicalGeometryV1 {
    named_geometry(
        vec![
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.],
            [0., 0., 1.],
            [0., 1., 0.],
            [1., 0., 0.],
        ],
        vec![
            vec![
                vec![1, 2, 4, 3],
                vec![0, 2, 1],
                vec![0, 4, 2],
                vec![0, 3, 4],
                vec![0, 1, 3],
            ],
            vec![
                vec![1, 3, 4, 2],
                vec![5, 1, 2],
                vec![5, 2, 4],
                vec![5, 4, 3],
                vec![5, 3, 1],
            ],
        ],
        [-1., 0., 0.],
    )
}

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

pub(crate) fn polyhedral_layout(
    geometry: &CanonicalGeometryV1,
    mesh: &eqiora_meshing::SimplicialMesh,
    partition: &super::super::FixedReferenceFsiPartition<3>,
    boundary: &super::super::FixedReferenceFsiBoundary<3>,
    config: FixedReferenceFsiStepConfig<3>,
    solver: SolverPlan,
    ale: bool,
) -> super::super::layout::FsiLayout<3> {
    use eqiora_artifact::{
        GeometryDefinitionV1, GeometryMeshCorrespondenceEnvelopeV1, SimplicialMeshEnvelopeV1,
    };
    let definition = GeometryDefinitionV1::from_canonical(geometry).unwrap();
    let artifact = SimplicialMeshEnvelopeV1::from_mesh(mesh).unwrap();
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_polyhedra(&definition, &artifact).unwrap();
    correspondence
        .validate_against_polyhedra(&definition, &artifact)
        .unwrap();
    for (name, cells) in [
        ("fluid", partition.fluid_cells()),
        ("solid", partition.solid_cells()),
    ] {
        let expected = correspondence
            .polyhedral_entity_set_entities(&definition, name)
            .unwrap()
            .into_iter()
            .map(|entity| entity.index())
            .collect::<BTreeSet<_>>();
        assert_eq!(expected, cells.iter().map(|cell| cell.index()).collect());
    }
    let expected = correspondence
        .polyhedral_entity_set_entities(&definition, "fluid_contact")
        .unwrap()
        .into_iter()
        .map(|entity| entity.index())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        expected,
        partition
            .interface_facets()
            .iter()
            .map(|facet| facet.index())
            .collect()
    );
    let model = authored_model(
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
    );
    super::super::layout::FsiLayout::bind(&model.program, &model.plan, mesh, partition, boundary)
        .unwrap()
}
