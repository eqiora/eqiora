use super::*;

// Independent incidence definition: a lower entity's vertex set is contained
// in the higher entity's vertex set. This exhaustive oracle does not use axis
// families, anchors, binary search or the production local-ordinal helper.
fn exhaustive(mesh: &CartesianMesh, source: MeshEntity, dimension: usize) -> Vec<EntityIncidence> {
    let source_vertices = mesh.entity_vertices(source).unwrap();
    let mut result = Vec::new();
    for index in 0..mesh.entity_count(dimension).unwrap() {
        let target = MeshEntity::new(dimension, index);
        let target_vertices = mesh.entity_vertices(target).unwrap();
        let (lower, higher) = if dimension < source.dimension() {
            (&target_vertices, &source_vertices)
        } else {
            (&source_vertices, &target_vertices)
        };
        if !lower.iter().all(|vertex| higher.contains(vertex)) {
            continue;
        }
        let mut closure = lower
            .iter()
            .map(|vertex| higher.iter().position(|v| v == vertex).unwrap())
            .collect::<Vec<_>>();
        closure.sort_unstable();
        let reference =
            ReferenceTopology::new(reference_cell(source.dimension().max(dimension)).unwrap())
                .unwrap();
        let lower_dimension = source.dimension().min(dimension);
        let ordinal = (0..reference.entity_count(lower_dimension).unwrap())
            .find(|&ordinal| {
                reference
                    .entity(lower_dimension, ordinal)
                    .unwrap()
                    .vertex_ordinals()
                    == closure
            })
            .unwrap();
        result.push(EntityIncidence {
            entity: target,
            local_ordinal: ordinal,
            orientation: OrientationCode::identity(),
        });
    }
    if dimension < source.dimension() {
        result.sort_by_key(|entry| entry.local_ordinal);
    }
    result
}

#[test]
fn every_rank_and_direction_matches_vertex_set_incidence_through_four_dimensions() {
    for counts in [vec![3], vec![2, 3], vec![2, 1, 2], vec![1, 2, 1, 2]] {
        let axes = counts
            .iter()
            .map(|&count| (0..=count).map(|i| (i * i + i) as f64).collect())
            .collect();
        let mesh = CartesianMesh::from_axes(axes).unwrap();
        for dimension in 0..=counts.len() {
            for index in 0..mesh.entity_count(dimension).unwrap() {
                let source = MeshEntity::new(dimension, index);
                for target_dimension in 0..=counts.len() {
                    assert_eq!(
                        mesh.incidence(source, target_dimension).unwrap(),
                        exhaustive(&mesh, source, target_dimension),
                        "counts={counts:?}, source={source:?}, target dimension={target_dimension}",
                    );
                }
            }
        }
        assert!(mesh.incidence(MeshEntity::new(0, usize::MAX), 1).is_none());
        assert!(
            mesh.incidence(MeshEntity::new(counts.len() + 1, 0), 0)
                .is_none()
        );
        assert!(
            mesh.incidence(MeshEntity::new(0, 0), counts.len() + 1)
                .is_none()
        );
    }
}

#[test]
fn interior_and_boundary_vertices_have_exact_neighbor_cells() {
    let mesh = CartesianMesh::uniform(&[[0., 2.], [0., 3.]], &[2, 3]).unwrap();
    let cells = |vertex| {
        mesh.incidence(MeshEntity::new(0, vertex), 2)
            .unwrap()
            .into_iter()
            .map(|entry| entry.entity.index())
            .collect::<Vec<_>>()
    };
    assert_eq!(cells(0), [0]);
    assert_eq!(cells(5), [0, 1, 3, 4]);
    assert_eq!(cells(11), [5]);
}
