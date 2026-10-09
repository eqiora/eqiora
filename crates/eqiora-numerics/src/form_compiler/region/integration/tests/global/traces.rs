use super::*;

fn field_at(
    mesh: &SimplicialMesh,
    mapping: &RegionDofMap<f64>,
    space: Space,
    cell: usize,
    weights: &[f64; 3],
) -> [f64; 3] {
    let geometry = mesh.geometry_map(MeshEntity::new(3, cell)).unwrap();
    // Exact face barycentric coordinates avoid an inverse-map rounding probe
    // at the boundary; permutation changes only which local slots carry them.
    let vertex_weights = [0.0, weights[0], weights[1], weights[2], 0.0];
    let reference: [f64; 3] =
        std::array::from_fn(|axis| vertex_weights[mesh.cells()[cell][axis + 1]]);
    let element = DiscreteSpace::new(space, ReferenceCell::simplex(3).unwrap()).unwrap();
    let table = element.tabulate_on(&geometry, &reference).unwrap();
    let field = mapping.keys().next().unwrap().field;
    let keys = mapping.cell_field_keys(cell, field).unwrap();
    let signs = mapping.cell_signs(cell).unwrap();
    assert_eq!(keys.len(), element.local_dofs().len());
    std::array::from_fn(|axis| {
        keys.iter()
            .enumerate()
            .map(|(local, key)| {
                // Arbitrary global moment data, independent of any interpolated polynomial.
                let coefficient = (key.entity.index() + 1) as f64;
                f64::from(signs[local]) * coefficient * table.value(local).unwrap()[axis]
            })
            .sum()
    })
}

#[test]
fn tangential_edge_traces_and_normal_face_traces_are_shared() {
    for permuted in [false, true] {
        let mesh = mesh(permuted);
        for space in [Space::tetrahedral_edge(), Space::tetrahedral_face()] {
            let (domain, layouts) = layout::<f64>(space);
            let mapping = RegionDofMap::<f64>::new(
                &mesh,
                &layouts,
                ReferenceCell::simplex(3).unwrap(),
                &[domain; 2],
                &[],
                &BTreeMap::new(),
            )
            .unwrap();
            // Interior points of face 123; tangents (-2,3,0) and (-2,0,4),
            // oriented area vector (6,4,3). The face is deliberately non-unit.
            for weights in [[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]] {
                let first = field_at(&mesh, &mapping, space, 0, &weights);
                let second = field_at(&mesh, &mapping, space, 1, &weights);
                let directions = if space == Space::tetrahedral_edge() {
                    vec![[-2.0, 3.0, 0.0], [-2.0, 0.0, 4.0]]
                } else {
                    vec![[6.0, 4.0, 3.0]]
                };
                for direction in directions {
                    let a = (0..3).map(|axis| direction[axis] * first[axis]).sum();
                    let b = (0..3).map(|axis| direction[axis] * second[axis]).sum();
                    close(C::new(a, 0.0), C::new(b, 0.0));
                }
                // Stronger full-value continuity is not required or implied.
                assert!((0..3).any(|axis| (first[axis] - second[axis]).abs()
                    > 4096.0 * f64::EPSILON * first[axis].abs().max(second[axis].abs()).max(1.0)));
            }
        }
    }
}
