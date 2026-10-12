use eqiora_meshing::{CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology, SimplicialMesh};

pub(crate) fn mesh_3d(quality: MeshQualityGate) -> SimplicialMesh {
    mesh_3d_with_upper_z(1.0, quality)
}

pub(crate) fn mesh_3d_with_upper_z(upper_z: f64, quality: MeshQualityGate) -> SimplicialMesh {
    let vertices = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, upper_z],
        vec![0.0, 0.0, upper_z],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, upper_z],
        vec![1.0, 0.0, upper_z],
        vec![2.0, 0.0, 0.0],
        vec![2.0, 1.0, 0.0],
        vec![2.0, 1.0, upper_z],
        vec![2.0, 0.0, upper_z],
        vec![0.5, 0.5, upper_z / 2.0],
        vec![1.0, 0.5, upper_z / 2.0],
        vec![1.5, 0.5, upper_z / 2.0],
    ];
    let interface = [[4, 5, 13], [5, 6, 13], [6, 7, 13], [7, 4, 13]];
    let fluid_surface = [
        [0, 3, 2],
        [0, 2, 1],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
    ];
    let solid_surface = [
        [8, 9, 10],
        [8, 10, 11],
        [4, 8, 11],
        [4, 11, 7],
        [5, 6, 10],
        [5, 10, 9],
        [4, 5, 9],
        [4, 9, 8],
        [7, 11, 10],
        [7, 10, 6],
    ];
    let mut cells = fluid_surface
        .into_iter()
        .chain(interface)
        .map(|face| vec![12, face[0], face[1], face[2]])
        .chain(
            solid_surface
                .into_iter()
                .chain(interface)
                .map(|face| vec![14, face[0], face[1], face[2]]),
        )
        .collect::<Vec<_>>();
    for cell in &mut cells {
        if signed_tetrahedron_measure(&vertices, cell) < 0.0 {
            cell.swap(1, 2);
        }
    }
    SimplicialMesh::new(3, vertices, cells, quality).unwrap()
}

pub(crate) fn inventories_3d(mesh: &SimplicialMesh) -> (Vec<CellId>, Vec<CellId>, Vec<FacetId>) {
    let fluid = (0..14).map(CellId::new).collect();
    let solid = (14..28).map(CellId::new).collect();
    let interface = (0..mesh.entity_count(2).unwrap())
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(2, facet))
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.0)
        })
        .map(FacetId::new)
        .collect();
    (fluid, solid, interface)
}

fn signed_tetrahedron_measure(vertices: &[Vec<f64>], cell: &[usize]) -> f64 {
    let origin = &vertices[cell[0]];
    let column = |vertex: usize, axis: usize| vertices[cell[vertex]][axis] - origin[axis];
    column(1, 0) * (column(2, 1) * column(3, 2) - column(3, 1) * column(2, 2))
        - column(2, 0) * (column(1, 1) * column(3, 2) - column(3, 1) * column(1, 2))
        + column(3, 0) * (column(1, 1) * column(2, 2) - column(2, 1) * column(1, 2))
}
