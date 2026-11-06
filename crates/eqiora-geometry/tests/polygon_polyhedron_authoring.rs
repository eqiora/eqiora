//! Analytic geometry expectations frozen before graph/Python implementation.
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};

fn skew_polygon() -> Vec<[f64; 2]> {
    vec![[0.25, -0.5], [1.375, 1.0], [0.0625, 5.5], [-1.0625, 4.0]]
}

fn skew_polyhedron() -> (Vec<[f64; 3]>, Vec<Vec<usize>>) {
    let vertices = (0..8)
        .map(|n| {
            let q = [
                f64::from(n & 1),
                f64::from((n >> 1) & 1),
                f64::from((n >> 2) & 1),
            ];
            [
                0.25 + 15. / 16. * q[0] - 3. / 32. * q[1] + 11. / 64. * q[2],
                -0.5 + 5. / 4. * q[0] + 3. * q[1] - 51. / 32. * q[2],
                0.75 + 5. / 2. * q[1] + 15. / 4. * q[2],
            ]
        })
        .collect();
    (
        vertices,
        vec![
            vec![0, 4, 6, 2],
            vec![1, 3, 7, 5],
            vec![0, 1, 5, 4],
            vec![2, 6, 7, 3],
            vec![0, 2, 3, 1],
            vec![4, 5, 7, 6],
        ],
    )
}

#[test]
fn analytic_skew_domains_use_existing_exact_geometry_families() {
    let v = skew_polygon();
    let twice_area: f64 = (0..4)
        .map(|i| v[i][0] * v[(i + 1) % 4][1] - v[(i + 1) % 4][0] * v[i][1])
        .sum();
    assert_eq!(twice_area / 2., 225. / 32.);
    let region = PlanarRegion::new(
        v,
        vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
        vec![NamedEntitySet::new("domain", 2, vec![0])],
        1e-12,
    )
    .unwrap();
    assert_eq!(region.vertices().len(), 4);
    let (v, f) = skew_polyhedron();
    let a = [15. / 16., 5. / 4., 0.];
    let b = [-3. / 32., 3., 5. / 2.];
    let c = [11. / 64., -51. / 32., 15. / 4.];
    let determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1]);
    assert_eq!(determinant, 15625. / 1024.);
    let geometry = CanonicalGeometryV1::from_convex_polyhedra(
        v,
        vec![f],
        vec![NamedEntitySet::new("domain", 3, vec![0])],
        1e-12,
    )
    .unwrap();
    assert_eq!(geometry.polyhedral_vertices().unwrap().len(), 8);
}

fn named(
    graph: &eqiora_geometry::GeometryGraph,
    op: &eqiora_geometry::GeometryOperation,
) -> CanonicalGeometryV1 {
    let mut names = std::collections::BTreeMap::new();
    names.insert("domain".into(), vec![op.region().into()]);
    for (i, h) in op.boundaries().into_iter().enumerate() {
        names.insert(format!("boundary_{i}"), vec![h.into()]);
    }
    graph.build(op, &names).unwrap()
}

#[test]
fn polygon_windings_keep_named_physical_edges_and_bounds() {
    use eqiora_geometry::GeometryGraph;
    let graph = GeometryGraph::new();
    let op = graph.polygon(skew_polygon(), 1e-12).unwrap();
    let original = named(&graph, &op);
    assert_eq!(
        original.bounds_m(),
        vec![[-17. / 16., 11. / 8.], [-0.5, 5.5]]
    );
    let v = skew_polygon();
    // Reverse winding while retaining the initial vertex. Physical edge i now
    // occupies construction position 3-i; names must follow that physical edge.
    let reversed = graph.polygon(vec![v[0], v[3], v[2], v[1]], 1e-12).unwrap();
    let boundaries = reversed.boundaries();
    let mut names = std::collections::BTreeMap::new();
    names.insert("domain".into(), vec![reversed.region().into()]);
    for i in 0..4 {
        names.insert(format!("boundary_{i}"), vec![boundaries[3 - i].into()]);
    }
    assert_eq!(original, graph.build(&reversed, &names).unwrap());
    assert_eq!(op.region().dimension(), 2);
    assert!(op.boundaries().iter().all(|b| b.dimension() == 1));
}

#[test]
fn polyhedron_permutations_keep_named_faces_and_bounds() {
    use eqiora_geometry::GeometryGraph;
    let graph = GeometryGraph::new();
    let (v, f) = skew_polyhedron();
    let op = graph
        .convex_polyhedron(v.clone(), f.clone(), 1e-12)
        .unwrap();
    let original = named(&graph, &op);
    assert_eq!(
        original.bounds_m(),
        vec![[5. / 32., 87. / 64.], [-67. / 32., 15. / 4.], [3. / 4., 7.]]
    );
    let v2 = v.iter().rev().copied().collect();
    let f2 = f
        .iter()
        .rev()
        .map(|face| {
            let mut loop_ = face.iter().map(|i| 7 - i).collect::<Vec<_>>();
            loop_.rotate_left(1);
            loop_
        })
        .collect();
    let reordered = graph.convex_polyhedron(v2, f2, 1e-12).unwrap();
    let boundaries = reordered.boundaries();
    let mut names = std::collections::BTreeMap::new();
    names.insert("domain".into(), vec![reordered.region().into()]);
    for i in 0..6 {
        names.insert(format!("boundary_{i}"), vec![boundaries[5 - i].into()]);
    }
    assert_eq!(original, graph.build(&reordered, &names).unwrap());
    assert_eq!(op.region().dimension(), 3);
    assert!(op.boundaries().iter().all(|b| b.dimension() == 2));
    let decoded =
        CanonicalGeometryV1::replay_canonical(original.canonical_bytes(), Default::default())
            .unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn naming_rejects_foreign_stale_incomplete_and_overlapping_handles() {
    use eqiora_geometry::GeometryGraph;
    let g = GeometryGraph::new();
    let foreign = GeometryGraph::new();
    let op = g.polygon(skew_polygon(), 1e-12).unwrap();
    let same = g.polygon(skew_polygon(), 1e-12).unwrap();
    let other = foreign.polygon(skew_polygon(), 1e-12).unwrap();
    for source in [&same, &other] {
        let names =
            std::collections::BTreeMap::from([("bad".into(), vec![source.region().into()])]);
        assert!(g.build(&op, &names).is_err());
    }
    let mut names = std::collections::BTreeMap::new();
    names.insert("domain".into(), vec![op.region().into()]);
    assert!(g.build(&op, &names).is_err());
    for (i, b) in op.boundaries().into_iter().enumerate() {
        names.insert(format!("edge_{i}"), vec![b.into()]);
    }
    assert!(g.build(&op, &names).is_ok());
    names.insert("overlap".into(), vec![op.boundaries()[0].into()]);
    assert!(g.build(&op, &names).is_err());
    assert!(foreign.build(&op, &names).is_err());
}

#[test]
fn invalid_polygons_and_convex_shells_reject_before_handles() {
    use eqiora_geometry::GeometryGraph;
    let g = GeometryGraph::new();
    for vertices in [
        vec![],
        vec![[0., 0.], [1., 0.], [2., 0.]],
        vec![[0., 0.], [1., 1.], [0., 1.], [1., 0.]],
        vec![[f64::NAN, 0.], [1., 0.], [0., 1.]],
        vec![[0., 0.], [1., 0.], [0., 0.]],
    ] {
        assert!(g.polygon(vertices, 1e-12).is_err());
    }
    assert!(g.polygon(vec![[0., 0.]; 4097], 1e-12).is_err());
    for tolerance in [0., -1., f64::INFINITY, f64::NAN] {
        assert!(g.polygon(skew_polygon(), tolerance).is_err());
        let (v, f) = skew_polyhedron();
        assert!(g.convex_polyhedron(v, f, tolerance).is_err());
    }
    let (v, f) = skew_polyhedron();
    let mut open = f.clone();
    open.pop();
    assert!(g.convex_polyhedron(v.clone(), open, 1e-12).is_err());
    let reversed = f
        .iter()
        .map(|face| face.iter().rev().copied().collect())
        .collect();
    assert!(g.convex_polyhedron(v.clone(), reversed, 1e-12).is_err());
    for kind in 0..5 {
        let mut vertices = v.clone();
        let mut faces = f.clone();
        match kind {
            0 => vertices[7][0] += 0.125,   // nonplanar quadrilateral
            1 => vertices[7] = vertices[0], // collapsed shell
            2 => vertices[7][0] = f64::INFINITY,
            3 => faces[0][0] = 99,
            _ => {
                let first = faces[0][0];
                faces[0].push(first);
            }
        }
        assert!(g.convex_polyhedron(vertices, faces, 1e-12).is_err());
    }
    // A closed triangulated cube with its far corner pushed into the interior
    // is concave, so planarity alone cannot cause this rejection.
    let mut cube = (0..8)
        .map(|n| {
            [
                f64::from(n & 1),
                f64::from((n >> 1) & 1),
                f64::from((n >> 2) & 1),
            ]
        })
        .collect::<Vec<_>>();
    cube[7] = [0.25, 0.25, 0.25];
    let triangles = f
        .iter()
        .flat_map(|q| [vec![q[0], q[1], q[2]], vec![q[0], q[2], q[3]]])
        .collect();
    assert!(g.convex_polyhedron(cube, triangles, 1e-12).is_err());
}
