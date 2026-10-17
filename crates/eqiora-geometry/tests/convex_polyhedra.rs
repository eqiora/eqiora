//! Independent outer-shell fixtures: no tetrahedralization or mesh identities.

use eqiora_geometry::{CanonicalGeometryLimits, CanonicalGeometryV1, NamedEntitySet};

type Shells = Vec<Vec<Vec<usize>>>;
fn bipyramid() -> (Vec<[f64; 3]>, Shells) {
    (
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
    )
}
fn tetrahedra() -> (Vec<[f64; 3]>, Shells) {
    (
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
    )
}
fn names(interface: usize) -> Vec<NamedEntitySet> {
    vec![
        NamedEntitySet::new("left", 3, vec![0]),
        NamedEntitySet::new("right", 3, vec![1]),
        NamedEntitySet::new("both", 3, vec![0, 1]),
        NamedEntitySet::new("interface", 2, vec![interface]),
        NamedEntitySet::new("exterior", 2, vec![0]),
        NamedEntitySet::new("vertex", 0, vec![0]),
        NamedEntitySet::new("edge", 1, vec![0]),
    ]
}
fn admit(vertices: Vec<[f64; 3]>, shells: Shells, interface: usize) -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_convex_polyhedra(vertices, shells, names(interface), 1e-12).unwrap()
}

#[test]
fn exact_bipyramid_and_tetrahedral_outer_shells_replay_with_opposite_parent_incidence() {
    for (fixture, interface, facet_count, vertex_count) in
        [(bipyramid(), 4, 5, 6), (tetrahedra(), 3, 4, 5)]
    {
        let geometry = admit(fixture.0, fixture.1, interface);
        assert_eq!(geometry.ambient_dimension(), 3);
        assert_eq!(geometry.topological_dimension(), 3);
        assert_eq!(geometry.polyhedral_vertices().unwrap().len(), vertex_count);
        assert_eq!(
            geometry.polyhedral_volume_facets(0).unwrap().len(),
            facet_count
        );
        assert_eq!(
            geometry.polyhedral_volume_facets(1).unwrap().len(),
            facet_count
        );
        assert!(geometry.selections_form_opposite_parent_interface(
            "interface",
            "left",
            "interface",
            "right"
        ));
        assert!(geometry.selections_form_opposite_parent_interface(
            "interface",
            "right",
            "interface",
            "left"
        ));
        for (a, b, c, d) in [
            ("interface", "left", "interface", "left"),
            ("interface", "both", "interface", "right"),
            ("exterior", "left", "interface", "right"),
        ] {
            assert!(!geometry.selections_form_opposite_parent_interface(a, b, c, d));
        }
        let boundary = geometry.entity_set("interface").unwrap();
        assert!(geometry.selection_is_boundary_of(boundary, geometry.entity_set("left").unwrap()));
        assert!(!geometry.selection_is_boundary_of(boundary, geometry.entity_set("both").unwrap()));
        assert!(
            geometry
                .cartesian_boundary_embedding(boundary, geometry.entity_set("left").unwrap())
                .is_none()
        );
        let left = geometry.polyhedral_outward_facet(interface, 0).unwrap();
        let right = geometry.polyhedral_outward_facet(interface, 1).unwrap();
        assert_eq!(left[0], right[0]);
        assert_eq!(
            left[1..].iter().rev().copied().collect::<Vec<_>>(),
            right[1..]
        );
        // These interfaces lie in x=0. The yz shoelace sign independently
        // determines the parent-outward x direction, not implementation output.
        for (loop_, sign) in [(left, 1.), (right, -1.)] {
            let vertices = geometry.polyhedral_vertices().unwrap();
            let twice_area = loop_
                .iter()
                .zip(loop_.iter().cycle().skip(1))
                .take(loop_.len())
                .map(|(&a, &b)| vertices[a][1] * vertices[b][2] - vertices[b][1] * vertices[a][2])
                .sum::<f64>();
            assert_eq!(twice_area.signum(), sign);
        }
        let replay = CanonicalGeometryV1::replay_canonical(
            geometry.canonical_bytes(),
            CanonicalGeometryLimits::default(),
        )
        .unwrap();
        assert_eq!(replay, geometry);
        assert_eq!(replay.digest_bytes(), geometry.digest_bytes());
        assert!(!geometry.selection_is_boundary_of(
            replay.entity_set("interface").unwrap(),
            geometry.entity_set("left").unwrap()
        ));
        assert!(!geometry.selection_is_boundary_of(boundary, replay.entity_set("left").unwrap()));
    }
}

#[test]
fn canonical_identity_ignores_vertex_shell_and_loop_start_order() {
    let (vertices, shells) = bipyramid();
    let expected = admit(vertices.clone(), shells.clone(), 4);
    let reversed_vertices = vertices.iter().rev().copied().collect::<Vec<_>>();
    let mut reordered = shells.into_iter().rev().collect::<Vec<_>>();
    for shell in &mut reordered {
        shell.reverse();
        for polygon in shell {
            for id in polygon.iter_mut() {
                *id = vertices.len() - 1 - *id;
            }
            polygon.rotate_left(1);
        }
    }
    let actual = admit(reversed_vertices, reordered, 4);
    assert_eq!(actual.canonical_bytes(), expected.canonical_bytes());
    let (mut vertices, shells) = bipyramid();
    vertices[0][0] = -2.;
    let changed = admit(vertices, shells, 4);
    assert_ne!(changed.digest_bytes(), expected.digest_bytes());
}

#[test]
fn shell_and_membership_mutants_fail_at_geometry_owner() {
    let (vertices, shells) = bipyramid();
    let mut missing = shells.clone();
    missing[0].pop();
    let mut reversed = shells.clone();
    reversed[0][0].reverse();
    let mut duplicate = shells.clone();
    let extra = duplicate[0][0].clone();
    duplicate[0].push(extra);
    let mut absent = shells.clone();
    absent[0][0][0] = 100;
    let mut crossed = shells.clone();
    crossed[0][0].swap(1, 2);
    let mut collapsed = vertices.clone();
    collapsed[0][0] = 0.;
    let mut nonplanar = vertices.clone();
    nonplanar[1][0] = 0.1;
    let mut inward = vertices.clone();
    inward[4][1] = -0.5;
    for (v, s, word) in [
        (vertices.clone(), missing, "shell"),
        (vertices.clone(), reversed, "orientation"),
        (vertices.clone(), duplicate, "nonmanifold"),
        (vertices.clone(), absent, "absent"),
        (vertices.clone(), crossed, "facet"),
        (collapsed, shells.clone(), "degenerate"),
        (nonplanar, shells.clone(), "planar"),
        (inward, shells.clone(), "nonconvex"),
    ] {
        let error = CanonicalGeometryV1::from_convex_polyhedra(v, s, vec![], 1e-12).unwrap_err();
        assert!(error.message().contains(word), "{word}: {error:?}");
    }
    for set in [
        NamedEntitySet::new("bad", 4, vec![0]),
        NamedEntitySet::new("bad", 2, vec![999]),
        NamedEntitySet::new("bad", 3, vec![0, 0]),
        NamedEntitySet::new("bad", 3, vec![]),
    ] {
        assert!(
            CanonicalGeometryV1::from_convex_polyhedra(
                vertices.clone(),
                shells.clone(),
                vec![set],
                1e-12
            )
            .is_err()
        );
    }
    for precision in [0., -1., f64::NAN, f64::INFINITY, 10.] {
        assert!(
            CanonicalGeometryV1::from_convex_polyhedra(
                vertices.clone(),
                shells.clone(),
                vec![],
                precision
            )
            .is_err()
        );
    }
    let mut nonfinite = vertices.clone();
    nonfinite[0][0] = f64::INFINITY;
    assert!(
        CanonicalGeometryV1::from_convex_polyhedra(nonfinite, shells.clone(), vec![], 1e-12)
            .is_err()
    );
    let mut isolated = vertices;
    isolated.push([9., 9., 9.]);
    assert!(CanonicalGeometryV1::from_convex_polyhedra(isolated, shells, vec![], 1e-12).is_err());
}

fn prism(bounds: [[f64; 2]; 3]) -> (Vec<[f64; 3]>, Vec<Vec<usize>>) {
    let vertices = (0..8)
        .map(|i| [bounds[0][i / 4], bounds[1][i / 2 % 2], bounds[2][i % 2]])
        .collect();
    (
        vertices,
        vec![
            vec![0, 1, 3, 2],
            vec![4, 6, 7, 5],
            vec![0, 4, 5, 1],
            vec![2, 3, 7, 6],
            vec![0, 2, 6, 4],
            vec![1, 5, 7, 3],
        ],
    )
}
fn pair_prisms(
    a: [[f64; 2]; 3],
    b: [[f64; 2]; 3],
) -> Result<CanonicalGeometryV1, eqiora_core::Diagnostic> {
    let (mut av, af) = prism(a);
    let (bv, mut bf) = prism(b);
    for id in bf.iter_mut().flatten() {
        *id += av.len();
    }
    av.extend(bv);
    CanonicalGeometryV1::from_convex_polyhedra(av, vec![af, bf], vec![], 1e-12)
}
#[test]
fn crossing_thin_prism_interiors_reject_without_vertex_containment() {
    // Horizontal and vertical bars overlap in a central 0.2-cube. Every
    // vertex of either lies strictly outside the other along its long axis.
    let a = [[-2., 2.], [-0.1, 0.1], [-0.1, 0.1]];
    let b = [[-0.1, 0.1], [-2., 2.], [-0.1, 0.1]];
    let error = pair_prisms(a, b).unwrap_err();
    assert!(error.message().contains("interiors overlap"), "{error:?}");
    assert!(pair_prisms(a, [[3., 4.], [-2., 2.], [-0.1, 0.1]]).is_ok());
    let containment = pair_prisms([[-2., 2.]; 3], [[-1., 1.]; 3]).unwrap_err();
    assert!(containment.message().contains("interiors overlap"));
}

#[test]
fn decoder_revalidates_orientation_membership_canonical_order_and_budgets() {
    let (v, s) = bipyramid();
    let geometry = admit(v, s, 4);
    let canonical: serde_json::Value = serde_json::from_slice(geometry.canonical_bytes()).unwrap();
    for mutation in 0..5 {
        let mut wire = canonical.clone();
        match mutation {
            0 => {
                let orientation = wire["volumes"][0][0]["orientation"].as_str().unwrap();
                wire["volumes"][0][0]["orientation"] =
                    serde_json::json!(if orientation == "forward" {
                        "reverse"
                    } else {
                        "forward"
                    });
            }
            1 => wire["volumes"][0][0]["facet"] = serde_json::json!(999),
            2 => wire["facets"][0].as_array_mut().unwrap().rotate_left(1),
            3 => wire["schema"] = serde_json::json!("unknown"),
            _ => wire["unexpected"] = serde_json::json!(true),
        }
        let error = CanonicalGeometryV1::replay_canonical(
            &serde_json::to_vec(&wire).unwrap(),
            CanonicalGeometryLimits::default(),
        )
        .unwrap_err();
        let intended = [
            "orientation",
            "absent facet",
            "not canonical",
            "unknown",
            "unknown field",
        ][mutation];
        assert!(error.message().contains(intended), "{intended}: {error:?}");
    }
    for limits in [
        CanonicalGeometryLimits {
            max_bytes: 1,
            ..Default::default()
        },
        CanonicalGeometryLimits {
            max_vertices: 5,
            ..Default::default()
        },
        CanonicalGeometryLimits {
            max_faces: 1,
            ..Default::default()
        },
        CanonicalGeometryLimits {
            max_loop_indices: 30,
            ..Default::default()
        },
        CanonicalGeometryLimits {
            max_entity_sets: 1,
            ..Default::default()
        },
        CanonicalGeometryLimits {
            max_entity_set_members: 1,
            ..Default::default()
        },
    ] {
        assert!(CanonicalGeometryV1::replay_canonical(geometry.canonical_bytes(), limits).is_err());
    }
}

#[test]
fn disjoint_skew_prisms_use_edge_cross_edge_separation() {
    // Both rods cross in xy projection and overlap on every facet-normal
    // projection. Their z slabs are disjoint: |z| <= 1/4 versus
    // |z-1| <= 1/4. The long x/y edge cross supplies the z axis.
    let (mut a, af) = prism([[-2., 2.], [-0.125, 0.125], [-0.125, 0.125]]);
    let (mut b, mut bf) = prism([[-0.125, 0.125], [-2., 2.], [-0.125, 0.125]]);
    let r = 1.;
    for p in &mut a {
        *p = [p[0], (p[1] - p[2]) * r, (p[1] + p[2]) * r];
    }
    for p in &mut b {
        *p = [(p[0] + p[2]) * r, p[1], 1. + (p[2] - p[0]) * r];
    }
    for id in bf.iter_mut().flatten() {
        *id += a.len();
    }
    a.extend(b);
    CanonicalGeometryV1::from_convex_polyhedra(a, vec![af, bf], vec![], 1e-12).unwrap();
}

#[test]
fn outward_shells_enclose_independently_derived_pyramid_and_tetrahedron_volumes() {
    for (fixture, interface, expected) in [(bipyramid(), 4, 2. / 3.), (tetrahedra(), 3, 1. / 6.)] {
        let geometry = admit(fixture.0, fixture.1, interface);
        let points = geometry.polyhedral_vertices().unwrap();
        for volume in 0..2 {
            let mut determinant_sum = 0.;
            for &facet in geometry.polyhedral_volume_facets(volume).unwrap() {
                let polygon = geometry.polyhedral_outward_facet(facet, volume).unwrap();
                let a = points[polygon[0]];
                for pair in polygon[1..].windows(2) {
                    let b = points[pair[0]];
                    let c = points[pair[1]];
                    determinant_sum += a[0] * (b[1] * c[2] - b[2] * c[1])
                        - a[1] * (b[0] * c[2] - b[2] * c[0])
                        + a[2] * (b[0] * c[1] - b[1] * c[0]);
                }
            }
            // V=A_base*h/3, diamond area2 or right-triangle area1/2.
            assert!((determinant_sum / 6. - expected).abs() < 16. * f64::EPSILON);
        }
    }
}

#[test]
fn subprecision_nonplanarity_and_strict_positive_overlap_are_not_rounded_away() {
    let displacement = 2f64.powi(-45); // strictly positive and less than 1e-12 m
    let (mut vertices, shells) = bipyramid();
    vertices[1][0] = displacement;
    let error =
        CanonicalGeometryV1::from_convex_polyhedra(vertices, shells, vec![], 1e-12).unwrap_err();
    assert!(error.message().contains("exactly planar"), "{error:?}");
    // Different y/z extents keep all distinct vertices well separated. The
    // intended rejection is volume overlap, not the precision/vertex gate.
    let error = pair_prisms(
        [[-1., 0.], [-1., 1.], [-1., 1.]],
        [[-displacement, 1.], [-0.5, 0.5], [-0.5, 0.5]],
    )
    .unwrap_err();
    assert!(error.message().contains("interiors overlap"), "{error:?}");
}

#[test]
fn triangulated_subprecision_inward_dent_rejects_as_nonconvex() {
    let (mut vertices, mut facets) = prism([[-1., 1.]; 3]);
    vertices.push([0., 0., 1. - 2f64.powi(-45)]);
    facets.pop(); // replace only the upper-z square with its four triangles
    facets.extend([vec![1, 5, 8], vec![5, 7, 8], vec![7, 3, 8], vec![3, 1, 8]]);
    let error = CanonicalGeometryV1::from_convex_polyhedra(vertices, vec![facets], vec![], 1e-12)
        .unwrap_err();
    assert!(error.message().contains("nonconvex"), "{error:?}");
}

#[test]
fn exact_predicate_budget_rejects_bounded_but_excessive_polygon_work() {
    // The convex integer parabola chain plus its closing chord is extruded.
    // 1024 vertices and 3072 loop indices fit the ordinary decoder limits,
    // but a 512-vertex facet alone requires >262144 exact predicates.
    let n = 512usize;
    let vertices = (0..2)
        .flat_map(|z| (0..n).map(move |i| [i as f64, (i * i) as f64, z as f64]))
        .collect();
    let mut facets = vec![(0..n).rev().collect::<Vec<_>>(), (n..2 * n).collect()];
    for i in 0..n {
        let j = (i + 1) % n;
        facets.push(vec![i, j, n + j, n + i]);
    }
    let error = CanonicalGeometryV1::from_convex_polyhedra(vertices, vec![facets], vec![], 1e-12)
        .unwrap_err();
    assert!(
        error.message().contains("exact predicate budget exceeded"),
        "{error:?}"
    );
}

#[test]
fn point_membership_uses_exact_named_half_spaces_and_closed_boundaries() {
    let (vertices, shells) = tetrahedra();
    let geometry = admit(vertices, shells, 3);
    let replay = CanonicalGeometryV1::replay_canonical(
        geometry.canonical_bytes(),
        CanonicalGeometryLimits::default(),
    )
    .unwrap();
    for geometry in [&geometry, &replay] {
        // The two tetrahedra satisfy y,z >= 0 and |x|+y+z <= 1,
        // with left/right selected by the sign of x.
        for (point, left, right) in [
            ([-0.25, 0.25, 0.25], true, false),
            ([0.25, 0.25, 0.25], false, true),
            ([0., 0.25, 0.25], true, true),
            ([0.5, 0.25, 0.25], false, true),
            ([1., 0., 0.], false, true),
            ([0.5, 0.5, 0.5], false, false),
            ([0., -f64::EPSILON, 0.], false, false),
            ([1. + f64::EPSILON, 0., 0.], false, false),
        ] {
            assert_eq!(geometry.contains_point("left", &point).unwrap(), left);
            assert_eq!(geometry.contains_point("right", &point).unwrap(), right);
            assert_eq!(
                geometry.contains_point("both", &point).unwrap(),
                left || right
            );
        }
        for selection in ["missing", "interface", "edge", "vertex"] {
            assert!(geometry.contains_point(selection, &[0., 0., 0.]).is_err());
        }
        assert!(geometry.contains_point("both", &[0., 0.]).is_err());
        assert!(
            geometry
                .contains_point("both", &[0., 0., f64::NAN])
                .is_err()
        );
        assert!(
            geometry
                .contains_point("both", &[0., 0., f64::INFINITY])
                .is_err()
        );
    }
}
