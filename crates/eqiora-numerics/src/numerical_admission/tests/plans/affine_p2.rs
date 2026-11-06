//! Ordinary admission/replay only. Independent candidate solves are separate.
use super::*;
use eqiora_geometry::GeometryGraph;

fn geometry(dimension: usize) -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let op = if dimension == 2 {
        graph
            .polygon(
                vec![[0.25, -0.5], [1.375, 1.0], [0.0625, 5.5], [-1.0625, 4.0]],
                1e-12,
            )
            .unwrap()
    } else {
        let vertices = (0..8)
            .map(|i| {
                let q = [
                    f64::from(i & 1),
                    f64::from((i >> 1) & 1),
                    f64::from((i >> 2) & 1),
                ];
                [
                    0.25 + 15. / 16. * q[0] - 3. / 32. * q[1] + 11. / 64. * q[2],
                    -0.5 + 5. / 4. * q[0] + 3. * q[1] - 51. / 32. * q[2],
                    0.75 + 5. / 2. * q[1] + 15. / 4. * q[2],
                ]
            })
            .collect();
        graph
            .convex_polyhedron(
                vertices,
                vec![
                    vec![0, 4, 6, 2],
                    vec![1, 3, 7, 5],
                    vec![0, 1, 5, 4],
                    vec![2, 6, 7, 3],
                    vec![0, 2, 3, 1],
                    vec![4, 5, 7, 6],
                ],
                1e-12,
            )
            .unwrap()
    };
    let boundaries = op.boundaries();
    let mut names = BTreeMap::from([("body".to_owned(), vec![op.region().into()])]);
    let positions = if dimension == 2 {
        vec![3, 1, 0, 2]
    } else {
        (0..6).collect()
    };
    for axis in 0..dimension {
        for (side, label) in ["lower", "upper"].iter().enumerate() {
            names.insert(
                format!("q{axis}_{label}"),
                vec![boundaries[positions[2 * axis + side]].into()],
            );
        }
    }
    graph.build(&op, &names).unwrap()
}

fn resources(geometry: &CanonicalGeometryV1, dimension: usize) -> AuthenticatedCommonMesh {
    let bytes = if dimension == 2 {
        include_bytes!("../../../../../../verify/numerics/affine-p2-scalar/models/skew-2d-n2.msh")
            .as_slice()
    } else {
        include_bytes!("../../../../../../verify/numerics/affine-p2-scalar/models/skew-3d-n2.msh")
            .as_slice()
    };
    AuthenticatedCommonMesh::import_msh41(
        geometry.clone(),
        eqiora_artifact::MshImportPolicyV1::metres(0.01, 128, 16 * 1024 * 1024).unwrap(),
        bytes.to_vec(),
    )
    .unwrap()
}

fn plan(
    dimension: usize,
    mass: bool,
    forcing: &str,
    flux_degree: usize,
    points: Option<usize>,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    let geometry = geometry(dimension);
    let names = (0..dimension)
        .flat_map(|axis| [format!("q{axis}_lower"), format!("q{axis}_upper")])
        .collect::<Vec<_>>();
    let supports = names
        .iter()
        .map(|name| format!("support {name}: boundary(parent=body)"))
        .collect::<Vec<_>>()
        .join(",");
    let mut laws = String::new();
    for name in &names {
        laws += &if name == "q0_upper" {
            format!("relation value_{name} on {name} {{trace(u)=0;}}")
        } else {
            format!(
                "relation flux_{name} on {name} {{normal(grad(u))=1[1/m]*(coordinate(0)/1[m])^{flux_degree};}}"
            )
        };
    }
    let reaction = if mass { "+u/1[m^2]" } else { "" };
    let source = format!(
        "public component AffineP2(support body:volume(ambient_dimension={dimension}),{supports}) {{variable u:1 on body in smooth;relation balance on body {{-div(grad(u)){reaction}=1[1/m^2]*({forcing});}}{laws}}}"
    );
    let body = geometry.entity_set("body").unwrap();
    let mut bindings = vec![("body", body, None)];
    bindings.extend(names.iter().map(|name| {
        (
            name.as_str(),
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    let model = compile_model(
        "affine-p2.eqi",
        &source,
        &geometry,
        "AffineP2",
        &bindings,
        &[],
    );
    let quadrature =
        points.map(
            |n| eqiora_realization::QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: NonZeroUsize::new(dimension).unwrap(),
                points_per_axis: NonZeroUsize::new(n).unwrap(),
            },
        );
    ResolvedCommonPlan::resolve(
        &model,
        resources(&geometry, dimension),
        CommonSpatialPolicy::P2,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-14,
            1e-14,
            NonZeroUsize::new(10000).unwrap(),
        )),
        None,
        None,
        &ResolveOnlyBackend,
        None,
        None,
        quadrature,
    )
}

fn points(plan: &ResolvedCommonPlan) -> usize {
    match plan.as_spatial().unwrap().portable_realization().domains()[0]
        .discretization()
        .quadrature()
    {
        eqiora_realization::QuadraturePolicy::SimplexDuffyGaussLegendre {
            points_per_axis, ..
        } => points_per_axis.get(),
        _ => panic!("expected simplex Duffy policy"),
    }
}

#[test]
fn p2_skew_simplex_plan_preserves_complete_vertex_edge_inventory_and_rule_on_replay() {
    for d in [2, 3] {
        let original = plan(d, false, "1", 0, Some(3)).unwrap();
        assert_eq!(points(&original), 3);
        let replayed = replay_plan(original, &ResolveOnlyBackend);
        assert_eq!(points(&replayed), 3);
        let spatial = replayed.as_spatial().unwrap();
        let field = spatial.fields().next().unwrap().0;
        let entities = spatial.field_coefficient_entities(field).unwrap();
        assert_eq!(
            entities.iter().filter(|e| e.dimension() == 0).count(),
            if d == 2 { 9 } else { 27 }
        );
        assert_eq!(
            entities.iter().filter(|e| e.dimension() == 1).count(),
            if d == 2 { 16 } else { 98 }
        );
        assert_eq!(entities.len(), if d == 2 { 25 } else { 125 });
        assert_eq!(
            spatial.field_space(field).unwrap(),
            Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap())
        );
    }
}

#[test]
fn p2_bound_volume_and_natural_facet_degrees_select_one_persisted_rule() {
    for d in [2, 3] {
        assert!(plan(d, false, "1", 0, Some(2)).is_err());
        let mass_min = if d == 2 { 3 } else { 4 };
        assert_eq!(points(&plan(d, true, "1", 0, None).unwrap()), mass_min);
        if d == 3 {
            assert!(plan(d, true, "1", 0, Some(3)).is_err());
        }
        let high = plan(d, false, "(coordinate(0)/1[m])^6", 0, None).unwrap();
        assert_eq!(points(&high), if d == 2 { 5 } else { 6 });
        let facet = plan(d, false, "1", 6, None).unwrap();
        assert_eq!(points(&facet), 5);
        assert!(plan(d, false, "1", 6, Some(4)).is_err());
        assert_eq!(points(&replay_plan(facet, &ResolveOnlyBackend)), 5);
    }
}

#[test]
fn p2_replay_rejects_deficient_volume_facet_and_wrong_policy_tags() {
    for d in [2, 3] {
        for flux_degree in [0, 6] {
            let original = plan(d, false, "1", flux_degree, None).unwrap();
            let mut wire: serde_json::Value =
                serde_json::from_slice(&original.to_bytes().unwrap()).unwrap();
            assert_eq!(wire["schema"], "eqiora.resolved-common-plan/v28");
            assert_eq!(wire["spatial"]["policy"], "p2");
            let n = if flux_degree == 0 { 2 } else { 4 };
            wire["assembly_quadrature"]["points_per_axis"] = serde_json::json!(n);
            assert!(
                ResolvedCommonPlan::from_bytes(
                    &serde_json::to_vec(&wire).unwrap(),
                    &ResolveOnlyBackend,
                    &[]
                )
                .is_err()
            );
        }
        let original = plan(d, false, "1", 0, Some(3)).unwrap();
        for wrong in ["q2", "p3"] {
            let mut wire: serde_json::Value =
                serde_json::from_slice(&original.to_bytes().unwrap()).unwrap();
            wire["spatial"]["policy"] = serde_json::json!(wrong);
            assert!(
                ResolvedCommonPlan::from_bytes(
                    &serde_json::to_vec(&wire).unwrap(),
                    &ResolveOnlyBackend,
                    &[]
                )
                .is_err()
            );
        }
    }
    let geometry = cartesian_box_3d();
    let model = scalar_box_model(
        &geometry,
        POISSON_BOX,
        "PoissonBox",
        &[
            "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
        ],
    );
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[2, 2, 2]),
            CommonSpatialPolicy::P2,
            CommonSolvePolicy::Linear(exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-14,
                1e-14,
                NonZeroUsize::new(10000).unwrap()
            )),
            None,
            None,
            &ResolveOnlyBackend,
            None,
            None,
            None
        )
        .is_err()
    );
}
