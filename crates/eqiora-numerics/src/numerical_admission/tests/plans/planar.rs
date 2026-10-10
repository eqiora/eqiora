use super::*;
use eqiora_geometry::{PlanarFace, PlanarRegion};

mod moving;
mod storage;

fn geometry(mixed: bool) -> CanonicalGeometryV1 {
    let region = PlanarRegion::new(
        vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        vec![PlanarFace::new(vec![0, 1, 2, 3], vec![])],
        vec![
            NamedEntitySet::new("body", 2, vec![0]),
            NamedEntitySet::new("outer", 1, vec![0, 1, 2, 3]),
        ],
        1e-12,
    )
    .unwrap();
    if !mixed {
        return CanonicalGeometryV1::from_region(&region).unwrap();
    }
    let mut sets = vec![NamedEntitySet::new("body", 2, vec![0])];
    let corners = region.faces()[0].outer();
    for (index, (&a, &b)) in corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .enumerate()
    {
        let a = region.vertices()[a];
        let b = region.vertices()[b];
        let name = if a[0] == 0. && b[0] == 0. {
            "left"
        } else if a[0] == 1. && b[0] == 1. {
            "right"
        } else if a[1] == 0. && b[1] == 0. {
            "bottom"
        } else {
            "top"
        };
        sets.push(NamedEntitySet::new(name, 1, vec![index]));
    }
    CanonicalGeometryV1::from_region(
        &PlanarRegion::new(
            region.vertices().to_vec(),
            region.faces().to_vec(),
            sets,
            1e-12,
        )
        .unwrap(),
    )
    .unwrap()
}

// Bounded provider bytes exercise authentication; no external Gmsh run is claimed.
fn resources(geometry: &CanonicalGeometryV1, permuted: bool) -> AuthenticatedCommonMesh {
    let mut msh = String::from(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 5 1 5\n2 1 0 5\n1\n2\n3\n4\n5\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 0.5 0\n$EndNodes\n$Elements\n1 4 1 4\n2 1 2 4\n",
    );
    for (index, mut cell) in [[1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]]
        .into_iter()
        .enumerate()
    {
        if permuted {
            cell.rotate_left(1);
        }
        msh += &format!("{} {} {} {}\n", index + 1, cell[0], cell[1], cell[2]);
    }
    msh += "$EndElements\n";
    AuthenticatedCommonMesh::gmsh_4152(
        geometry.clone(),
        eqiora_artifact::GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap(),
        msh.into_bytes(),
    )
    .unwrap()
}

#[test]
fn planar_p1_executes_affine_scalar_and_replays_exact_plan_and_result() {
    for mixed in [false, true] {
        affine_profile(mixed);
    }
}

fn affine_profile(mixed: bool) {
    let geometry = geometry(mixed);
    let body = geometry.entity_set("body").unwrap();
    let source = r#"
public component Affine(
  support body: volume(ambient_dimension = 2),
  support outer: boundary(parent = body)
) {
  variable u: m on body in h1;
  relation balance on body { -div(grad(u)) = 0 [1 / m]; }
  relation prescribed on outer { trace(u) = coordinate(0); }
}
"#;
    let source = if mixed {
        source.replace("in h1", "in smooth").replace("support outer: boundary(parent = body)", "support left: boundary(parent = body), support right: boundary(parent = body), support bottom: boundary(parent = body), support top: boundary(parent = body)")
            .replace("relation prescribed on outer { trace(u) = coordinate(0); }", "relation prescribed on left { trace(u) = 0 [m]; } relation outflow on right { normal(grad(u)) = 1; } relation bottom_flux on bottom { normal(grad(u)) = 0; } relation top_flux on top { normal(grad(u)) = 0; }")
    } else {
        source.to_owned()
    };
    let boundaries = if mixed {
        vec!["left", "right", "bottom", "top"]
    } else {
        vec!["outer"]
    };
    let mut supports = vec![("body", body, None)];
    supports.extend(boundaries.iter().map(|name| {
        (
            *name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    let model = compile_model(
        "planar-affine.eqi",
        &source,
        &geometry,
        "Affine",
        &supports,
        &[],
    );
    let linear = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1e-11,
        1e-13,
        NonZeroUsize::new(1000).unwrap(),
    );
    for permuted in [false, true] {
        let owner = resources(&geometry, permuted);
        let resolve = |policy| {
            ResolvedCommonPlan::resolve(
                &model,
                owner.clone(),
                policy,
                CommonSolvePolicy::Linear(linear),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
        };
        let resolved = resolve(CommonSpatialPolicy::P1).unwrap();
        let plan = resolved.as_linear().unwrap();
        assert_eq!(plan.spatial(), CommonSpatialPolicy::P1);
        assert_eq!(
            plan.portable_realization().domains()[0]
                .discretization()
                .quadrature(),
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: NonZeroUsize::new(2).unwrap(),
                points_per_axis: NonZeroUsize::new(3).unwrap(),
            }
        );
        let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
        let (association, values, shape) = result.field_block(0, 0).unwrap();
        assert_eq!(association, "vertex");
        assert_eq!(shape, &[5]);
        // q=x is harmonic and lies exactly in P1. The center is a free DOF,
        // so this checks a solve as well as the prescribed exterior trace.
        let vertices = owner.simplicial_mesh().unwrap().mesh().vertices();
        assert_eq!(values.len(), vertices.len());
        for (value, point) in values.iter().zip(vertices) {
            assert!((value - point[0]).abs() < 1e-10);
        }
        let bytes = resolved.to_bytes().unwrap();
        let time = eqiora_time::TimeBackendCapabilities::new(
            eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[
                eqiora_core::ScalarDomain::Real,
                eqiora_core::ScalarDomain::Complex,
            ],
            &[eqiora_core::ScalarType::F64],
        );
        let replay =
            ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, time).unwrap();
        assert_eq!(replay.to_bytes().unwrap(), bytes);
        let result_bytes = result.to_bytes().unwrap();
        assert_eq!(
            crate::CommonResult::from_bytes(&result_bytes, &replay)
                .unwrap()
                .to_bytes()
                .unwrap(),
            result_bytes
        );
        assert!(
            resolve(CommonSpatialPolicy::Q1)
                .unwrap_err()
                .message()
                .contains("Q1 requires Cartesian")
        );
        assert!(resolve(CommonSpatialPolicy::TetrahedralEdge).is_err());
    }
}
