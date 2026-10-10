use super::*;

#[test]
fn planar_vector_traces_preserve_every_component() {
    let source = r#"
public model Throughflow(
    support body:volume(ambient_dimension=2),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    support bottom:boundary(parent=body),
    support top:boundary(parent=body)
) {
    variable u:vector<m/s,2> on body in smooth;
    variable g:m^2/s on body in smooth;
    variable f:kg/(m*s^2) on body in smooth;
    relation prescribed_velocity on body {g=1[m/s]*coordinate(0)+2[m/s]*coordinate(1);}
    relation force on body {f=0[kg/(m*s^2)];}
    relation momentum on body {-div(2[kg/(m*s)]*symmetric_part(grad(u)))-grad(f)=0;}
    relation inlet on left {trace(u)=trace(grad(g));}
    relation lower on bottom {trace(u)=trace(grad(g));}
    relation upper on top {trace(u)=trace(grad(g));}
    relation outlet on right {normal(2[kg/(m*s)]*symmetric_part(grad(u)))=0;}
}
"#;
    let geometry = geometry(true);
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    for source in [
        source.to_owned(),
        source.replace("Throughflow", "RenamedFlow"),
    ] {
        let entry = if source.contains("model RenamedFlow") {
            "RenamedFlow"
        } else {
            "Throughflow"
        };
        let model = compile_model(
            "common-planar-vector.eqi",
            &source,
            &geometry,
            entry,
            &supports,
            &[],
        );
        for permuted in [false, true] {
            let resolved = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonSpatialPolicy::P1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    1e-11,
                    1e-13,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let plan = resolved
                .as_linear()
                .expect("vector P1 must use the shared linear owner");
            let result = plan.run(&REFERENCE_LINEAR_SOLVER).unwrap();
            let published = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
            let (association, published_values, shape) = published.field_block(0, 0).unwrap();
            assert_eq!(association, "vertex");
            assert_eq!(shape, &[6, 2]);
            assert_eq!(published_values, result.fields[0].2);
            let bytes = published.to_bytes().unwrap();
            assert_eq!(
                crate::CommonResult::from_bytes(&bytes, &resolved)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            assert_eq!(result.fields.len(), 1);
            // u=(1,2) has zero symmetric gradient, so both the volume residual
            // and the right-edge traction vanish. P1 reproduces this exact
            // field at all six vertices, including the free outlet midpoint.
            let (_, value_type, values, space) = &result.fields[0];
            assert!(!value_type.shape().is_scalar());
            assert_eq!(
                *space,
                Space::continuous_lagrange(std::num::NonZeroU16::MIN)
            );
            assert_eq!(values.len(), 12);
            for value in values.as_chunks::<2>().0 {
                assert!((value[0] - 1.).abs() < 1e-10);
                assert!((value[1] - 2.).abs() < 1e-10);
            }
        }
    }
}

// A free outlet midpoint exercises natural boundaries as well as the
// constrained corners and the unconstrained interior vertex.
fn open_resources(geometry: &CanonicalGeometryV1, permuted: bool) -> AuthenticatedCommonMesh {
    let mut msh = String::from(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 6 1 6\n2 1 0 6\n1\n2\n3\n4\n5\n6\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 0.5 0\n1 0.5 0\n$EndNodes\n$Elements\n1 5 1 5\n2 1 2 5\n",
    );
    for (index, mut cell) in [[1, 2, 5], [2, 6, 5], [6, 3, 5], [3, 4, 5], [4, 1, 5]]
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
