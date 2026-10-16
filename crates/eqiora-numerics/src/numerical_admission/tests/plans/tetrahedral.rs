use super::*;
use crate::numerical_admission::tests::mesh::polyhedra::{geometry_with_boundaries, observation};

fn geometry() -> CanonicalGeometryV1 {
    let base = geometry_with_boundaries(vec![]);
    let vertices = base.polyhedral_vertices().unwrap();
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut walls = Vec::new();
    for &facet in base.polyhedral_volume_facets(0).unwrap() {
        let points = base.polyhedral_outward_facet(facet, 0).unwrap();
        if points.iter().all(|&v| vertices[v][0] == 0.) {
            left.push(facet);
        } else if points.iter().all(|&v| vertices[v][0] == 2.) {
            right.push(facet);
        } else {
            walls.push(facet);
        }
    }
    geometry_with_boundaries(vec![
        NamedEntitySet::new("left", 2, left),
        NamedEntitySet::new("right", 2, right),
        NamedEntitySet::new("walls", 2, walls),
    ])
}

fn resources(geometry: &CanonicalGeometryV1, permuted: bool) -> AuthenticatedCommonMesh {
    AuthenticatedCommonMesh::gmsh_4152(
        geometry.clone(),
        eqiora_artifact::GmshMeshPolicyV1::explicit(1e-12, 0.01, 12, 5.).unwrap(),
        observation(permuted),
    )
    .unwrap()
}

fn model(geometry: &CanonicalGeometryV1, source: &str) -> ModelEnvelope {
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "walls"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    compile_model(
        "tetrahedral.eqi",
        source,
        geometry,
        "Example",
        &supports,
        &[],
    )
}

fn resolve(
    model: &ModelEnvelope,
    mesh: AuthenticatedCommonMesh,
    policy: CommonSpatialPolicy,
    step: Option<CommonBackwardEuler>,
) -> ResolvedCommonPlan {
    ResolvedCommonPlan::resolve(
        model,
        mesh,
        policy,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-14,
            NonZeroUsize::new(1000).unwrap(),
        )),
        None,
        step,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
}

const AFFINE: &str = r#"
public model Example(
    support body: volume(ambient_dimension=3),
    support left: boundary(parent=body),
    support right: boundary(parent=body),
    support walls: boundary(parent=body)
) {
    variable u: m on body in smooth;
    relation balance on body {-div(grad(u))=0[1/m];}
    relation prescribed on left {trace(u)=0[m];}
    relation outflow on right {normal(grad(u))=1;}
    relation insulated on walls {normal(grad(u))=0;}
}
"#;

#[test]
fn tetrahedral_p1_solves_affine_mixed_boundaries_and_replays() {
    let geometry = geometry();
    let model = model(&geometry, AFFINE);
    for permuted in [false, true] {
        let mesh = resources(&geometry, permuted);
        let resolved = replay_plan(
            resolve(&model, mesh.clone(), CommonSpatialPolicy::P1, None),
            &REFERENCE_LINEAR_SOLVER,
        );
        let result = resolved
            .as_spatial()
            .unwrap()
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let (_, values, shape) = result.field_block(0, 0).unwrap();
        assert_eq!(shape, &[8]);
        // u=x is harmonic, has outward derivative 1 on x=2, and zero
        // flux on the other free faces. Four right-face DOFs must be solved.
        for (value, point) in values
            .iter()
            .zip(mesh.simplicial_mesh().unwrap().mesh().vertices())
        {
            assert!((value - point[0]).abs() < 1e-10, "{value} at {point:?}");
        }
        let bytes = result.to_bytes().unwrap();
        assert_eq!(
            crate::CommonResult::from_bytes(&bytes, &resolved)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
    }
}

#[test]
fn tetrahedral_storage_uses_the_common_run_and_restart() {
    let source = AFFINE.replace("variable u: m", "state u: m")
        .replace("relation balance on body {-div(grad(u))=0[1/m];}",
                 "initial {u=2[m];} law balance on body {storage 1[s/m^2]*u; flux -grad(u); source -1[1/m^2]*u;}")
        .replace("trace(u)=0[m]", "normal(grad(u))=0")
        .replace("normal(grad(u))=1", "normal(grad(u))=0");
    let geometry = geometry();
    let model = model(&geometry, &source);
    let resolved = replay_plan(
        resolve(
            &model,
            resources(&geometry, false),
            CommonSpatialPolicy::P1,
            Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
        ),
        &REFERENCE_LINEAR_SOLVER,
    );
    let initial = resolved
        .as_spatial()
        .unwrap()
        .initial_state(0., Vec::new())
        .unwrap();
    let run =
        CommonTransientRunRequest::from_steps(resolved.clone(), initial, 3, vec![1, 2, 3]).unwrap();
    let std::ops::ControlFlow::Continue(outputs) = run
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("unexpected cancellation")
    };
    for (step, output) in &outputs {
        // A constant field has zero diffusion and boundary flux. Backward
        // Euler reduces exactly to u_next=u/(1+dt), independently of the mesh.
        let expected = 2. * 0.8_f64.powi(*step as i32);
        for value in output.state().spatial_values().unwrap() {
            assert!((value - expected).abs() < 1e-10);
        }
    }
    let restart =
        CommonState::from_bytes(&outputs[0].1.state().to_bytes().unwrap(), &resolved).unwrap();
    let request = CommonTransientRunRequest::from_steps(resolved, restart, 2, vec![2]).unwrap();
    let std::ops::ControlFlow::Continue(restarted) = request
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("unexpected cancellation")
    };
    assert_eq!(restarted[0].1.state(), outputs[2].1.state());
}

const MIXED: &str = r#"
public model Example(
    support body: volume(ambient_dimension=3),
    support left: boundary(parent=body),
    support right: boundary(parent=body),
    support walls: boundary(parent=body)
) {
    variable u: vector<m/s,3> on body in smooth;
    variable p: kg/(m*s^2) on body in h1;
    variable g: m^2/s on body in smooth;
    variable f: kg/(m*s^2) on body in smooth;
    relation profile on body {g=1[m/s]*coordinate(0)+2[m/s]*coordinate(1)+3[m/s]*coordinate(2);}
    relation force on body {f=0[kg/(m*s^2)];}
    relation momentum on body {-div(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))-grad(f)=0;}
    relation constraint on body {div(u)=0;}
    relation inlet on left {trace(u)=trace(grad(g));}
    relation outlet on right {normal(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))=0;}
    relation side_traction on walls {normal(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))+normal(isotropic_lift(f))=0;}
}
"#;

#[test]
fn tetrahedral_mini_preserves_vector_components_and_pressure_coupling() {
    let geometry = geometry();
    for gradient in [0., 1.] {
        let source = if gradient == 0. {
            MIXED.to_owned()
        } else {
            MIXED.replace(
                "f=0[kg/(m*s^2)]",
                "f=1[kg/(m^2*s^2)]*coordinate(0)-2[kg/(m*s^2)]",
            )
        };
        let model = model(&geometry, &source);
        for permuted in [false, true] {
            let mesh = resources(&geometry, permuted);
            let resolved = replay_plan(
                resolve(&model, mesh.clone(), CommonSpatialPolicy::MiniP1, None),
                &REFERENCE_LINEAR_SOLVER,
            );
            let plan = resolved.as_spatial().unwrap();
            assert_eq!(
                plan.portable_realization().domains()[0]
                    .discretization()
                    .quadrature(),
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(3).unwrap(),
                    points_per_axis: NonZeroUsize::new(6).unwrap(),
                }
            );
            let raw = plan.run(&REFERENCE_LINEAR_SOLVER).unwrap();
            let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
            assert_eq!(raw.fields.len(), 2);
            for (index, (_, ty, _, _)) in raw.fields.iter().enumerate() {
                let (_, values, shape) = result.field_block(index, 0).unwrap();
                if ty.shape().is_scalar() {
                    assert_eq!(shape, &[8]);
                    // Constant u=(1,2,3) has zero strain and divergence.
                    // grad(p)=grad(f); p=f=gradient*(x-2) gives the prescribed
                    // wall traction and zero outlet traction at x=2.
                    for (value, point) in values
                        .iter()
                        .zip(mesh.simplicial_mesh().unwrap().mesh().vertices())
                    {
                        assert!(
                            (value - gradient * (point[0] - 2.)).abs() < 1e-9,
                            "{value} at {point:?}"
                        );
                    }
                } else {
                    assert_eq!(shape, &[8, 3]);
                    for value in values.as_chunks::<3>().0 {
                        for (actual, expected) in value.iter().zip([1., 2., 3.]) {
                            assert!((actual - expected).abs() < 1e-9);
                        }
                    }
                    let (_, bubbles, shape) = result.field_block(index, 1).unwrap();
                    assert_eq!(shape, &[6, 3]);
                    assert!(bubbles.iter().all(|value| value.abs() < 1e-9));
                }
            }
            let bytes = result.to_bytes().unwrap();
            assert_eq!(
                crate::CommonResult::from_bytes(&bytes, &resolved)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
        }
    }
}

#[test]
fn tetrahedral_boundary_coverage_and_multiplier_closure_are_required() {
    let geometry = geometry();
    let incomplete = model(
        &geometry,
        &AFFINE.replace("relation insulated on walls {normal(grad(u))=0;}", ""),
    );
    let error = ResolvedCommonPlan::resolve(
        &incomplete,
        resources(&geometry, false),
        CommonSpatialPolicy::P1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-14,
            NonZeroUsize::new(1000).unwrap(),
        )),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap_err();
    assert!(error.message().contains("boundary"), "{error:?}");

    let closed = MIXED
        .replace("normal(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))=0;",
                 "trace(u)=trace(grad(g));")
        .replace("normal(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))+normal(isotropic_lift(f))=0;",
                 "trace(u)=trace(grad(g));");
    let closed = model(&geometry, &closed);
    let resolved = resolve(
        &closed,
        resources(&geometry, false),
        CommonSpatialPolicy::MiniP1,
        None,
    );
    let plan = resolved.as_spatial().unwrap();
    for error in [
        plan.reaccept_static_coefficients(&zero_real_coefficients(plan), None)
            .unwrap_err(),
        plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap_err(),
    ] {
        assert!(
            error.message().contains("unclosed constant multiplier"),
            "{error:?}"
        );
    }
}
