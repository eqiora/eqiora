use super::*;
use eqiora_core::RawId;
use eqiora_meshing::{CartesianMesh, MeshEntity};

fn source(connections: bool) -> String {
    let mut source = String::from(
        "model VectorTransmission() { parameter r:1/m^2=1; parameter k:1=2; parameter other:1=2;",
    );
    for index in 0..3 {
        source += &format!("domain body{index}=box({index},{end},0,1);
            domain left{index}=boundary(body{index},axis=0,side=lower);
            domain right{index}=boundary(body{index},axis=0,side=upper);
            domain bottom{index}=boundary(body{index},axis=1,side=lower);
            domain top{index}=boundary(body{index},axis=1,side=upper);
            variable g{index}:m on body{index} in smooth;
            relation potential{index} on body{index} {{ g{index}=(coordinate(0)^2+coordinate(1)^2)/2[m]; }}
            variable u{index}:vector<1,2> on body{index} in smooth;
            variable v{index}:vector<1,2> on body{index} in smooth;
            relation first{index} on body{index} {{ -div(k*grad(u{index}))+r*(u{index}-v{index})=-r*grad(g{index}); }}
            relation second{index} on body{index} {{ -div(k*grad(v{index}))+r*(v{index}-u{index})=r*grad(g{index}); }}", end=index+1);
        for side in ["left", "right", "bottom", "top"] {
            if (side == "left" && index > 0) || (side == "right" && index < 2) {
                continue;
            }
            source += &format!(
                "relation fixed_{side}{index} on {side}{index} {{
                trace(u{index})=trace(grad(g{index}));
                trace(v{index})=2*trace(grad(g{index}));
            }}"
            );
        }
    }
    for index in 0..2 {
        let next = index + 1;
        if connections {
            for field in ["u", "v"] {
                source += &format!("instance outgoing_{field}{index}: VectorInterface(body=body{index},face=right{index},value={field}{index},conductivity=k);
                    instance incoming_{field}{index}: VectorInterface(body=body{next},face=left{next},value={field}{next},conductivity=k);
                    connect outgoing_{field}{index}.edge,incoming_{field}{index}.edge;");
            }
            continue;
        }
        source += &format!("domain contact{index}=interface(right{index},left{next});");
        for field in ["u", "v"] {
            source += &format!(
                "relation trace_{field}{index} on contact{index} {{
                trace({field}{index})=trace({field}{next});
            }}
            relation flux_{field}{index} on contact{index} {{
                normal(k*grad({field}{index}))=normal(k*grad({field}{next}));
            }}"
            );
        }
    }
    if connections {
        source = String::from(
            r#"
public connector VectorBoundary {
    trace value:1;
    flux outward_flux:1/m;
    shape spatial_vector;
    frame spatial;
    pairing euclidean_boundary_duality;
    orientation parent_outward;
}
public component VectorInterface(
    support body:volume(ambient_dimension=2),
    support face:boundary(parent=body),
    variable value:vector<1,2> on body in smooth,
    parameter conductivity:1,
    port edge:VectorBoundary over face
) {
    relation carrier on face {
        trace(value)=edge.value;
        normal(conductivity*grad(value))=edge.outward_flux;
    }
}
"#,
        ) + &source;
    }
    source + "}"
}

fn resolve(source: &str) -> Result<(ResolvedCommonPlan, BTreeMap<String, RawId>), Diagnostic> {
    let (transaction, model, symbols) = eqiora_compiler::compile("vector-interfaces.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let ids = (0..3)
        .flat_map(|index| [format!("u{index}"), format!("v{index}")])
        .map(|name| {
            let id = symbols.get(&name).unwrap();
            (name, id)
        })
        .collect();
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,3.0],[0.0,1.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"body","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    ).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[6, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-13,
            NonZeroUsize::new(1000).unwrap(),
        )),
        None,
        None,
        &ResolveOnlyBackend,
        None,
    )?;
    Ok((resolved, ids))
}

#[test]
fn multiple_vector_field_pairs_cross_interfaces_and_replay() {
    for connections in [false, true] {
        check_solution(&source(connections));
        check_solution(&source(connections).replace("VectorTransmission", "RenamedTransmission"));
    }
}

fn check_solution(source: &str) {
    let (resolved, ids) = resolve(source).unwrap();
    let resolved = replay_plan(resolved, &ResolveOnlyBackend);
    let plan = resolved.as_linear().unwrap();
    assert_eq!(plan.portable_realization().domains().len(), 3);
    assert_eq!(plan.portable_realization().transformations().len(), 4);
    let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    assert_eq!(result.field_count(), 6);
    let result = crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &resolved).unwrap();
    for region in 0..3 {
        let local_mesh = CartesianMesh::from_axes(vec![
            vec![region as f64, region as f64 + 0.5, region as f64 + 1.0],
            vec![0.0, 0.5, 1.0],
        ])
        .unwrap();
        for (field, factor) in [("u", 1.0), ("v", 2.0)] {
            let id = ids[&format!("{field}{region}")].ulid().to_string();
            let index = (0..6)
                .find(|&index| result.field(index).unwrap().0 == id)
                .unwrap();
            let (_, values, shape) = result.field_block(index, 0).unwrap();
            assert_eq!(shape, &[3, 3, 2]);
            // Independent exact solution u=(x,y)/m, v=2u: zero Laplacians,
            // opposite reaction loads, continuous values and common-normal flux.
            for vertex in 0..9 {
                let point = local_mesh
                    .vertex_coordinates(MeshEntity::new(0, vertex))
                    .unwrap();
                for component in 0..2 {
                    assert!(
                        (values[2 * vertex + component] - factor * point[component]).abs() < 1e-9
                    );
                }
            }
        }
    }
}

#[test]
fn physical_vector_interface_requires_each_complete_constitutive_flux() {
    let source = source(false);
    let missing = source.replace("normal(k*grad(v0))=normal(k*grad(v1));", "");
    // Remove the entire now-empty Relation so this probe reaches admission.
    let missing = missing.replace(
        "relation flux_v0 on contact0 {\n                \n            }",
        "",
    );
    assert!(
        resolve(&missing)
            .unwrap_err()
            .message()
            .contains("flux balance")
    );
    let wrong = source.replace("normal(k*grad(v0))", "normal(other*grad(v0))");
    assert!(
        resolve(&wrong)
            .unwrap_err()
            .message()
            .contains("constitutive flux balance")
    );
}
