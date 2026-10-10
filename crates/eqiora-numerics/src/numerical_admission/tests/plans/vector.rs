use super::*;
use eqiora_core::RawId;

const SOURCE: &str = r#"
public model Coupled(
    support body: volume(ambient_dimension=2),
    support left: boundary(parent=body),
    support right: boundary(parent=body),
    support bottom: boundary(parent=body),
    support top: boundary(parent=body)
) {
    coordinate x: m on body from body[0];
    coordinate y: m on body from body[1];
    parameter r: 1/m^2 = 1;
    variable u: vector<1,2> on body in h1;
    variable v: vector<1,2> on body in h1;
    variable w: vector<1,2> on body in h1;
    variable fu: 1/m on body in smooth;
    variable fv: 1/m on body in smooth;
    variable fw: 1/m on body in smooth;
    relation forcing_u on body { fu=3*(x+2*y)/1[m^2]; }
    relation forcing_v on body { fv=-(x+2*y)/1[m^2]; }
    relation forcing_w on body { fw=(x+2*y)/1[m^2]; }
    relation first on body { -div(grad(u))+r*(u-v)=grad(fu); }
    relation second on body { -div(grad(v))+r*(2*v-u-w)=grad(fv); }
    relation third on body { -div(grad(w))+r*(w-v)=grad(fw); }
    relation left_values on left { trace(u)=0; trace(v)=0; trace(w)=0; }
    relation right_values on right { trace(u)=0; trace(v)=0; trace(w)=0; }
    relation bottom_values on bottom { trace(u)=0; trace(v)=0; trace(w)=0; }
    relation top_values on top { trace(u)=0; trace(v)=0; trace(w)=0; }
}
"#;

fn resolve(source: &str) -> (ResolvedCommonPlan, [RawId; 3]) {
    let geometry = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        br#"{"schema":"eqiora.cartesian-box-envelope/v1","encoding":"eqiora.canonical-json/v1","length_unit":"metre","bounds":[[0.0,1.0],[0.0,1.0]],"entity_sets":[{"name":"bottom","dimension":1,"members":[2]},{"name":"left","dimension":1,"members":[0]},{"name":"right","dimension":1,"members":[1]},{"name":"top","dimension":1,"members":[3]},{"name":"body","dimension":2,"members":[0]}]}"#,
        eqiora_geometry::CanonicalGeometryLimits::default(),
    ).unwrap();
    let body = geometry.entity_set("body").unwrap();
    let mut bindings = vec![(
        "body",
        StaticBindingValue::GeometrySupport {
            geometry: &geometry,
            selection: body,
            parent: None,
        },
    )];
    for name in ["left", "right", "bottom", "top"] {
        bindings.push((
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: Some(body),
            },
        ));
    }
    let compiled =
        CompiledModel::compile_selected("coupled-vector.eqi", source, "Coupled", &bindings)
            .unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let fields = ["u", "v", "w"].map(|name| symbols.get(name).unwrap());
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    (replay_plan(resolved, &REFERENCE_LINEAR_SOLVER), fields)
}

#[test]
fn coupled_vector_fields_execute_every_component_and_replay() {
    let (replay, fields) = resolve(SOURCE);
    let plan = replay.as_linear().unwrap();
    let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    // Four square Q1 cells: central K=8/3, M=1/9, integral(phi)=1/4.
    // Reaction L has modes (1,1,1), (1,0,-1), (1,-2,1) and eigenvalues
    // 0,1,3. Load (3,-1,1) is their sum, so (24 I + L)u = 9/4 f.
    // The second physical component has exactly twice this load.
    let (a, b, c) = (9.0 / 4.0 / 24.0, 9.0 / 4.0 / 25.0, 9.0 / 4.0 / 27.0);
    assert_eq!(result.field_count(), 3);
    for (field, expected) in fields.into_iter().zip([a + b + c, a - 2.0 * c, a - b + c]) {
        let rank = plan
            .fields()
            .position(|(id, _)| id.erase() == field)
            .unwrap();
        let (association, values, shape) = result.field_block(rank, 0).unwrap();
        assert_eq!(association, "vertex");
        assert_eq!(shape, &[3, 3, 2]);
        for vertex in 0..9 {
            for component in 0..2 {
                let expected = if vertex == 4 {
                    expected * (component + 1) as f64
                } else {
                    0.0
                };
                assert!(
                    (values[2 * vertex + component] - expected).abs() < 1e-11,
                    "field {rank}, vertex {vertex}, component {component}: {} != {expected}",
                    values[2 * vertex + component]
                );
            }
        }
    }
    let bytes = result.to_bytes().unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&bytes, &replay)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
}

#[test]
fn vector_natural_boundaries_retain_both_normal_components() {
    let mut source = SOURCE.replace("on body in h1", "on body in smooth")
        .replace("parameter r:", "variable g: m on body in smooth; relation datum on body { g=(x^2+y^2)/1[m]; } variable p:1/m on body in smooth; relation pressure on body { p=2[1/m]; } parameter r:")
        .replace("fu=3*(x+2*y)/1[m^2]", "fu=1[1/m]")
        .replace("fv=-(x+2*y)/1[m^2]", "fv=1[1/m]")
        .replace("fw=(x+2*y)/1[m^2]", "fw=1[1/m]")
        .replace("trace(u)=0; trace(v)=0; trace(w)=0;", "trace(u)=trace(grad(g)); trace(v)=trace(grad(g)); trace(w)=trace(grad(g));");
    for boundary in ["right", "bottom", "top"] {
        source = source.replace(&format!("relation {boundary}_values on {boundary} {{ trace(u)=trace(grad(g)); trace(v)=trace(grad(g)); trace(w)=trace(grad(g)); }}"),
            &format!("relation {boundary}_values on {boundary} {{ normal(grad(u))=normal(isotropic_lift(p)); normal(grad(v))=normal(isotropic_lift(p)); normal(grad(w))=normal(isotropic_lift(p)); }}"));
    }
    let (replay, _) = resolve(&source);
    let plan = replay.as_linear().unwrap();
    let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    let NativeMeshResources::Cartesian { mesh, .. } = plan.admission.resources() else {
        panic!("Cartesian mesh")
    };
    // All three exact fields are grad((x²+y²)/m)=(2x/m,2y/m).
    // Their coupling vanishes and grad(u)=2 I/m supplies the outward traction.
    for field in 0..3 {
        let (_, values, shape) = result.field_block(field, 0).unwrap();
        assert_eq!(shape, &[3, 3, 2]);
        for vertex in 0..9 {
            let point = mesh
                .mesh()
                .vertex_coordinates(eqiora_meshing::MeshEntity::new(0, vertex))
                .unwrap();
            for component in 0..2 {
                assert!((values[2 * vertex + component] - 2.0 * point[component]).abs() < 1e-11);
            }
        }
    }
}

#[test]
fn vector_observables_evaluate_prescribed_data_without_extra_state_fields() {
    let center_u = 9.0 / 4.0 / 24.0 + 9.0 / 4.0 / 25.0 + 9.0 / 4.0 / 27.0;
    for (units, integrand, expected, variation) in [
        ("m", "fu", 4.5, 0.0),
        (
            "1/m^2",
            "contract(grad(fu),grad(fu),axes=((0,0),))",
            45.0,
            0.0,
        ),
        (
            "1",
            "contract(grad(fu),u,axes=((0,0),))",
            15.0 * center_u / 4.0,
            9.0,
        ),
    ] {
        let source = SOURCE.replace("relation first", &format!(
            "observable diagnostic: {units} = integral({integrand},measure(body)); relation first"
        ));
        let (replay, _) = resolve(&source);
        let plan = replay.as_linear().unwrap();
        let result = plan.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
        assert_eq!(result.field_count(), 3); // Prescribed fu is not a fourth unknown.
        let observable = plan
            .observation_program()
            .nodes()
            .find_map(|node| match node {
                eqiora_schema::kernel::KernelNode::Observable(definition) => Some(definition),
                _ => None,
            })
            .unwrap();
        let rules = std::collections::HashMap::from([(
            observable.reduction().domain().unwrap(),
            eqiora_meshing::QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap(),
        )]);
        let observed = result
            .observe(replay.model_artifact(), observable.id(), &rules)
            .unwrap();
        // fu=3x+6y on the unit square; grad(fu)=(3,6). The Q1
        // center basis integrates to 1/4 and u_y=2u_x. A unit State
        // direction leaves fu fixed and changes grad(fu).u by 3+6.
        assert!((observed.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-11);
        let direction = result
            .observable_state_tangent(plan.fields().map(|(field, ty)| {
                (
                    field,
                    vec![eqiora_core::DynQuantity::new(1.0, ty.dimension()); 18],
                )
            }))
            .unwrap();
        let jvp = result
            .observe_state_jvp(replay.model_artifact(), observable.id(), &rules, &direction)
            .unwrap();
        assert!((jvp.real_scalar_value().unwrap().value() - variation).abs() < 1e-11);
    }
}

#[test]
fn coupled_vector_reactions_keep_field_components_and_volume_loads() {
    let (replay, fields) = resolve(SOURCE);
    let mut recovered = None;
    replay
        .as_linear()
        .unwrap()
        .admission
        .execute_linear_with_completion(&REFERENCE_LINEAR_SOLVER, |reactions, full| {
            let result = reactions.recover(full)?;
            recovered = Some(result.clone());
            Ok(result)
        })
        .unwrap();
    let recovered = recovered.unwrap();
    assert_eq!(recovered.constrained_actions.len(), 3 * 8 * 2);
    assert_eq!(recovered.volume_loads.len(), 3 * 9 * 2);
    let b = 9.0 / 4.0 / 25.0;
    let c = 9.0 / 4.0 / 27.0;
    for ((field, force), coupling) in
        fields
            .into_iter()
            .zip([3.0, -1.0, 1.0])
            .zip([b + 3.0 * c, -6.0 * c, -b + 3.0 * c])
    {
        for component in 0..2 {
            let select = |key: &crate::region_assembly::mapping::FieldDof| {
                key.field == field && key.component == component
            };
            let load = recovered
                .volume_loads
                .iter()
                .filter(|(key, _)| select(key))
                .map(|(_, value)| value)
                .sum::<f64>();
            let reaction = recovered
                .constrained_actions
                .iter()
                .filter(|(key, _)| select(key))
                .map(|(_, value)| value)
                .sum::<f64>();
            let factor = (component + 1) as f64;
            assert!((load - force * factor).abs() < 1e-11);
            // Partition of unity cancels stiffness rows. The discrete coupling
            // integrates to L*u_center/4, leaving reaction = coupling - load.
            assert!((reaction - (coupling / 4.0 - force) * factor).abs() < 1e-11);
        }
    }
}

#[test]
fn shared_region_solve_preserves_the_admitted_operator_class() {
    let (replay, fields) = resolve(SOURCE);
    let admission = &replay.as_linear().unwrap().admission;
    let RecognizedNativeModel::Linear(equations) = admission.recognized_model() else {
        panic!("linear equations");
    };
    let NativeMeshResources::Cartesian { mesh, .. } = admission.resources() else {
        panic!("Cartesian mesh");
    };
    let (mapping, mut input) = equations.cartesian_assembly(mesh.mesh()).unwrap();
    // The independent modal derivation above gives 24I+L with eigenvalues
    // 24,25,27 for each physical component. Its positive factor 1/9 and
    // complete homogeneous boundary elimination preserve SPD.
    input.operator_properties = LinearOperatorProperties::SymmetricPositiveDefinite;
    let policy = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1e-13,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let output = mapping
        .solve(
            mesh.mesh(),
            input,
            NonZeroUsize::MIN,
            eqiora_solver::LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, policy),
            |reactions, full| reactions.recover(full),
        )
        .unwrap();
    let (_, unclassified) = equations.cartesian_assembly(mesh.mesh()).unwrap();
    let error = mapping
        .solve(
            mesh.mesh(),
            unclassified,
            NonZeroUsize::MIN,
            eqiora_solver::LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, policy),
            |reactions, full| reactions.recover(full),
        )
        .err()
        .unwrap();
    assert!(error.message().contains("General") && error.message().contains("ConjugateGradient"));
    let (a, b, c) = (9.0 / 4.0 / 24.0, 9.0 / 4.0 / 25.0, 9.0 / 4.0 / 27.0);
    for (field, expected) in fields.into_iter().zip([a + b + c, a - 2.0 * c, a - b + c]) {
        for (key, value) in &output.fields[&field].coefficients {
            let expected = if key.entity.index() == 4 {
                expected * (key.component + 1) as f64
            } else {
                0.0
            };
            assert!((value - expected).abs() < 1e-11);
        }
    }
}

#[test]
fn unrelated_linear_result_rejects_an_elastic_scientific_projection() {
    let (plan, _) = resolve(SOURCE);
    let result = plan
        .as_linear()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert!(result.elasticity_observation().is_none());
    let mut wire: serde_json::Value = serde_json::from_slice(&result.to_bytes().unwrap()).unwrap();
    wire["content"]["payload"]["observation"]["elasticity"] = serde_json::json!({
        "constrained_reaction": [0.0, 0.0], "integrated_body_force": [0.0, 0.0],
        "exact_bounds": [[0.0, 1.0], [0.0, 1.0]]
    });
    assert!(
        crate::CommonResult::from_bytes(&serde_json::to_vec(&wire).unwrap(), &plan)
            .unwrap_err()
            .message()
            .contains("elastic observation differs")
    );
}
