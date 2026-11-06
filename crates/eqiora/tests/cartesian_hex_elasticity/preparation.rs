fn preparation_evidence(source: &str) -> Option<PathBuf> {
    static ATTEMPT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::var_os("EQIORA_HEX_PREPARATION_DIR")?;
    let directory = PathBuf::from(root).join(format!(
        "{}-{}",
        std::process::id(),
        ATTEMPT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&directory).unwrap();
    persist(&Some(directory.clone()), "model.eqi", source.as_bytes());
    Some(directory)
}

fn persist(directory: &Option<PathBuf>, name: &str, bytes: &[u8]) {
    use std::io::Write;
    if let Some(directory) = directory {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(directory.join(name))
            .unwrap();
        file.write_all(bytes).unwrap();
    }
}

struct Prepared {
    document: ModelDocument,
    model: eqiora::artifact::ModelEnvelope,
    mesh: CartesianMeshEnvelopeV1,
    plan: eqiora_numerics::ResolvedCommonPlan,
    evidence: Option<PathBuf>,
}

fn prepare_source(source: &str) -> Result<Prepared, Vec<eqiora::Diagnostic>> {
    let evidence = preparation_evidence(source);
    let graph = GeometryGraph::new();
    let cuboid = graph.cuboid([0., 2.], [0., 3.], [0., 4.]).unwrap();
    let mut names = BTreeMap::from([("body".to_owned(), vec![cuboid.region().into()])]);
    for (name, boundary) in SIDES.into_iter().zip(cuboid.boundaries()) {
        names.insert(
            name.to_owned(),
            vec![GeometryTopologyHandle::from(boundary)],
        );
    }
    let geometry = graph.build(&cuboid, &names).unwrap();
    let pressure = eqiora::DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).unwrap();
    let values = [
        (
            "mu",
            eqiora::ValueLiteral::new(
                eqiora::ValueType::scalar(eqiora::ScalarDomain::Real, pressure).unwrap(),
                [(3., 0.)],
            )
            .unwrap(),
        ),
        (
            "lambda",
            eqiora::ValueLiteral::new(
                eqiora::ValueType::scalar(eqiora::ScalarDomain::Real, pressure).unwrap(),
                [(2., 0.)],
            )
            .unwrap(),
        ),
    ];
    let mut bindings = vec![(
        "body",
        StaticBindingValue::GeometrySupport {
            geometry: &geometry,
            selection: geometry.entity_set("body").unwrap(),
            parent: None,
        },
    )];
    for side in SIDES {
        bindings.push((
            side,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(side).unwrap(),
                parent: Some(geometry.entity_set("body").unwrap()),
            },
        ));
    }
    bindings.extend(
        values
            .iter()
            .map(|(name, value)| (*name, StaticBindingValue::Value(value))),
    );
    let document = ModelDocument::compile_selected(
        "cartesian-hex-elasticity.eqi",
        source,
        "CartesianHexElasticity",
        &bindings,
    )?;
    assert!(
        document
            .authored_formulation_projection()
            .unwrap()
            .is_some()
    );
    let model = eqiora::artifact::ModelEnvelope::from_program(document.program()).unwrap();
    persist(&evidence, "model.json", &model.canonical_json().unwrap());
    let cells = CartesianMeshCellsV2::new([2; 3]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(&geometry, cells.cells())
            .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner = eqiora_numerics::AuthenticatedCommonMesh::structured_cartesian(
        geometry.clone(),
        mesh.clone(),
        correspondence,
        production,
    )
    .unwrap();
    persist(&evidence, "mesh.json", &owner.to_bytes().unwrap());
    let solver = eqiora_numerics::CommonSolvePolicy::Linear(
        eqiora_numerics::CommonLinearRequest::exact(
            eqiora::solver::SolverPlan::new(
                eqiora::solver::LinearSolver::BiConjugateGradientStabilized,
                1e-14,
                1e-14,
                NonZeroUsize::new(10000).unwrap(),
            )
            .unwrap()
            .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
            .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
            eqiora::solver::REFERENCE_SOLVER_PROVIDER,
        )
        .unwrap(),
    );
    let plan = eqiora_numerics::ResolvedCommonPlan::resolve(
        &model,
        owner,
        eqiora_numerics::CommonSpatialPolicy::Q1,
        solver,
        None,
        None,
        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
        None,
        document.authored_formulation_projection().unwrap(),
        Some(eqiora_realization::QuadraturePolicy::GaussLegendre {
            points_per_axis: NonZeroUsize::new(2).unwrap(),
        }),
    )
    .map_err(|diagnostic| vec![diagnostic])?;
    let bytes = plan.to_bytes().unwrap();
    persist(&evidence, "plan.json", &bytes);
    let description = plan.formulation().unwrap();
    assert_eq!(
        description.requested(),
        eqiora_numerics::FormulationSelectionMode::Authored
    );
    assert_eq!(
        description.requested_source_identity(),
        Some(
            document
                .authored_formulation_projection()
                .unwrap()
                .unwrap()
                .source_identity()
        )
    );
    let replay = eqiora_numerics::ResolvedCommonPlan::from_bytes(
        &bytes,
        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
        &[],
    )
    .unwrap();
    assert_eq!(replay.to_bytes().unwrap(), bytes);
    let replay_description = replay.formulation().unwrap();
    assert_eq!(
        replay_description.requested(),
        eqiora_numerics::FormulationSelectionMode::Authored
    );
    assert_eq!(
        replay_description.requested_source_identity(),
        description.requested_source_identity()
    );
    assert_physical_hex_corners(mesh.mesh());
    let prepared = Prepared {
        document,
        model,
        mesh,
        plan: replay,
        evidence,
    };
    let field = selected_field(&prepared);
    assert_eq!(
        prepared
            .plan
            .as_spatial()
            .unwrap()
            .field_coefficient_entities(field)
            .unwrap()
            .len(),
        27
    );
    let observables = ["energy", "left_x", "left_y", "left_z"]
        .map(|name| selected_observable(&prepared, name).ulid().to_string());
    persist(&prepared.evidence, "export-bindings.json", &serde_json::to_vec_pretty(&serde_json::json!({
        "field":field.ulid().to_string(),"observables":observables,"volume_quadrature":[3,2],"facet_quadrature":[2,2]})).unwrap());
    Ok(prepared)
}

#[test]
fn ordinary_hex_typed_source_and_retained_mesh_prepare_without_solving() {
    prepare_source(SOURCE).unwrap();
}

#[test]
fn physical_authored_traction_mismatches_reject_before_any_result() {
    prepare_source(SOURCE).expect("positive preparation must pass before mutation controls");
    let (equations, weak) = SOURCE.split_once("  form weak for balance {").unwrap();
    assert_eq!(weak.matches("dot(trace(w),tx)").count(), 1);
    let changed_component = format!(
        "{}{}{}{}",
        equations,
        "  parameter incorrect_tx: vector<Pa,3> = tensor_value(frame=body,components=[1.125,0.5,0.1875]);\n",
        "  form weak for balance {",
        weak.replacen("dot(trace(w),tx)", "dot(trace(w),incorrect_tx)", 1)
    );
    let reversed_traction = format!(
        "{}  form weak for balance {{{}",
        equations,
        weak.replacen("dot(trace(w),tx)", "dot(trace(w),-tx)", 1)
    );
    assert_rejected_controls([
        ("changed component", changed_component),
        ("reversed traction", reversed_traction),
    ]);
}

#[test]
fn unsupported_or_illtyped_authored_forms_reject_before_any_result() {
    prepare_source(SOURCE).expect("positive preparation must pass before unsupported controls");
    let (equations, weak) = SOURCE.split_once("  form weak for balance {").unwrap();
    let incomplete_contraction = format!(
        "{}  form weak for balance {{{}",
        equations,
        weak.replacen("2*mu*frobenius(symmetric_part(grad(w)),symmetric_part(grad(displacement))) + lambda*div(w)*div(displacement)", "2*mu*contract(symmetric_part(grad(w)),symmetric_part(grad(displacement)),axes=((0,0),)) + lambda*div(w)*div(displacement)", 1)
    );
    let wrong_unit = SOURCE.replacen("parameter tx: vector<Pa,3>", "parameter tx: vector<N,3>", 1);
    let insufficient_regularity = SOURCE.replacen(
        "variable displacement: vector<m,3> on body in smooth;",
        "variable displacement: vector<m,3> on body in h1;",
        1,
    );
    assert_rejected_controls([
        ("partial contraction", incomplete_contraction),
        ("wrong unit", wrong_unit),
        ("insufficient trial regularity", insufficient_regularity),
    ]);
}

fn assert_rejected_controls<const N: usize>(controls: [(&str, String); N]) {
    for (control, source) in controls {
        assert_ne!(source, SOURCE, "{control} did not alter the source");
        let diagnostics = prepare_source(&source).err().expect(control);
        assert!(
            !diagnostics.is_empty(),
            "{control} returned no typed diagnostic"
        );
        assert!(
            diagnostics.iter().all(|diagnostic| !diagnostic
                .message()
                .contains("coefficient alias exceeds the authored expression inventory")),
            "{control} reached the old coefficient inventory limitation"
        );
    }
}

fn assert_physical_hex_corners(native: &CartesianMesh) {
    assert_eq!(native.entity_count(0), Some(27));
    assert_eq!(native.entity_count(3), Some(8));
    for cell in 0..8 {
        let vertices = native.entity_vertices(MeshEntity::new(3, cell)).unwrap();
        assert_eq!(vertices.len(), 8);
        let corners: Vec<_> = vertices
            .into_iter()
            .map(|v| native.vertex_coordinates(v).unwrap())
            .collect();
        assert!(physical_corner_mapping_valid(&corners));
    }
}

fn physical_corner_mapping_valid(points: &[Vec<f64>]) -> bool {
    if points.len() != 8 || points.iter().any(|point| point.len() != 3) {
        return false;
    }
    let lower = (0..3)
        .map(|a| points.iter().map(|p| p[a]).min_by(f64::total_cmp).unwrap())
        .collect::<Vec<_>>();
    points.iter().enumerate().all(|(corner, point)| {
        (0..3).all(|axis| {
            point[axis] == lower[axis] + [1., 1.5, 2.][axis] * ((corner >> axis) & 1) as f64
        })
    })
}

#[test]
fn native_hex_corner_order_binds_nonunit_physical_axes_without_solving() {
    let native = CartesianMesh::uniform(&[[0., 2.], [0., 3.], [0., 4.]], &[2; 3]).unwrap();
    assert_physical_hex_corners(&native);
}

#[test]
fn invalid_axes_and_real_decoder_budget_reject_before_any_result() {
    for axes in [
        vec![vec![0., 0.], vec![0., 3.], vec![0., 4.]],
        vec![vec![2., 0.], vec![0., 3.], vec![0., 4.]],
        vec![vec![0., f64::INFINITY], vec![0., 3.], vec![0., 4.]],
        vec![vec![0., f64::NAN], vec![0., 3.], vec![0., 4.]],
    ] {
        assert!(CartesianMesh::from_axes(axes).is_err());
    }
    let mesh = CartesianMesh::uniform(&[[0., 2.], [0., 3.], [0., 4.]], &[2; 3]).unwrap();
    let artifact = CartesianMeshEnvelopeV1::from_mesh(&mesh).unwrap();
    let limits = CartesianMeshDecoderLimits {
        max_cartesian_entities: 1,
        ..Default::default()
    };
    assert!(
        CartesianMeshEnvelopeV1::from_json_with_limits(&artifact.canonical_json().unwrap(), limits)
            .is_err()
    );
}
