//! Ordinary 3D Hex preparation. Numerical qualification is separately reviewed.
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    CartesianMeshCellsV2, CartesianMeshDecoderLimits, CartesianMeshEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::{GeometryGraph, GeometryTopologyHandle};
use eqiora::meshing::{CartesianMesh, MeshEntity, MeshTopology};

const SOURCE: &str = include_str!("../../../verify/numerics/cartesian-hex-elasticity/model.eqi");
const SIDES: [&str; 6] = [
    "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
];

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

fn binary64_hex(value: f64) -> String {
    assert!(value.is_finite(), "nonfinite exported value");
    assert!(
        value != 0. || !value.is_sign_negative(),
        "negative zero exported value"
    );
    if value == 0. {
        return "0x0.0p+0".to_owned();
    }
    let bits = value.to_bits();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        format!("{sign}0x0.{fraction:013x}p-1022")
    } else {
        format!("{sign}0x1.{fraction:013x}p{:+}", exponent - 1023)
    }
}

#[test]
fn native_binary64_projection_preserves_ieee_components() {
    for (value, expected) in [
        (0., "0x0.0p+0"),
        (1., "0x1.0000000000000p+0"),
        (1.5, "0x1.8000000000000p+0"),
        (-2., "-0x1.0000000000000p+1"),
        (f64::from_bits(1), "0x0.0000000000001p-1022"),
        (f64::MIN_POSITIVE, "0x1.0000000000000p-1022"),
    ] {
        assert_eq!(binary64_hex(value), expected);
    }
}

fn selected_field(prepared: &Prepared) -> eqiora::Id<eqiora::entity::kinds::Field> {
    let selected = prepared
        .document
        .aliases()
        .get("definition.displacement")
        .or_else(|| prepared.document.aliases().get("displacement"))
        .unwrap()
        .downcast()
        .unwrap();
    assert!(prepared.document.program().nodes().any(
        |node| matches!(node, eqiora::kernel::KernelNode::Field(value) if value.id() == selected)
    ));
    selected
}

fn selected_observable(
    prepared: &Prepared,
    name: &str,
) -> eqiora::Id<eqiora::entity::kinds::Observable> {
    let selected = prepared
        .document
        .aliases()
        .get(&format!("definition.{name}"))
        .or_else(|| prepared.document.aliases().get(name))
        .unwrap()
        .downcast()
        .unwrap();
    assert!(prepared.document.program().nodes().any(|node|
        matches!(node, eqiora::kernel::KernelNode::Observable(value) if value.id() == selected)));
    selected
}

fn cartesian_block_shape(mesh: &CartesianMesh, value_type: &eqiora::ValueType) -> Vec<usize> {
    let mut shape = (0..mesh.topological_dimension())
        .map(|axis| mesh.axis_coordinates(axis).unwrap().len())
        .collect::<Vec<_>>();
    shape.extend(
        value_type
            .shape()
            .extents()
            .iter()
            .map(|extent| extent.get() as usize),
    );
    shape
}

#[test]
fn cartesian_vector_block_retains_axes_before_components() {
    let mesh = CartesianMesh::uniform(&[[0., 2.], [0., 3.], [0., 4.]], &[2; 3]).unwrap();
    let ty = eqiora::ValueType::shaped(
        eqiora::ScalarDomain::Real,
        eqiora::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
        eqiora::ValueShape::new([3]).unwrap(),
        eqiora::ValueFrame::SpatialCartesian,
    )
    .unwrap();
    assert_eq!(cartesian_block_shape(&mesh, &ty), [3, 3, 3, 3]);
}

fn native_export(prepared: &Prepared, result: &eqiora_numerics::CommonResult) -> serde_json::Value {
    use eqiora::kernel::KernelNode;
    use std::collections::{BTreeSet, HashMap};
    let spatial = prepared.plan.as_spatial().unwrap();
    assert_eq!(
        result.plan().to_bytes().unwrap(),
        prepared.plan.to_bytes().unwrap()
    );
    assert_eq!(result.field_count(), 1);
    let field = selected_field(prepared);
    let (actual_field, value_type, space) = result.field(0).unwrap();
    assert_eq!(actual_field, field.ulid().to_string());
    assert_eq!(space, spatial.field_space(field).unwrap());
    assert_eq!(value_type.scalar_domain(), eqiora::ScalarDomain::Real);
    assert_eq!(
        value_type
            .shape()
            .extents()
            .iter()
            .map(|v| v.get())
            .collect::<Vec<_>>(),
        [3]
    );
    assert_eq!(value_type.array_rank(), 0);
    assert_eq!(value_type.frame(), eqiora::ValueFrame::SpatialCartesian);
    assert_eq!(
        value_type.dimension(),
        eqiora::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
    );
    assert_eq!(result.field_block_count(0), 1);
    let (association, coefficients, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(association, "vertex");
    let expected_shape = cartesian_block_shape(prepared.mesh.mesh(), value_type);
    assert_eq!(shape, expected_shape);
    assert_eq!(coefficients.len(), 81);
    let entities = spatial.field_coefficient_entities(field).unwrap();
    assert_eq!(entities.len(), 27);
    let mut records = Vec::new();
    let mut retained = BTreeSet::new();
    for (entity, values) in entities.into_iter().zip(coefficients.chunks_exact(3)) {
        assert_eq!(entity.dimension(), 0);
        let closure = prepared.mesh.mesh().entity_vertices(entity).unwrap();
        assert_eq!(closure, [entity]);
        let point = prepared.mesh.mesh().vertex_coordinates(entity).unwrap();
        for (component, &value) in ["x", "y", "z"].into_iter().zip(values) {
            assert!(retained.insert((
                point.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                component
            )));
            records.push(serde_json::json!({"coordinate_m": point.iter().copied().map(binary64_hex).collect::<Vec<_>>(),
                "component": component, "value_m": binary64_hex(value)}));
        }
    }
    let mut expected = BTreeSet::new();
    for x in [0_f64, 1., 2.] {
        for y in [0_f64, 1.5, 3.] {
            for z in [0_f64, 2., 4.] {
                for component in ["x", "y", "z"] {
                    expected.insert((vec![x.to_bits(), y.to_bits(), z.to_bits()], component));
                }
            }
        }
    }
    assert_eq!(retained, expected);
    let mut observed = Vec::new();
    for (index, name) in ["energy", "left_x", "left_y", "left_z"]
        .into_iter()
        .enumerate()
    {
        let observable = selected_observable(prepared, name);
        let domain = prepared
            .document
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(value) if value.id() == observable => {
                    value.reduction().domain()
                }
                _ => None,
            })
            .unwrap();
        let dimension = if index == 0 { 3 } else { 2 };
        let rule =
            eqiora::meshing::QuadratureRule::tensor_product_gauss_legendre(dimension, 2).unwrap();
        let rules = HashMap::from([(domain, rule)]);
        let observation = result
            .observe(&prepared.model, observable, &rules, None)
            .unwrap();
        assert_eq!(observation.observable(), observable);
        assert_eq!(observation.result_identity(), result.identity());
        assert_eq!(observation.quadratures().len(), 1);
        let actual = observation.quadratures().get(&domain).unwrap();
        assert_eq!(actual.reference_cell().dimension(), dimension);
        assert_eq!(actual.points().len(), 1 << dimension);
        let value = observation.value();
        assert!(value.value_type().shape().is_scalar());
        let units = if index == 0 {
            [1, 2, -2, 0, 0, 0, 0]
        } else {
            [1, 1, -2, 0, 0, 0, 0]
        };
        assert_eq!(
            value.value_type().dimension(),
            eqiora::DimExponents::from_integers(units).unwrap()
        );
        observed.push(binary64_hex(value.real_scalar_value().unwrap().value()));
    }
    serde_json::json!({"identity": {"dimension":3,"family":"Hex","order":"Q1","geometry":"affine-cartesian",
        "cells":[2,2,2],"quadrature":[2,2,2],"units":["m","J","N"]},
        "lineage":{"mesh":spatial.mesh_digest(),"model":prepared.model.digest().unwrap().to_string(),"result":result.identity()},
        "coefficients":records,"energy_J":observed[0],"left_traction_N":&observed[1..]})
}

#[test]
#[ignore = "one scientific Run requires a reviewed native producer/build receipt"]
fn native_ordinary_hex_result_observation_and_canonical_replay() {
    use sha2::{Digest, Sha256};
    let hash = |bytes: &[u8]| {
        Sha256::digest(bytes)
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect::<String>()
    };
    let receipt_path =
        std::env::var("EQIORA_HEX_NATIVE_REVIEW_RECEIPT").expect("reviewed receipt required");
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(receipt_path).unwrap()).unwrap();
    assert_eq!(receipt["producer_review"], "accepted");
    let binary = std::env::current_exe().unwrap();
    assert_eq!(
        receipt["binary_sha256"],
        hash(&std::fs::read(&binary).unwrap())
    );
    assert_eq!(
        receipt["native_producer_sha256"],
        hash(include_bytes!("cartesian_hex_elasticity.rs"))
    );
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert_eq!(
        receipt["source_commit"],
        String::from_utf8(commit.stdout).unwrap().trim()
    );
    let tree = std::process::Command::new("git")
        .args(["rev-parse", "HEAD^{tree}"])
        .output()
        .unwrap();
    assert_eq!(
        receipt["source_tree"],
        String::from_utf8(tree.stdout).unwrap().trim()
    );
    assert_eq!(receipt["model_source_sha256"], hash(SOURCE.as_bytes()));
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../verify/numerics/cartesian-hex-elasticity/reference");
    for name in ["reference.py", "protocol.py"] {
        assert_eq!(
            receipt["reference_sha256"][name],
            hash(&std::fs::read(reference.join(name)).unwrap())
        );
    }
    assert!(
        std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .unwrap()
            .stdout
            .is_empty()
    );
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        assert_eq!(std::env::var(name).as_deref(), Ok("1"));
    }
    assert!(
        std::env::var("LD_LIBRARY_PATH")
            .unwrap_or_default()
            .is_empty()
    );
    let cpu = std::env::var("EQIORA_HEX_NATIVE_CPU").expect("declared CPU required");
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    assert_eq!(
        status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .unwrap()
            .trim(),
        cpu
    );
    let output =
        PathBuf::from(std::env::var_os("EQIORA_HEX_NATIVE_OUTPUT").expect("new output required"));
    assert!(output.is_absolute(), "absolute evidence directory required");
    assert!(!output.exists(), "preserve earlier attempts");
    assert_eq!(
        std::env::var_os("EQIORA_HEX_PREPARATION_DIR").map(PathBuf::from),
        Some(output.clone())
    );
    std::fs::create_dir(&output).unwrap();
    let root = Some(output.clone());
    persist(&root, "launch.json", &serde_json::to_vec_pretty(&serde_json::json!({"pid":std::process::id(),
        "binary":binary,"binary_sha256":receipt["binary_sha256"],"source_commit":receipt["source_commit"],
        "review_receipt":receipt,"cpu":cpu,"ordinary_run_attempted":0,"pde_solves":0,"preparation_manifest":"preparation.json",
        "environment":(["CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","RAYON_NUM_THREADS","LD_LIBRARY_PATH"].into_iter()
            .map(|name| (name, std::env::var(name).ok())).collect::<BTreeMap<_,_>>())})).unwrap());
    let prepared = prepare_source(SOURCE).unwrap();
    assert!(prepared.evidence.is_some());
    let plan_wire: serde_json::Value =
        serde_json::from_slice(&prepared.plan.to_bytes().unwrap()).unwrap();
    persist(&root, "preparation.json", &serde_json::to_vec_pretty(&serde_json::json!({
        "artifact_directory":prepared.evidence,"result_path":prepared.evidence.as_ref().unwrap().join("result.json"),
        "model_digest":prepared.model.digest().unwrap().to_string(),"mesh_digest":prepared.plan.as_spatial().unwrap().mesh_digest(),
        "plan_identity":plan_wire["identity"],"field":selected_field(&prepared).ulid().to_string(),
        "observables":(["energy","left_x","left_y","left_z"].map(|name| selected_observable(&prepared,name).ulid().to_string())),
        "status":"prepared-no-run"})).unwrap());
    persist(
        &root,
        "run-attempt.json",
        b"{\"ordinary_run_attempted\":1,\"declared_pde_solves\":1}\n",
    );
    let result = match prepared
        .plan
        .as_spatial()
        .unwrap()
        .run_result(None, &eqiora::solver::REFERENCE_LINEAR_SOLVER)
    {
        Ok(result) => result,
        Err(error) => {
            persist(
                &root,
                "failure.json",
                &serde_json::to_vec_pretty(
                    &serde_json::json!({"diagnostic":error.to_string(),"ordinary_run_attempted":1}),
                )
                .unwrap(),
            );
            panic!("native ordinary Run rejected: {error}");
        }
    };
    let result_bytes = result.to_bytes().unwrap();
    // Retain the actual accepted Result before replay, observations or comparison.
    persist(&prepared.evidence, "result.json", &result_bytes);
    let replay = eqiora_numerics::CommonResult::from_bytes(&result_bytes, &prepared.plan).unwrap();
    assert_eq!(replay.to_bytes().unwrap(), result_bytes);
    for (name, lane) in [("primary.json", &result), ("replay.json", &replay)] {
        persist(
            &root,
            name,
            &serde_json::to_vec_pretty(&native_export(&prepared, lane)).unwrap(),
        );
    }
    compare_native_exports(&output);
    persist(
        &root,
        "qualification.json",
        b"{\"status\":\"passed\",\"pde_solves\":1,\"profile_calls\":0,\"result_reopens\":1}\n",
    );
}

fn compare_native_exports(output: &std::path::Path) {
    let root = Some(output.to_path_buf());
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../verify/numerics/cartesian-hex-elasticity/reference");
    let protocol_python = std::env::var_os("EQIORA_HEX_PROTOCOL_PYTHON")
        .expect("independent protocol interpreter required");
    let checked = std::process::Command::new(protocol_python).current_dir(reference).args(["-c",
        "import pathlib,protocol,sys; q=protocol.r.budgets(); protocol.check(pathlib.Path(sys.argv[1]).read_text(),q); protocol.check(pathlib.Path(sys.argv[2]).read_text(),q)"])
        .arg(output.join("primary.json")).arg(output.join("replay.json")).output().unwrap();
    persist(&root, "protocol-stdout.txt", &checked.stdout);
    persist(&root, "protocol-stderr.txt", &checked.stderr);
    assert!(
        checked.status.success(),
        "independent primary/replay protocol rejected; inspect retained stderr"
    );
}

#[test]
fn actual_hex_corner_mapping_rejects_axis_and_local_node_permutations() {
    let mesh = CartesianMesh::uniform(&[[0., 2.], [0., 3.], [0., 4.]], &[2; 3]).unwrap();
    let corners = mesh
        .entity_vertices(MeshEntity::new(3, 0))
        .unwrap()
        .into_iter()
        .map(|vertex| mesh.vertex_coordinates(vertex).unwrap())
        .collect::<Vec<_>>();
    assert!(physical_corner_mapping_valid(&corners));
    let mut wrong_local_nodes = corners.clone();
    wrong_local_nodes.swap(1, 2);
    assert!(!physical_corner_mapping_valid(&wrong_local_nodes));
    let mut wrong_axes = corners;
    for point in &mut wrong_axes {
        point.swap(0, 1);
    }
    assert!(!physical_corner_mapping_valid(&wrong_axes));
}

#[test]
#[ignore = "read-only recovery requires review of its distinct harness and original artifact bindings"]
fn recover_original_native_result_without_running() {
    use sha2::{Digest, Sha256};
    let hash = |bytes: &[u8]| {
        Sha256::digest(bytes)
            .iter()
            .map(|value| format!("{value:02x}"))
            .collect::<String>()
    };
    let receipt_path = std::env::var_os("EQIORA_HEX_NATIVE_RECOVERY_RECEIPT")
        .expect("reviewed recovery receipt required");
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(receipt_path).unwrap()).unwrap();
    assert_eq!(receipt["recovery_review"], "accepted");
    let binary = std::env::current_exe().unwrap();
    assert_eq!(
        receipt["recovery_binary_sha256"],
        hash(&std::fs::read(&binary).unwrap())
    );
    assert_eq!(
        receipt["recovery_harness_sha256"],
        hash(include_bytes!("cartesian_hex_elasticity.rs"))
    );
    for (argument, key) in [
        ("HEAD", "recovery_source_commit"),
        ("HEAD^{tree}", "recovery_source_tree"),
    ] {
        let actual = std::process::Command::new("git")
            .args(["rev-parse", argument])
            .output()
            .unwrap();
        assert_eq!(
            receipt[key],
            String::from_utf8(actual.stdout).unwrap().trim()
        );
    }
    assert!(
        std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .unwrap()
            .stdout
            .is_empty()
    );
    assert_eq!(
        receipt["original_source_commit"],
        "6826671d537882b3daded212866e76250b9c036f"
    );
    assert_eq!(
        receipt["original_runtime_binary_sha256"],
        "a4fc848c5540fbbab3637f1f2f86216015b25340100c9fed6198a8e3ad35a986"
    );
    assert_eq!(
        receipt["original_files_sha256"]["result.json"],
        "d31edd4e04c651d5a8fc4586d17d1c8e09aa67e2a3a7c82f2e68431373d70517"
    );
    let original_binary = PathBuf::from(receipt["original_runtime_binary_copy"].as_str().unwrap());
    assert_eq!(
        receipt["original_runtime_binary_sha256"],
        hash(&std::fs::read(original_binary).unwrap())
    );
    let original = PathBuf::from(receipt["original_attempt_directory"].as_str().unwrap());
    let original_artifacts =
        PathBuf::from(receipt["original_artifact_directory"].as_str().unwrap());
    for (name, directory) in [
        ("launch.json", &original),
        ("preparation.json", &original),
        ("run-attempt.json", &original),
        ("model.eqi", &original_artifacts),
        ("model.json", &original_artifacts),
        ("mesh.json", &original_artifacts),
        ("plan.json", &original_artifacts),
        ("result.json", &original_artifacts),
    ] {
        assert_eq!(
            receipt["original_files_sha256"][name],
            hash(&std::fs::read(directory.join(name)).unwrap())
        );
    }
    let original_launch: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original.join("launch.json")).unwrap()).unwrap();
    let original_preparation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original.join("preparation.json")).unwrap()).unwrap();
    assert_eq!(
        original_launch["source_commit"],
        receipt["original_source_commit"]
    );
    assert_eq!(
        original_launch["binary_sha256"],
        receipt["original_runtime_binary_sha256"]
    );
    assert_eq!(
        original_launch["review_receipt"]["producer_review"],
        "accepted"
    );
    assert_eq!(original_launch["pid"], 1431892);
    assert_eq!(
        original_preparation["artifact_directory"].as_str().unwrap(),
        original_artifacts.to_str().unwrap()
    );
    assert_eq!(
        std::fs::read(original_artifacts.join("model.eqi")).unwrap(),
        SOURCE.as_bytes()
    );
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../verify/numerics/cartesian-hex-elasticity/reference");
    for name in ["reference.py", "protocol.py"] {
        assert_eq!(
            receipt["reference_sha256"][name],
            hash(&std::fs::read(reference.join(name)).unwrap())
        );
    }
    let output = PathBuf::from(
        std::env::var_os("EQIORA_HEX_NATIVE_RECOVERY_OUTPUT")
            .expect("new recovery directory required"),
    );
    assert!(
        output.is_absolute() && !output.exists(),
        "preserve original and earlier recovery attempts"
    );
    assert_eq!(
        std::env::var_os("EQIORA_HEX_PREPARATION_DIR").map(PathBuf::from),
        Some(output.clone())
    );
    std::fs::create_dir(&output).unwrap();
    let root = Some(output.clone());
    persist(
        &root,
        "recovery-launch.json",
        &serde_json::to_vec_pretty(&serde_json::json!({
        "recovery_pid":std::process::id(),"recovery_binary":binary,"review_receipt":receipt,
        "original_attempt_status":"export-failed-after-one-completed-run","new_pde_solves":0}))
        .unwrap(),
    );
    // Compile/resolve only: bind fresh source selections to the exact archived artifacts.
    let mut prepared = prepare_source(SOURCE).unwrap();
    for name in ["model.eqi", "model.json", "mesh.json", "plan.json"] {
        assert_eq!(
            std::fs::read(prepared.evidence.as_ref().unwrap().join(name)).unwrap(),
            std::fs::read(original_artifacts.join(name)).unwrap(),
            "fresh no-solve binding differs for {name}"
        );
    }
    let plan_bytes = std::fs::read(original_artifacts.join("plan.json")).unwrap();
    prepared.plan = eqiora_numerics::ResolvedCommonPlan::from_bytes(
        &plan_bytes,
        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
        &[],
    )
    .unwrap();
    assert_eq!(prepared.plan.to_bytes().unwrap(), plan_bytes);
    let plan_wire: serde_json::Value = serde_json::from_slice(&plan_bytes).unwrap();
    assert_eq!(plan_wire["identity"], original_preparation["plan_identity"]);
    let model_bytes = std::fs::read(original_artifacts.join("model.json")).unwrap();
    prepared.model =
        eqiora::artifact::ModelEnvelope::from_json(&model_bytes, Default::default()).unwrap();
    assert_eq!(prepared.model.canonical_json().unwrap(), model_bytes);
    assert_eq!(
        prepared.model.digest().unwrap().to_string(),
        original_preparation["model_digest"].as_str().unwrap()
    );
    assert_eq!(
        selected_field(&prepared).ulid().to_string(),
        original_preparation["field"].as_str().unwrap()
    );
    for (index, name) in ["energy", "left_x", "left_y", "left_z"]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            selected_observable(&prepared, name).ulid().to_string(),
            original_preparation["observables"][index].as_str().unwrap()
        );
    }
    let result_bytes = std::fs::read(original_artifacts.join("result.json")).unwrap();
    let result = eqiora_numerics::CommonResult::from_bytes(&result_bytes, &prepared.plan).unwrap();
    assert_eq!(result.to_bytes().unwrap(), result_bytes);
    persist(&root, "original-result.json", &result_bytes);
    let replay =
        eqiora_numerics::CommonResult::from_bytes(&result.to_bytes().unwrap(), &prepared.plan)
            .unwrap();
    assert_eq!(replay.to_bytes().unwrap(), result_bytes);
    for (name, lane) in [("primary.json", &result), ("replay.json", &replay)] {
        persist(
            &root,
            name,
            &serde_json::to_vec_pretty(&native_export(&prepared, lane)).unwrap(),
        );
    }
    compare_native_exports(&output);
    persist(&root,"recovery-qualification.json",&serde_json::to_vec_pretty(&serde_json::json!({
        "status":"read-only-recovery-passed","new_pde_solves":0,"original_completed_pde_solves":1,
        "original_attempt_status":"export-failed-after-result-save","original_source_commit":receipt["original_source_commit"],
        "original_runtime_binary_sha256":receipt["original_runtime_binary_sha256"],
        "original_result_sha256":receipt["original_files_sha256"]["result.json"],
        "recovery_harness_commit":receipt["recovery_source_commit"],"result_identity":result.identity()})).unwrap());
}
