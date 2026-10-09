use std::collections::{BTreeMap, BTreeSet};

use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::diagnostic::codes;
use eqiora::kernel::BoundarySide;
use eqiora::language::{ComponentItem, DomainSyntax, Item, SignatureItem};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageCompilationRecordV2, PackageDependencyV1, PackageManifestV1, PackageReleaseV1,
    PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::sem::KernelProgram;
use eqiora_numerics::{
    common::ScalarSpatialExpression, fluid::SteadyIncompressibleStokesCartesianModel2d,
    fluid::lower_steady_incompressible_stokes_cartesian_2d,
};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const COMPONENT: &str =
    include_str!("../../../verify/fluid/packaged-steady-stokes-2d/models/component.eqi");
const COMPONENT_README: &[u8] =
    include_bytes!("../../../verify/fluid/packaged-steady-stokes-2d/package-v0.1.0/README.md");
const COMPONENT_PERMUTED: &str =
    include_str!("../../../verify/fluid/packaged-steady-stokes-2d/models/component-permuted.eqi");
const PACKAGED: &str =
    include_str!("../../../verify/fluid/packaged-steady-stokes-2d/models/packaged.eqi");
const PACKAGED_PERMUTED: &str =
    include_str!("../../../verify/fluid/packaged-steady-stokes-2d/models/packaged-permuted.eqi");
const DIRECT: &str =
    include_str!("../../../verify/fluid/packaged-steady-stokes-2d/models/direct.eqi");

const PUBLIC_PACKAGE: &str = "Eqiora.Fluid.Incompressible";
const ROOT_PACKAGE: &str = "org.eqiora.verify.packaged_steady_stokes_2d";
const VERSION: &str = "0.1.0";

const PACKAGED_TO_DIRECT: [(&str, &str); 18] = [
    ("body", "body"),
    ("x_lower", "x_lower"),
    ("x_upper", "x_upper"),
    ("y_lower", "y_lower"),
    ("y_upper", "y_upper"),
    ("velocity", "velocity"),
    ("pressure", "pressure"),
    ("force_potential", "force_potential"),
    ("fluid_law.dynamic_viscosity", "dynamic_viscosity"),
    ("wave_number", "wave_number"),
    ("force_scale", "force_scale"),
    ("force_definition", "force_definition"),
    ("fluid_law.momentum", "momentum"),
    ("fluid_law.incompressibility", "incompressibility"),
    ("x_lower_value", "x_lower_value"),
    ("x_upper_value", "x_upper_value"),
    ("y_lower_value", "y_lower_value"),
    ("y_upper_value", "y_upper_value"),
];

fn component_release() -> PackageReleaseV1 {
    let sources = embedded_package::generated_sources(
        PUBLIC_PACKAGE,
        VERSION,
        &[
            ("README.md", BundleRoleV1::Documentation, COMPONENT_README),
            (
                "src/incompressible.eqi",
                BundleRoleV1::ModelSource,
                COMPONENT.as_bytes(),
            ),
        ],
    );
    prepare_package_release_v1(sources, &[]).expect("prepare exact fluid package release")
}

fn synthetic_component_release(
    package_name: &str,
    source: &str,
    reverse_files: bool,
) -> PackageReleaseV1 {
    let readme_path = NormalizedRelativePath::parse("README.md").expect("README path");
    let model_path =
        NormalizedRelativePath::parse("src/incompressible.eqi").expect("component source path");
    let manifest = PackageManifestV1::new(
        "incompressible",
        QualifiedName::parse(package_name).expect("component package name"),
        ExactVersion::parse(VERSION).expect("component package version"),
        Vec::new(),
        vec![
            BundleEntryV1::new(readme_path.clone(), BundleRoleV1::Documentation),
            BundleEntryV1::new(model_path.clone(), BundleRoleV1::ModelSource),
        ],
    )
    .expect("component manifest");
    let mut files = vec![
        SourceFileV1::new(
            readme_path,
            BundleRoleV1::Documentation,
            COMPONENT_README.to_vec(),
        ),
        SourceFileV1::new(
            model_path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        ),
    ];
    if reverse_files {
        files.reverse();
    }
    let sources = PackageSourcesV1::new(manifest, files).expect("closed component sources");
    prepare_package_release_v1(sources, &[]).expect("prepare synthetic fluid provider")
}

fn root_sources(
    component: &PackageReleaseV1,
    alias: &str,
    source: &str,
    reverse_files: bool,
) -> PackageSourcesV1 {
    let readme_path = NormalizedRelativePath::parse("README.md").expect("root README path");
    let model_path = NormalizedRelativePath::parse("src/main.eqi").expect("root source path");
    let dependency = PackageDependencyV1::new(
        component
            .package_identity()
            .expect("component package identity"),
    );
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(ROOT_PACKAGE).expect("root package name"),
        ExactVersion::parse(VERSION).expect("root version"),
        vec![dependency],
        vec![
            BundleEntryV1::new(readme_path.clone(), BundleRoleV1::Documentation),
            BundleEntryV1::new(model_path.clone(), BundleRoleV1::ModelSource),
        ],
    )
    .expect("root manifest");
    let package_name = component
        .package_identity()
        .expect("component package identity")
        .name;
    let source = format!(
        "import {package_name}.incompressible as {alias};\n{}",
        source.replace("fluid.", &format!("{alias}."))
    );
    let mut files = vec![
        SourceFileV1::new(
            readme_path,
            BundleRoleV1::Documentation,
            b"Exact packaged steady-Stokes verification root.\n".to_vec(),
        ),
        SourceFileV1::new(model_path, BundleRoleV1::ModelSource, source.into_bytes()),
    ];
    if reverse_files {
        files.reverse();
    }
    PackageSourcesV1::new(manifest, files).expect("closed root author sources")
}

fn root_release(
    component: &PackageReleaseV1,
    alias: &str,
    source: &str,
    reverse_files: bool,
) -> PackageReleaseV1 {
    prepare_package_release_v1(
        root_sources(component, alias, source, reverse_files),
        std::slice::from_ref(component),
    )
    .expect("prepare exact packaged Stokes root")
}

fn compile_locked(
    component: &PackageReleaseV1,
    root: &PackageReleaseV1,
) -> (PackagedModelDocument, ResolutionRecordV1) {
    let resolution = ResolutionRecordV1::from_exact_releases(root, std::slice::from_ref(component))
        .expect("exact two-package resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(component).expect("insert component release");
    store.insert(root).expect("insert root release");
    let packaged = PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile exact packaged Stokes Model");
    packaged
        .compilation()
        .validate_against(&resolution)
        .expect("compilation matches exact resolution");
    (packaged, resolution)
}

fn direct_document(source: &str) -> ModelDocument {
    eqiora::api::ModelDocument::compile("direct.eqi", source)
        .expect("direct Stokes source compiles")
}

fn id_ulid(value: &serde_json::Value) -> String {
    value["ulid"].as_str().expect("typed ID ULID").to_owned()
}

fn activations_by_relation(model: &serde_json::Value) -> BTreeMap<String, String> {
    let mut activations = BTreeMap::new();
    for edge in model["edges"].as_array().expect("edge array") {
        if edge["kind"].as_str() == Some("activates") {
            assert!(
                activations
                    .insert(id_ulid(&edge["to"]), id_ulid(&edge["from"]))
                    .is_none()
            );
        }
    }
    activations
}

fn collect_model_ulids(model: &serde_json::Value) -> BTreeSet<String> {
    fn collect(value: &serde_json::Value, identities: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(object) => {
                if let Some(ulid) = object.get("ulid") {
                    identities.insert(ulid.as_str().expect("typed ID ULID").to_owned());
                }
                for child in object.values() {
                    collect(child, identities);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    collect(child, identities);
                }
            }
            _ => {}
        }
    }

    let mut identities =
        BTreeSet::from([model["model_ulid"].as_str().expect("Model ULID").to_owned()]);
    collect(model, &mut identities);
    identities
}

fn rewrite_model_ulids(value: &mut serde_json::Value, identities: &BTreeMap<String, String>) {
    match value {
        serde_json::Value::Object(object) => {
            for key in ["model_ulid", "ulid"] {
                if let Some(value) = object.get_mut(key) {
                    let source = value.as_str().expect("ULID string");
                    *value = identities
                        .get(source)
                        .unwrap_or_else(|| panic!("unmapped semantic identity `{source}`"))
                        .clone()
                        .into();
                }
            }
            for child in object.values_mut() {
                rewrite_model_ulids(child, identities);
            }
        }
        serde_json::Value::Array(values) => {
            for child in values {
                rewrite_model_ulids(child, identities);
            }
        }
        _ => {}
    }
}

fn first_json_difference(
    left: &serde_json::Value,
    right: &serde_json::Value,
    path: &str,
) -> Option<String> {
    match (left, right) {
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => left
            .keys()
            .chain(right.keys())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .find_map(|key| match (left.get(key), right.get(key)) {
                (Some(left), Some(right)) => {
                    first_json_difference(left, right, &format!("{path}.{key}"))
                }
                values => Some(format!("{path}.{key}: {values:?}")),
            }),
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            if left.len() != right.len() {
                return Some(format!("{path}.len: {} != {}", left.len(), right.len()));
            }
            left.iter()
                .zip(right)
                .enumerate()
                .find_map(|(index, (left, right))| {
                    first_json_difference(left, right, &format!("{path}[{index}]"))
                })
        }
        _ if left == right => None,
        _ => Some(format!("{path}: {left:?} != {right:?}")),
    }
}

fn identity_normalized_program(packaged: &ModelDocument, direct: &ModelDocument) -> KernelProgram {
    let packaged_envelope =
        ModelEnvelope::from_program(packaged.program()).expect("packaged current Model");
    let direct_envelope =
        ModelEnvelope::from_program(direct.program()).expect("direct current Model");
    let mut packaged_value: serde_json::Value = serde_json::from_slice(
        &packaged_envelope
            .canonical_json()
            .expect("packaged canonical Model"),
    )
    .expect("packaged Model JSON");
    let direct_value: serde_json::Value = serde_json::from_slice(
        &direct_envelope
            .canonical_json()
            .expect("direct canonical Model"),
    )
    .expect("direct Model JSON");

    let mut identities = BTreeMap::from([(
        packaged.program().model().ulid().to_string(),
        direct.program().model().ulid().to_string(),
    )]);
    // One common spatial support synthesizes one continuum owner with no source alias.
    let continuum = |model: &serde_json::Value| {
        let nodes = model["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|node| node["definition"]["kind"] == "representation")
            .collect::<Vec<_>>();
        assert_eq!(nodes.len(), 1);
        id_ulid(&nodes[0]["id"])
    };
    identities.insert(continuum(&packaged_value), continuum(&direct_value));
    for (packaged_name, direct_name) in PACKAGED_TO_DIRECT {
        let packaged_id = packaged
            .aliases()
            .get(packaged_name)
            .unwrap_or_else(|| panic!("packaged symbol `{packaged_name}`"));
        let direct_id = direct
            .aliases()
            .get(direct_name)
            .unwrap_or_else(|| panic!("direct symbol `{direct_name}`"));
        assert!(
            identities
                .insert(packaged_id.ulid().to_string(), direct_id.ulid().to_string())
                .is_none()
        );
    }
    for (packaged_relation, packaged_activation) in activations_by_relation(&packaged_value) {
        let direct_relation = identities
            .get(&packaged_relation)
            .expect("mapped Relation identity");
        let direct_activation = activations_by_relation(&direct_value)
            .remove(direct_relation)
            .expect("corresponding direct Activation");
        assert!(
            identities
                .insert(packaged_activation, direct_activation)
                .is_none()
        );
    }
    assert_eq!(
        identities.keys().cloned().collect::<BTreeSet<_>>(),
        collect_model_ulids(&packaged_value),
        "normalization covers every packaged semantic identity"
    );
    assert_eq!(
        identities.values().cloned().collect::<BTreeSet<_>>(),
        collect_model_ulids(&direct_value),
        "normalization is a complete direct-Model identity bijection"
    );

    rewrite_model_ulids(&mut packaged_value, &identities);
    let rewritten = ModelEnvelope::from_json(
        &serde_json::to_vec(&packaged_value).expect("normalized Model JSON"),
        Default::default(),
    )
    .expect("normalized current Model");
    rewritten
        .to_program()
        .expect("normalized Model reconstructs exactly")
}

fn assert_same_lowered_stokes(
    left: &SteadyIncompressibleStokesCartesianModel2d,
    right: &SteadyIncompressibleStokesCartesianModel2d,
) {
    assert_eq!(left.domain(), right.domain());
    assert_eq!(left.velocity(), right.velocity());
    assert_eq!(left.pressure(), right.pressure());
    assert_eq!(left.force_potential(), right.force_potential());
    assert_eq!(left.bounds(), right.bounds());
    assert_eq!(
        left.force_potential_definition(),
        right.force_potential_definition()
    );
    assert_eq!(left.momentum_relation(), right.momentum_relation());
    assert_eq!(
        left.incompressibility_relation(),
        right.incompressibility_relation()
    );
    assert_eq!(left.boundary_inventory(), right.boundary_inventory());
    assert_same_expression_action(
        left.dynamic_viscosity_expression(),
        right.dynamic_viscosity_expression(),
    );
    assert_same_expression_action(
        left.force_potential_expression(),
        right.force_potential_expression(),
    );
}

fn assert_same_expression_action(
    left: &ScalarSpatialExpression<f64>,
    right: &ScalarSpatialExpression<f64>,
) {
    assert_eq!(left.coordinate_dimension(), right.coordinate_dimension());
    assert_eq!(left.parameter_fields(), right.parameter_fields());
    assert_eq!(left.parameter_values(), right.parameter_values());
    let coordinates = [0.25, 0.125];
    let coordinate_tangent = [0.375, -0.25];
    let parameter_tangent = (0..left.parameter_fields().len())
        .map(|index| (index + 1) as f64 * 0.5)
        .collect::<Vec<_>>();
    assert_eq!(left.evaluate(&coordinates), right.evaluate(&coordinates));
    assert_eq!(
        left.evaluate_jvp(&coordinates, &coordinate_tangent, &parameter_tangent),
        right.evaluate_jvp(&coordinates, &coordinate_tangent, &parameter_tangent)
    );
    assert_eq!(
        left.evaluate_vjp(&coordinates, 1.25),
        right.evaluate_vjp(&coordinates, 1.25)
    );
}

fn assert_component_and_root_boundaries() {
    let component = eqiora::language::parse("incompressible.eqi", COMPONENT)
        .into_document()
        .expect("component source parses");
    assert!(component.connectors().is_empty());
    assert!(component.models().is_empty());
    assert_eq!(component.components().len(), 1);
    let component = &component.components()[0];
    assert_eq!(component.name(), "SteadyStokesWithPotential2d");
    assert_eq!(component.signature().len() + component.items().len(), 7);
    assert_eq!(
        component
            .signature()
            .iter()
            .filter(|item| matches!(item, SignatureItem::Support(_)))
            .count(),
        1
    );
    assert_eq!(
        component
            .signature()
            .iter()
            .filter(|item| matches!(item, SignatureItem::Field(_)))
            .count(),
        3
    );
    assert_eq!(
        component
            .signature()
            .iter()
            .filter(|item| matches!(item, SignatureItem::Parameter(_)))
            .count(),
        1
    );
    assert_eq!(
        component
            .items()
            .iter()
            .filter(|item| matches!(item, ComponentItem::Relation(_)))
            .count(),
        2
    );

    let root = eqiora::language::parse("packaged.eqi", PACKAGED)
        .into_document()
        .expect("root source parses");
    assert!(root.connectors().is_empty());
    assert!(root.components().is_empty());
    assert_eq!(root.models().len(), 1);
    let items = root.models()[0].items();
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, Item::Domain(_)))
            .count(),
        5
    );
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, Item::Domain(domain) if matches!(domain.syntax(), DomainSyntax::Boundary { .. })))
            .count(),
        4
    );
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, Item::Field(_)))
            .count(),
        3
    );
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, Item::Relation(_)))
            .count(),
        5,
        "the root owns one force definition and four zero traces"
    );
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, Item::Instance(_)))
            .count(),
        1
    );
}

fn assert_lowering_rejects(source: &str) {
    let direct = direct_document(source);
    let diagnostic = lower_steady_incompressible_stokes_cartesian_2d(direct.program())
        .expect_err("near-miss Stokes semantics must fail closed");
    assert_eq!(diagnostic.code(), codes::INVALID_SPATIAL_LOWERING);
}

fn assert_model_or_lowering_rejects(source: &str) {
    match eqiora::api::ModelDocument::compile("near-miss.eqi", source) {
        Err(diagnostics) => assert!(!diagnostics.is_empty()),
        Ok(document) => {
            let diagnostic = lower_steady_incompressible_stokes_cartesian_2d(document.program())
                .expect_err("near-miss Stokes semantics must fail closed");
            assert_eq!(diagnostic.code(), codes::INVALID_SPATIAL_LOWERING);
        }
    }
}

#[test]
fn exact_package_identity_and_lowered_meaning_are_name_and_order_independent() {
    assert_component_and_root_boundaries();
    let component = component_release();
    assert_eq!(
        component
            .package_identity()
            .expect("public package identity")
            .name
            .as_str(),
        PUBLIC_PACKAGE
    );
    let permuted_component = synthetic_component_release(PUBLIC_PACKAGE, COMPONENT_PERMUTED, true);
    assert_eq!(
        component
            .package_identity()
            .expect("public package identity"),
        permuted_component
            .package_identity()
            .expect("permuted package identity"),
        "declaration and file order are non-semantic"
    );
    assert_ne!(
        component.source_digest().expect("public source digest"),
        permuted_component
            .source_digest()
            .expect("permuted source digest")
    );

    let root = root_release(&component, "fluid", PACKAGED, false);
    let permuted_root = root_release(
        &permuted_component,
        "constitutive_provider",
        PACKAGED_PERMUTED,
        true,
    );
    assert_eq!(
        root.package_identity().expect("root identity"),
        permuted_root
            .package_identity()
            .expect("permuted root identity"),
        "import alias, declaration, binding, and file order are non-semantic"
    );
    let (packaged, _) = compile_locked(&component, &root);
    let (permuted, _) = compile_locked(&permuted_component, &permuted_root);
    assert_eq!(
        packaged.model().canonical_json().expect("packaged Model"),
        permuted.model().canonical_json().expect("permuted Model")
    );

    let direct = direct_document(DIRECT);
    let normalized = identity_normalized_program(packaged.model(), &direct);
    let normalized_lowered = lower_steady_incompressible_stokes_cartesian_2d(&normalized)
        .expect("normalized packaged meaning lowers");
    let direct_lowered = lower_steady_incompressible_stokes_cartesian_2d(direct.program())
        .expect("direct meaning lowers");
    assert_same_lowered_stokes(&normalized_lowered, &direct_lowered);
    assert_eq!(direct_lowered.bounds(), &[[0.0, 1.0], [0.0, 1.0]]);
    assert_eq!(direct_lowered.dynamic_viscosity(), 2.0);
    for (axis, side) in [
        (0, BoundarySide::Lower),
        (0, BoundarySide::Upper),
        (1, BoundarySide::Lower),
        (1, BoundarySide::Upper),
    ] {
        assert!(matches!(
            direct_lowered
                .boundary_inventory()
                .boundary(axis, side)
                .expect("complete Stokes boundary inventory")
                .disposition(),
            eqiora_numerics::common::PhysicalBoundaryDisposition::TraceZero
        ));
    }
    let force = direct_lowered.force_potential_expression();
    assert_eq!(force.evaluate(&[0.0, 0.0]).expect("origin force"), 0.0);
    assert!((force.evaluate(&[0.25, 0.0]).expect("nonconstant force") - 1.0).abs() < 1.0e-15);
    assert!((force.evaluate(&[0.25, 0.25]).expect("peak force") - 2.0).abs() < 2.0e-15);
    let (_, x_derivative) = force
        .evaluate_jvp(&[0.0, 0.0], &[1.0, 0.0], &[0.0, 0.0])
        .expect("force-potential coordinate derivative");
    assert!((x_derivative - std::f64::consts::TAU).abs() < 1.0e-14);

    let renamed_component =
        synthetic_component_release("org.example.incompressible", COMPONENT, true);
    assert_ne!(
        renamed_component
            .package_identity()
            .expect("renamed provider identity"),
        component
            .package_identity()
            .expect("public provider identity")
    );
    let renamed_root = root_release(&renamed_component, "renamed_provider", PACKAGED, false);
    let (renamed, _) = compile_locked(&renamed_component, &renamed_root);
    let renamed_normalized = identity_normalized_program(renamed.model(), &direct);
    let renamed_lowered = lower_steady_incompressible_stokes_cartesian_2d(&renamed_normalized)
        .expect("renamed provider lowers");
    assert_same_lowered_stokes(&renamed_lowered, &direct_lowered);
}

#[test]
fn literal_component_coefficient_lowers_without_fabricating_a_parameter() {
    let component = component_release();
    let literal_source = PACKAGED
        .replace("  parameter dynamic_viscosity: kg / (m * s) = 2;\n", "")
        .replace(
            "dynamic_viscosity = dynamic_viscosity",
            "dynamic_viscosity = 2[kg / (m * s)]",
        );
    let root = root_release(&component, "fluid", &literal_source, false);
    let (packaged, _) = compile_locked(&component, &root);
    assert!(
        packaged
            .model()
            .aliases()
            .get("fluid_law.dynamic_viscosity")
            .is_none(),
        "a literal component argument is a typed lexical constant"
    );

    let lowered = lower_steady_incompressible_stokes_cartesian_2d(packaged.model().program())
        .expect("literal dynamic viscosity lowers through the same recognizer");
    assert_eq!(lowered.dynamic_viscosity(), 2.0);
    assert!(
        lowered
            .dynamic_viscosity_expression()
            .parameter_fields()
            .is_empty()
    );

    let changed_source = literal_source.replace("dynamic_viscosity = 2[", "dynamic_viscosity = 3[");
    let changed_root = root_release(&component, "fluid", &changed_source, false);
    let (changed, _) = compile_locked(&component, &changed_root);
    let changed_coefficient =
        lower_steady_incompressible_stokes_cartesian_2d(changed.model().program())
            .expect("a changed literal recompiles as changed immutable meaning");
    assert_eq!(changed_coefficient.dynamic_viscosity(), 3.0);
    assert_ne!(packaged.model().digest(), changed.model().digest());

    let arithmetic_source = PACKAGED.replace(
        "dynamic_viscosity = dynamic_viscosity",
        "dynamic_viscosity = 2 * dynamic_viscosity",
    );
    let arithmetic_root = root_release(&component, "fluid", &arithmetic_source, false);
    let (arithmetic, _) = compile_locked(&component, &arithmetic_root);
    let arithmetic_coefficient =
        lower_steady_incompressible_stokes_cartesian_2d(arithmetic.model().program())
            .expect("a coefficient may itself contain the dimensionless factor two");
    assert_eq!(arithmetic_coefficient.dynamic_viscosity(), 4.0);
    assert_eq!(
        arithmetic_coefficient
            .dynamic_viscosity_expression()
            .parameter_fields(),
        &[arithmetic.model().aliases()["dynamic_viscosity"]
            .downcast::<eqiora::entity::kinds::Parameter>()
            .unwrap()]
    );
    let arithmetic_expression = arithmetic_coefficient.dynamic_viscosity_expression();
    assert_eq!(
        arithmetic_expression
            .evaluate_parameter_jvp(&[0.0, 0.0], &[1.0])
            .unwrap(),
        (4.0, 2.0),
        "the derived binding retains the analytic forward chain rule"
    );
    assert_eq!(
        arithmetic_expression
            .evaluate_parameter_vjp(&[0.0, 0.0], 1.75)
            .unwrap(),
        (4.0, vec![3.5]),
        "the derived binding retains the analytic reverse chain rule"
    );

    let positive_zero_source = literal_source.replace(
        "parameter force_scale: kg / (m * s ^ 2) = 1;",
        "parameter force_scale: kg / (m * s ^ 2) = 0.0;",
    );
    let negative_zero_source = positive_zero_source.replace(
        "parameter force_scale: kg / (m * s ^ 2) = 0.0;",
        "parameter force_scale: kg / (m * s ^ 2) = -0.0;",
    );
    let positive_zero_root = root_release(&component, "fluid", &positive_zero_source, false);
    let negative_zero_root = root_release(&component, "fluid", &negative_zero_source, false);
    assert_eq!(
        positive_zero_root.package_identity().unwrap(),
        negative_zero_root.package_identity().unwrap(),
        "positive and negative zero have one semantic package identity"
    );
    assert_ne!(
        positive_zero_root.source_digest().unwrap(),
        negative_zero_root.source_digest().unwrap(),
        "raw source identity remains an exact byte claim"
    );
    let (positive_zero, _) = compile_locked(&component, &positive_zero_root);
    let (negative_zero, _) = compile_locked(&component, &negative_zero_root);
    let positive_bytes = positive_zero.model().canonical_json().unwrap();
    let negative_bytes = negative_zero.model().canonical_json().unwrap();
    if positive_bytes != negative_bytes {
        let byte_index = positive_bytes
            .iter()
            .zip(&negative_bytes)
            .position(|(positive, negative)| positive != negative)
            .unwrap_or_else(|| positive_bytes.len().min(negative_bytes.len()));
        let context_start = byte_index.saturating_sub(48);
        let positive_end = (byte_index + 48).min(positive_bytes.len());
        let negative_end = (byte_index + 48).min(negative_bytes.len());
        let positive_context =
            String::from_utf8_lossy(&positive_bytes[context_start..positive_end]);
        let negative_context =
            String::from_utf8_lossy(&negative_bytes[context_start..negative_end]);
        let positive_value = serde_json::from_slice(&positive_bytes).unwrap();
        let negative_value = serde_json::from_slice(&negative_bytes).unwrap();
        let difference = first_json_difference(&positive_value, &negative_value, "$")
            .unwrap_or_else(|| format!("byte {byte_index}"));
        panic!(
            "positive/negative-zero Models differ at {difference}:\npositive: {positive_context}\nnegative: {negative_context}"
        );
    }
}

#[test]
fn exact_offline_release_resolution_compilation_and_model_replay() {
    let component = component_release();
    let root = root_release(&component, "fluid", PACKAGED, false);
    let (packaged, resolution) = compile_locked(&component, &root);

    let component_bytes = component.canonical_json().expect("component release bytes");
    let component_replay =
        PackageReleaseV1::from_json(&component_bytes).expect("component release replay");
    assert_eq!(
        component_replay.canonical_json().expect("replayed release"),
        component_bytes
    );
    assert_eq!(
        component_replay
            .package_identity()
            .expect("replayed identity"),
        component.package_identity().expect("original identity")
    );

    let root_bytes = root.canonical_json().expect("root release bytes");
    let root_replay = PackageReleaseV1::from_json(&root_bytes).expect("root release replay");
    let resolution_bytes = resolution.canonical_json().expect("resolution bytes");
    let resolution_replay =
        ResolutionRecordV1::from_json(&resolution_bytes).expect("resolution replay");
    assert_eq!(
        resolution_replay
            .canonical_json()
            .expect("replayed resolution"),
        resolution_bytes
    );

    let compilation_bytes = packaged
        .compilation()
        .canonical_json()
        .expect("compilation bytes");
    let compilation_replay =
        PackageCompilationRecordV2::from_json(&compilation_bytes).expect("compilation replay");
    compilation_replay
        .validate_against(&resolution_replay)
        .expect("replayed compilation matches replayed resolution");

    let mut replay_store = InMemoryPackageStore::default();
    replay_store
        .insert(&component_replay)
        .expect("install replayed component");
    replay_store
        .insert(&root_replay)
        .expect("install replayed root");
    let recompiled =
        PackagedModelDocument::compile_locked(&replay_store, &resolution_replay, "Main")
            .expect("offline recompile from replayed exact releases");
    assert_eq!(
        recompiled
            .compilation()
            .canonical_json()
            .expect("recompiled record"),
        compilation_bytes
    );
    let model_bytes = packaged
        .model()
        .canonical_json()
        .expect("current Model bytes");
    assert_eq!(
        recompiled
            .model()
            .canonical_json()
            .expect("recompiled Model"),
        model_bytes
    );
    let model_replay =
        eqiora::api::ModelDocument::replay(&model_bytes).expect("current Model replay");
    assert_eq!(
        model_replay.canonical_json().expect("replayed Model"),
        model_bytes
    );
}

#[test]
fn canonical_stokes_recognizer_rejects_semantic_near_misses() {
    assert_lowering_rejects(&DIRECT.replace(
        "parameter dynamic_viscosity: kg / (m * s) = 2;",
        "parameter dynamic_viscosity: kg / (m * s) = 0;",
    ));
    assert_lowering_rejects(&DIRECT.replace("symmetric_part(grad(velocity))", "grad(velocity)"));
    assert_lowering_rejects(
        &DIRECT.replace("- isotropic_lift(pressure)", "+ isotropic_lift(pressure)"),
    );
    assert_lowering_rejects(&DIRECT.replace(
        "div(velocity) = 0;",
        "div(velocity) - pressure / dynamic_viscosity = 0;",
    ));
    assert_lowering_rejects(&DIRECT.replace(
        "  relation y_upper_value on y_upper { trace(velocity) = 0; }\n",
        "",
    ));

    let extra_relation = DIRECT.replace(
        "  relation x_lower_value on x_lower",
        "  relation redundant_pressure on body { pressure - pressure = 0; }\n\n  relation x_lower_value on x_lower",
    );
    assert_lowering_rejects(&extra_relation);

    let wrong_dimensions = DIRECT
        .replace(
            "variable velocity: vector<m / s, 2> on body in h1;",
            "variable velocity: vector<1, 2> on body in h1;",
        )
        .replace(
            "variable pressure: kg / (m * s ^ 2) on body;",
            "variable pressure: 1 / m on body;",
        )
        .replace(
            "variable force_potential: kg / (m * s ^ 2) on body;",
            "variable force_potential: 1 / m on body;",
        )
        .replace(
            "parameter dynamic_viscosity: kg / (m * s) = 2;",
            "parameter dynamic_viscosity: 1 = 2;",
        )
        .replace(
            "parameter force_scale: kg / (m * s ^ 2) = 1;",
            "parameter force_scale: 1 / m = 1;",
        );
    assert_lowering_rejects(&wrong_dimensions);

    let distinct_support = DIRECT
        .replace(
            "model Main() {",
            "model Main() { domain peer = box(0, 1, 0, 1);",
        )
        .replace(
            "variable pressure: kg / (m * s ^ 2) on body;",
            "variable pressure: kg / (m * s ^ 2) on peer;",
        );
    assert_model_or_lowering_rejects(&distinct_support);

    let component = component_release();
    let misbound = PACKAGED.replace("pressure = pressure,", "pressure = force_potential,");
    let root = root_release(&component, "fluid", &misbound, false);
    let (packaged, _) = compile_locked(&component, &root);
    let diagnostic = lower_steady_incompressible_stokes_cartesian_2d(packaged.model().program())
        .expect_err("pressure and force-potential slots must bind distinct Fields");
    assert_eq!(diagnostic.code(), codes::INVALID_SPATIAL_LOWERING);

    assert_lowering_rejects(&DIRECT.replace(
        "2 * dynamic_viscosity * symmetric_part(grad(velocity))",
        "dynamic_viscosity * symmetric_part(grad(velocity))",
    ));
}
