use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope, ModelTransactionEnvelope};
use eqiora::compiler::{CompiledModel, compile};
use eqiora::entity::kinds;
use eqiora::ir::ComponentScalarization;
use eqiora::kernel::pure_operator::PureOperatorDefinition;
use eqiora::kernel::typing::{ExpressionType, RootContract, SpatialSupport, TypedResidual};
use eqiora::kernel::{ExprDagBuilder, ExprNode, KernelNode, SymbolRef};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::{DimExponents, Id, ValueShape};
use eqiora_core::ValueFrame;
use serde_json::{Map, Value};

const DIRECT: &str =
    include_str!("../../../verify/language/canonical-pure-operator/models/direct.eqi");
const DIRECT_PERMUTED: &str =
    include_str!("../../../verify/language/canonical-pure-operator/models/direct-permuted.eqi");
const OPERATORS: &str =
    include_str!("../../../verify/language/canonical-pure-operator/models/operators.eqi");
const OPERATORS_PERMUTED: &str =
    include_str!("../../../verify/language/canonical-pure-operator/models/operators-permuted.eqi");
const RESOLVED: &str =
    include_str!("../../../verify/language/canonical-pure-operator/models/resolved.eqi");
const VERSION: &str = "0.1.0";
const OPERATOR_PACKAGE: &str = "Eqiora.Verify.CanonicalPureOperator";
const ROOT_PACKAGE: &str = "Eqiora.Verify.CanonicalPureOperatorRoot";

fn compile_one(file: &str, source: &str) -> CompiledModel {
    let mut compiled = compile(file, source).expect("canonical pure-operator fixture must compile");
    assert_eq!(compiled.len(), 1, "fixture must contain exactly one Model");
    compiled.pop().unwrap()
}

fn source_package(
    name: &str,
    path: &str,
    source: &str,
    dependencies: Vec<PackageDependencyV1>,
) -> PackageSourcesV1 {
    let path = NormalizedRelativePath::parse(path).expect("fixture source path");
    let manifest = PackageManifestV1::new(
        &path
            .as_str()
            .strip_prefix("src/")
            .unwrap()
            .strip_suffix(".eqi")
            .unwrap()
            .replace('/', "."),
        QualifiedName::parse(name).expect("fixture package name"),
        ExactVersion::parse(VERSION).expect("fixture package version"),
        dependencies,
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .expect("fixture manifest");
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .expect("closed fixture package")
}

fn operator_release(path: &str, source: &str) -> PackageReleaseV1 {
    prepare_package_release_v1(source_package(OPERATOR_PACKAGE, path, source, vec![]), &[])
        .expect("compiler-derived operator release")
}

fn root_release(
    operators: &PackageReleaseV1,
    alias: &str,
    source_path: &str,
    resolved_source: &str,
) -> PackageReleaseV1 {
    let dependency = PackageDependencyV1::new(
        operators
            .package_identity()
            .expect("operator package identity"),
    );
    let source = format!(
        "import {OPERATOR_PACKAGE}.operators as {alias};\n{}",
        resolved_source.replace("ops.", &format!("{alias}."))
    );
    prepare_package_release_v1(
        source_package(ROOT_PACKAGE, source_path, &source, vec![dependency]),
        std::slice::from_ref(operators),
    )
    .expect("compiler-derived root release")
}

fn compile_locked(operators: &PackageReleaseV1, root: &PackageReleaseV1) -> PackagedModelDocument {
    let resolution = ResolutionRecordV1::from_exact_releases(root, std::slice::from_ref(operators))
        .expect("exact offline resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(operators).expect("insert operator release");
    store.insert(root).expect("insert root release");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("locked V5 package compilation")
}

fn pure_definition(program: &eqiora::sem::KernelProgram) -> PureOperatorDefinition {
    let mut definitions = program.nodes().filter_map(|node| {
        let KernelNode::Relation(relation) = node else {
            return None;
        };
        relation.expression().definitions().values().next().cloned()
    });
    let definition = definitions.next().expect("one pure definition");
    assert!(
        definitions.next().is_none(),
        "exactly one Relation owns a definition"
    );
    definition
}

fn assert_one_generic_application(program: &eqiora::sem::KernelProgram) {
    let applications = program
        .nodes()
        .filter_map(|node| {
            let KernelNode::Relation(relation) = node else {
                return None;
            };
            Some(
                relation
                    .expression()
                    .nodes()
                    .iter()
                    .filter(|node| matches!(node, ExprNode::PureOperatorApplication(_)))
                    .count(),
            )
        })
        .sum::<usize>();
    assert_eq!(
        applications, 1,
        "ordinary lowering emits one generic application"
    );
}

fn first_expression_mut(value: &mut Value) -> Option<&mut Map<String, Value>> {
    match value {
        Value::Object(object) => {
            if object.contains_key("nodes") && object.contains_key("roots") {
                return Some(object);
            }
            object.values_mut().find_map(first_expression_mut)
        }
        Value::Array(values) => values.iter_mut().find_map(first_expression_mut),
        _ => None,
    }
}

#[test]
fn direct_and_exact_package_variants_share_name_free_meaning() {
    let direct =
        eqiora::api::ModelDocument::compile("direct.eqi", DIRECT).expect("direct V5 model");
    let direct_permuted =
        eqiora::api::ModelDocument::compile("elsewhere/permuted.eqi", DIRECT_PERMUTED)
            .expect("permuted direct V5 model");
    assert_eq!(
        direct.canonical_json().unwrap(),
        direct_permuted.canonical_json().unwrap(),
        "formal names, declaration order, source order, and file path are not Model meaning"
    );
    assert_one_generic_application(direct.program());
    assert_one_generic_application(direct_permuted.program());

    let operators = operator_release("src/operators.eqi", OPERATORS);
    let relocated = operator_release("src/operators.eqi", OPERATORS_PERMUTED);
    assert_eq!(
        operators.package_identity().unwrap(),
        relocated.package_identity().unwrap(),
        "formal spelling and declaration order are not package semantics"
    );
    let changed = operator_release(
        "src/operators.eqi",
        &OPERATORS.replace(
            "component(left, 0) * component(right, 1)",
            "component(right, 0) * component(left, 1)",
        ),
    );
    assert_ne!(
        operators.package_identity().unwrap(),
        changed.package_identity().unwrap(),
        "changing the exact body must change package semantic identity"
    );

    let root = root_release(&operators, "ops", "src/main.eqi", RESOLVED);
    let aliased_root = root_release(
        &relocated,
        "algebra",
        "src/main.eqi",
        &RESOLVED.replace("left = left, right = right", "a = left, b = right"),
    );
    assert_eq!(
        root.package_identity().unwrap(),
        aliased_root.package_identity().unwrap(),
        "import aliases are not root package semantics"
    );
    let packaged = compile_locked(&operators, &root);
    let packaged_aliased = compile_locked(&relocated, &aliased_root);
    assert_eq!(
        packaged.model().canonical_json().unwrap(),
        packaged_aliased.model().canonical_json().unwrap(),
        "resolved aliases cannot perturb Model V5 bytes"
    );
    assert_one_generic_application(packaged.model().program());
    assert_one_generic_application(packaged_aliased.model().program());

    let identity = pure_definition(direct.program()).digest();
    assert_eq!(
        identity,
        pure_definition(direct_permuted.program()).digest()
    );
    assert_eq!(
        identity,
        pure_definition(packaged.model().program()).digest()
    );
    assert_eq!(
        identity,
        pure_definition(packaged_aliased.model().program()).digest(),
        "local and package-resolved calls retain one content definition identity"
    );

    let model_text = String::from_utf8(direct.canonical_json().unwrap()).unwrap();
    assert!(model_text.contains("pure-operator-application"));
    assert!(model_text.contains("eqiora.pure-component-calculus/v2"));
    assert!(
        !model_text.contains("dyadic"),
        "source operator names and name recognizers cannot enter canonical Model bytes"
    );
}

#[test]
fn current_model_and_transaction_replay_exactly() {
    let compiled = compile_one("direct.eqi", DIRECT);
    let transaction = ModelTransactionEnvelope::from_transaction(compiled.transaction()).unwrap();
    let transaction_bytes = transaction.canonical_json().unwrap();
    let replayed =
        ModelTransactionEnvelope::from_json(&transaction_bytes, Default::default()).unwrap();
    assert_eq!(replayed.canonical_json().unwrap(), transaction_bytes);
    assert_eq!(
        replayed.to_transaction().unwrap().ops(),
        compiled.transaction().ops()
    );

    let document =
        eqiora::api::ModelDocument::compile("direct.eqi", DIRECT).expect("admitted current model");
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let model_bytes = model.canonical_json().unwrap();
    let replayed = ModelEnvelope::from_json(&model_bytes, Default::default()).unwrap();
    assert_eq!(replayed.canonical_json().unwrap(), model_bytes);
    assert_eq!(replayed.to_program().unwrap(), *document.program());
    assert_eq!(
        eqiora::api::ModelDocument::replay(&model_bytes)
            .unwrap()
            .canonical_json()
            .unwrap(),
        model_bytes
    );
}

#[test]
fn compiled_definition_scalarizes_as_the_exact_dyadic_map() {
    let document =
        eqiora::api::ModelDocument::compile("direct.eqi", DIRECT).expect("direct V5 model");
    let definition = pure_definition(document.program());
    let left = Id::<kinds::Field>::new();
    let right = Id::<kinds::Field>::new();
    let mut expression = ExprDagBuilder::new();
    let left_value = expression.symbol(SymbolRef::Field(left)).unwrap();
    let right_value = expression.symbol(SymbolRef::Field(right)).unwrap();
    let root = expression
        .pure_operator(&definition, [left_value, right_value])
        .unwrap();
    let expression = expression.finish([root]).unwrap();
    let support = SpatialSupport::Volume {
        domain: "body",
        dimensions: 2,
    };
    let typed = TypedResidual::infer(
        expression,
        Some(support.clone()),
        |_| None,
        RootContract::ComponentwiseResidual,
        |symbol| {
            assert!(matches!(symbol, SymbolRef::Field(field) if field == left || field == right));
            Ok::<_, ()>(
                ExpressionType::shaped(
                    DimExponents::DIMENSIONLESS,
                    ValueShape::new([2]).unwrap(),
                    ValueFrame::SpatialCartesian,
                    Some(support.clone()),
                )
                .unwrap(),
            )
        },
    )
    .expect("compiled definition instantiates under the exact vector type");
    let scalarized = ComponentScalarization::lower(&typed).unwrap();
    let values = scalarized
        .evaluate(|coordinate| match coordinate.symbol() {
            SymbolRef::Field(field) if field == left => match coordinate.component_index() {
                [0] => Some(2.0),
                [1] => Some(3.0),
                _ => None,
            },
            SymbolRef::Field(field) if field == right => match coordinate.component_index() {
                [0] => Some(5.0),
                [1] => Some(7.0),
                _ => None,
            },
            _ => None,
        })
        .unwrap();
    assert_eq!(values, [10.0, 14.0, 15.0, 21.0]);
}

#[test]
fn current_wire_unknown_feature_digest_and_resource_limit_fail_closed() {
    let document =
        eqiora::api::ModelDocument::compile("direct.eqi", DIRECT).expect("direct current model");
    let bytes = ModelEnvelope::from_program(document.program())
        .unwrap()
        .canonical_json()
        .unwrap();
    let original: Value = serde_json::from_slice(&bytes).unwrap();

    let mut unknown_feature = original.clone();
    first_expression_mut(&mut unknown_feature).unwrap()["definitions"][0]["required_features"][0] =
        Value::String("eqiora.unknown-pure-calculus/v9".to_owned());
    let diagnostic = ModelEnvelope::from_json(
        &serde_json::to_vec(&unknown_feature).unwrap(),
        Default::default(),
    )
    .unwrap_err();
    assert!(diagnostic.message().contains("unknown feature"));

    let mut mismatched_digest = original.clone();
    let digest = first_expression_mut(&mut mismatched_digest).unwrap()["definitions"][0]["digest"]
        .as_str()
        .unwrap()
        .to_owned();
    let replacement = if digest.starts_with('0') { '1' } else { '0' };
    first_expression_mut(&mut mismatched_digest).unwrap()["definitions"][0]["digest"] =
        Value::String(format!("{replacement}{}", &digest[1..]));
    let diagnostic = ModelEnvelope::from_json(
        &serde_json::to_vec(&mismatched_digest).unwrap(),
        Default::default(),
    )
    .unwrap_err();
    assert!(diagnostic.message().contains("digest mismatch"));

    let diagnostic = ModelEnvelope::from_json(
        &bytes,
        ModelDecoderLimits {
            max_pure_operator_definitions: 0,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(diagnostic.message().contains("exceeds decoder limit"));
}
