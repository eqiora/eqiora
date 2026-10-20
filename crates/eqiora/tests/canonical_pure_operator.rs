#[path = "support/exact_package.rs"]
#[allow(dead_code)] // Integration-test crates consume disjoint shared fixtures.
mod exact_package;

use eqiora::artifact::{ModelDecoderLimits, ModelEnvelope, ModelTransactionEnvelope};
use eqiora::compiler::{CompiledModel, compile};
use eqiora::entity::kinds;
use eqiora::ir::ComponentScalarization;
use eqiora::kernel::pure_operator::PureOperatorDefinition;
use eqiora::kernel::typing::{ExpressionType, RootContract, SpatialSupport, TypedResidual};
use eqiora::kernel::{ExprDagBuilder, ExprNode, KernelNode, SymbolRef};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, NormalizedRelativePath, PackageDependencyV1,
    PackageManifestV1, PackageReleaseV1, PackageSourcesV1, PackagedModelDocument, QualifiedName,
    SourceFileV1, prepare_package_release_v1,
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
    let (store, resolution) = exact_package::resolved_store(root, std::slice::from_ref(operators));
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
    assert!(model_text.contains("eqiora.pure-component-calculus/v5"));
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

#[test]
fn composed_complex_embedding_preserves_units_branch_and_replay() {
    use eqiora::kernel::pure_operator::{CalculusBuilder, CalculusNode, PureValueClass};
    use eqiora::{ScalarDomain, ValueLiteral, ValueType};
    let source = r#"
operator embed(input x:m^2):complex<m^2> = x;
operator response(input x:m^2):complex<m> = math.sqrt(embed(x=x));
model Main() {
    variable x:m^2;
    variable y:complex<m>;
    relation law { y=response(x=x); }
}
"#;
    let document = eqiora::api::ModelDocument::compile("embedding.eqi", source).unwrap();
    let replay = eqiora::api::ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let input_type = ValueType::scalar(ScalarDomain::Real, length.pow(2, 1).unwrap()).unwrap();
    let output_type = ValueType::scalar(ScalarDomain::Complex, length).unwrap();
    for program in [document.program(), replay.program()] {
        let definition = pure_definition(program);
        let mut derivative = CalculusBuilder::new(
            [PureValueClass::from_value_type(input_type.clone()).unwrap()],
            PureValueClass::from_value_type(
                output_type
                    .clone()
                    .with_dimension(length.pow(-1, 1).unwrap())
                    .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let input = derivative
            .push(CalculusNode::FormalComponent {
                formal: 0,
                axes: Box::new([]),
            })
            .unwrap();
        let root = derivative
            .apply_scalar_inputs(&definition, &[input])
            .unwrap();
        let root = derivative.partial(root, 0).unwrap();
        let derivative = derivative.finish(root).unwrap();
        // Principal sqrt(x+0i): sqrt(-4)=+2i, sqrt(9)=3.
        // Away from the principal cut the derivative is 1/(2 sqrt(x+0i)).
        // The existing complex derivative contract rejects the cut; a derivative
        // constrained to a real path along that cut is not supplied by this test.
        for (x, expected, slope) in [(-4., (0., 2.), None), (9., (3., 0.), Some((1. / 6., 0.)))] {
            let input = ValueLiteral::from_real(input_type.clone(), x).unwrap();
            let output =
                ComponentScalarization::evaluate_pure_operator(&definition, &[&input]).unwrap();
            assert_eq!(output.value_type(), &output_type);
            let (re, im) = output.components().unwrap().next().unwrap();
            assert!((re - expected.0).abs() < 1e-14 && (im - expected.1).abs() < 1e-14);
            let output = ComponentScalarization::evaluate_pure_operator(&derivative, &[&input]);
            let Some(slope) = slope else {
                assert!(
                    output
                        .unwrap_err()
                        .message()
                        .contains("principal branch cut")
                );
                continue;
            };
            let output = output.unwrap();
            assert_eq!(output.value_type().dimension(), length.pow(-1, 1).unwrap());
            let (re, im) = output.components().unwrap().next().unwrap();
            assert!((re - slope.0).abs() < 1e-14 && (im - slope.1).abs() < 1e-14);
        }
    }
    let narrowing = source.replace(
        "complex<m> = math.sqrt(embed(x=x))",
        "m = math.sqrt(embed(x=x))",
    );
    assert!(eqiora::api::ModelDocument::compile("narrowing.eqi", &narrowing).is_err());
}

#[test]
fn exact_array_signatures_compile_replay_and_execute_through_component_ir() {
    use eqiora::{ScalarDomain, ValueLiteral, ValueType};
    let fixtures = [
        (
            "operator swap_axes(input x:array<array<1,3>,2>):array<array<1,2>,3> = component(x,1,0); model Main(){variable x:array<array<1,3>,2>; variable y:array<array<1,2>,3>; relation r{y=swap_axes(x=x);}}",
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                .unwrap()
                .array(3)
                .unwrap()
                .array(2)
                .unwrap(),
            vec![(1., 0.), (2., 0.), (3., 0.), (4., 0.), (5., 0.), (6., 0.)],
            vec![(1., 0.), (4., 0.), (2., 0.), (5., 0.), (3., 0.), (6., 0.)],
        ),
        (
            "operator project(input x:array<complex<1>,3>):array<1,3> = math.real(component(x,0)) + math.imag(component(x,0)); model Main(){variable x:array<complex<1>,3>; variable y:array<1,3>; relation r{y=project(x=x);}}",
            ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .unwrap()
                .array(3)
                .unwrap(),
            vec![(1., 2.), (3., 4.), (5., 6.)],
            vec![(3., 0.), (7., 0.), (11., 0.)],
        ),
    ];
    for (source, value_type, components, expected) in fixtures {
        let document = eqiora::api::ModelDocument::compile("typed.eqi", source).unwrap();
        let bytes = document.canonical_json().unwrap();
        let replay = eqiora::api::ModelDocument::replay(&bytes).unwrap();
        assert_eq!(replay.canonical_json().unwrap(), bytes);
        for program in [document.program(), replay.program()] {
            let definition = pure_definition(program);
            let input = ValueLiteral::new(value_type.clone(), components.clone()).unwrap();
            let output =
                ComponentScalarization::evaluate_pure_operator(&definition, &[&input]).unwrap();
            assert_eq!(output.components().unwrap().collect::<Vec<_>>(), expected);
            assert_eq!(
                output.value_type(),
                definition.result_rule().exact_type().unwrap()
            );
        }
    }
}

#[test]
fn exact_nominal_signatures_bind_before_component_lowering() {
    let source = r#"
space A = orthonormal(a,b);
space B = orthonormal(c,d);
operator retain(input x:coordinates<complex<1>,A>):coordinates<complex<1>,A>
    = component(x,0);
operator retain_map(input x:map<complex<1>,A,B>):map<complex<1>,A,B>
    = component(x,0,1);
model Main() {
    variable x:coordinates<complex<1>,A>;
    variable y:coordinates<complex<1>,A>;
    variable m:map<complex<1>,A,B>;
    variable n:map<complex<1>,A,B>;
    relation r { y = retain(x=x); }
    relation s { n = retain_map(x=m); }
}
"#;
    let document = eqiora::api::ModelDocument::compile("nominal.eqi", source).unwrap();
    let root = prepare_package_release_v1(
        source_package(ROOT_PACKAGE, "src/main.eqi", source, vec![]),
        &[],
    )
    .unwrap();
    let (store, resolution) = exact_package::resolved_store(&root, &[]);
    let packaged = PackagedModelDocument::compile_locked(&store, &resolution, "Main").unwrap();
    assert_eq!(
        packaged.model().structural_fingerprint().unwrap(),
        document.structural_fingerprint().unwrap()
    );
    let bytes = document.canonical_json().unwrap();
    let replay = eqiora::api::ModelDocument::replay(&bytes).unwrap();
    assert_eq!(replay.canonical_json().unwrap(), bytes);
    let renamed = source
        .replace("input x:", "input arg:")
        .replace("component(x,", "component(arg,")
        .replace("retain(x=x)", "retain(arg=x)")
        .replace("retain_map(x=m)", "retain_map(arg=m)");
    assert_eq!(
        eqiora::api::ModelDocument::compile("moved.eqi", &renamed)
            .unwrap()
            .canonical_json()
            .unwrap(),
        bytes
    );
    for program in [
        document.program(),
        replay.program(),
        packaged.model().program(),
    ] {
        let mut definitions = 0;
        for node in program.nodes() {
            let KernelNode::Relation(relation) = node else {
                continue;
            };
            for definition in relation.expression().definitions().values() {
                definitions += 1;
                let ty = definition.formals()[0].exact_type().unwrap();
                let components = (0..ty.shape().component_count().unwrap())
                    .map(|i| (i as f64 + 1., -(i as f64 + 2.)))
                    .collect::<Vec<_>>();
                let input = eqiora::ValueLiteral::new(ty.clone(), components.clone()).unwrap();
                let output =
                    ComponentScalarization::evaluate_pure_operator(definition, &[&input]).unwrap();
                assert_eq!(output.value_type(), ty);
                assert_eq!(output.components().unwrap().collect::<Vec<_>>(), components);
            }
        }
        assert_eq!(definitions, 2);
    }
    for changed in [
        source.replace(
            "variable x:coordinates<complex<1>,A>",
            "variable x:coordinates<complex<1>,B>",
        ),
        source.replace(
            "variable m:map<complex<1>,A,B>",
            "variable m:map<complex<1>,B,A>",
        ),
    ] {
        assert!(eqiora::api::ModelDocument::compile("foreign.eqi", &changed).is_err());
    }
}
