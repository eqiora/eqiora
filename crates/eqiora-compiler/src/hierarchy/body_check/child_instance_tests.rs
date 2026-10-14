use super::*;
use crate::hierarchy::HierarchyLimits;
use crate::source_identity::LocalSourceIdentity;

fn bodies(
    source: &str,
    max_instances: usize,
) -> Result<BTreeMap<String, DefinitionBodyProof>, Vec<Diagnostic>> {
    let document = eqiora_lang::parse("children.eqi", source)
        .into_document()
        .expect("fixture parses");
    let identity = LocalSourceIdentity::from_document(&document).unwrap();
    let elaborator = Elaborator::new(
        "children.eqi",
        source.len(),
        &document,
        identity,
        HierarchyLimits {
            max_instances,
            ..HierarchyLimits::default()
        },
    )?;
    let mut proofs = BTreeMap::new();
    for (_, definition) in elaborator.components() {
        let proof = validate_component_body(
            &elaborator,
            definition,
            &SymbolicParameterMap::new(),
            &SupportInterface::default(),
            &FieldInterface::default(),
        )?;
        proofs.insert(definition.declaration.name().to_owned(), proof);
    }
    for (_, definition) in elaborator.models() {
        let proof = validate_model_body(&elaborator, definition, &SymbolicParameterMap::new())?;
        proofs.insert(definition.declaration.name().to_owned(), proof);
    }
    Ok(proofs)
}

#[test]
fn models_and_components_retain_scalar_and_indexed_child_proofs() {
    let children = "indexset I=range(2); instance scalar:Leaf(); instance family[i in I]:Leaf();";
    let source = format!(
        "component Leaf() {{}} component Holder() {{{children}}} model Main() {{{children}}}"
    );
    let proofs = bodies(&source, 3).unwrap();
    for owner in ["Holder", "Main"] {
        let proof = &proofs[owner];
        assert_eq!(
            proof
                .children
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["family[0]", "family[1]", "scalar"]
        );
        for child in proof.children.values() {
            assert_eq!(child.definition.name, "Leaf");
            assert_eq!(
                child.definition.namespace,
                super::super::preflight::DefinitionNamespace::Local
            );
        }
        assert_eq!(
            proof.children["family[0]"].range,
            proof.children["family[1]"].range
        );
        assert_ne!(
            proof.children["scalar"].range,
            proof.children["family[0]"].range
        );
        assert_eq!(proof.file, "children.eqi");
    }
}

#[test]
fn child_proof_budget_counts_scalar_and_indexed_neighbors_in_both_owners() {
    for owner in ["component", "model"] {
        let source = format!(
            "component Leaf() {{}} {owner} Owner() {{ indexset I=range(2); instance scalar:Leaf(); instance family[i in I]:Leaf(); }}"
        );
        bodies(&source, 3).expect("three child names fit exactly");
        for limit in [0, 2] {
            let errors = bodies(&source, limit).unwrap_err();
            assert!(
                errors.iter().any(|error| error
                    .message()
                    .contains("indexed child proof exceeds the instance expansion limit")),
                "{errors:?}"
            );
            assert!(errors.iter().all(|error| error.source_span().is_some()));
        }
    }
}

#[test]
fn both_owners_keep_preflight_diagnostics_and_reject_empty_families() {
    for owner in ["component", "model"] {
        for (body, expected) in [
            ("instance child:Missing();", "unresolved component"),
            ("instance child:Leaf(); instance child:Leaf();", "duplicate"),
            (
                "indexset I=range(0); instance child[i in I]:Leaf();",
                "positive",
            ),
        ] {
            let source = format!("component Leaf() {{}} {owner} Owner() {{{body}}}");
            let errors = bodies(&source, 3).unwrap_err();
            assert!(
                errors
                    .iter()
                    .any(|error| error.message().contains(expected)),
                "{errors:?}"
            );
            assert!(errors.iter().all(|error| error.source_span().is_some()));
        }
    }
}
