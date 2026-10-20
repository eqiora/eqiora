pub(crate) mod catalog;
pub(crate) mod descriptor;
mod resolution;
pub(crate) mod table;
use eqiora_schema::kernel::PropertyMeaning;
use resolution::resolve_path;

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{ComponentItem, Expr, InstanceDecl, Item, NamePath, TextRange, VisibilitySyntax};

use crate::diagnostics::source_error;
use crate::dimensions::lower_dimension;
use crate::resolved::{AnalyzedSourceUnit, CompilationModuleId, ResolvedAlias};

type Key = (CompilationModuleId, String);

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedPropertyBinding {
    composition: Option<String>,
    contract: String,
    release: String,
    component: String,
    requirement: String,
    meaning: PropertyMeaning,
    validity: &'static str,
    citation: String,
    license: String,
}

impl ResolvedPropertyBinding {
    #[must_use]
    pub fn composition(&self) -> Option<&str> {
        self.composition.as_deref()
    }
    #[must_use]
    pub fn contract(&self) -> &str {
        &self.contract
    }
    #[must_use]
    pub fn release(&self) -> &str {
        &self.release
    }
    #[must_use]
    pub fn component(&self) -> &str {
        &self.component
    }
    #[must_use]
    pub fn requirement(&self) -> &str {
        &self.requirement
    }
    #[must_use]
    pub const fn meaning(&self) -> &PropertyMeaning {
        &self.meaning
    }
    #[must_use]
    pub const fn validity(&self) -> &'static str {
        self.validity
    }
    #[must_use]
    pub fn citation(&self) -> &str {
        &self.citation
    }
    #[must_use]
    pub fn license(&self) -> &str {
        &self.license
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Contract {
    pub(crate) file: String,
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) value_type: eqiora_lang::ValueTypeSyntax,
    pub(crate) inputs: Vec<(String, eqiora_lang::ValueTypeSyntax)>,
    pub(crate) derivatives: eqiora_schema::kernel::PropertyDerivatives,
    pub(crate) branch: Option<String>,
}

impl Contract {
    pub(crate) fn result_type<I>(
        &self,
        support: Option<&eqiora_schema::kernel::typing::SpatialSupport<I>>,
    ) -> Result<eqiora_core::ValueType, Diagnostic> {
        crate::value_types::lower_signature_type(&self.file, &self.value_type, support)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Release {
    pub(crate) visibility: VisibilitySyntax,
    pub(crate) contract: Key,
    pub(crate) meaning: PropertyMeaning,
    pub(crate) branch: Option<String>,
    pub(crate) citation: String,
    pub(crate) license: String,
}

#[derive(Clone, Debug)]
struct Composition {
    visibility: VisibilitySyntax,
    properties: Vec<(String, Key, TextRange)>,
}

pub(crate) fn validate_bindings(
    units: &[AnalyzedSourceUnit],
    aliases: &[ResolvedAlias],
    catalog: &catalog::Catalog,
) -> Result<Box<[ResolvedPropertyBinding]>, Vec<Diagnostic>> {
    let has_property_syntax = units.iter().any(|unit| {
        unit.document.property_contract_syntax().len() != 0
            || unit.document.property_release_syntax().len() != 0
            || unit.document.material_composition_syntax().len() != 0
            || unit.document.components().iter().any(|component| {
                component
                    .signature()
                    .iter()
                    .any(|item| matches!(item, eqiora_lang::SignatureItem::Property(_)))
            })
            || unit.document.models().iter().any(|model| {
                model
                    .signature()
                    .iter()
                    .any(|item| matches!(item, eqiora_lang::SignatureItem::Property(_)))
            })
    });
    if !has_property_syntax {
        return Ok(Box::new([]));
    }
    let mut diagnostics = Vec::new();
    let catalog::Catalog {
        contracts,
        releases,
        compositions,
    } = catalog;

    let components = units
        .iter()
        .flat_map(|unit| {
            unit.document.components().iter().map(move |value| {
                (
                    (unit.module.clone(), value.name().to_owned()),
                    (value.clone(), unit.file.clone()),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();
    let mut projections = Vec::new();
    for unit in units.iter() {
        for signature in unit
            .document
            .components()
            .iter()
            .map(|value| value.signature())
            .chain(unit.document.models().iter().map(|value| value.signature()))
        {
            for requirement in signature.iter().filter_map(|item| match item {
                eqiora_lang::SignatureItem::Property(value) => Some(value),
                _ => None,
            }) {
                let contract_path = requirement.contract();
                if let Some(key) = resolve_path(
                    &unit.module,
                    contract_path,
                    aliases,
                    contracts,
                    |value| value.visibility,
                    &unit.file,
                    &mut diagnostics,
                ) {
                    let _ = &contracts[&key];
                }
            }
        }
        let instances = unit
            .document
            .components()
            .iter()
            .flat_map(|component| {
                component.items().iter().filter_map(|item| match item {
                    ComponentItem::Instance(value) => Some(value),
                    _ => None,
                })
            })
            .chain(unit.document.models().iter().flat_map(|model| {
                model.items().iter().filter_map(|item| match item {
                    Item::Instance(value) => Some(value),
                    _ => None,
                })
            }));
        for instance in instances {
            validate_instance(
                instance,
                &unit.module,
                &unit.file,
                aliases,
                &components,
                contracts,
                releases,
                compositions,
                &mut projections,
                &mut diagnostics,
            );
        }
    }
    if diagnostics.is_empty() {
        projections.sort_by(|a, b| {
            (&a.component, &a.requirement, &a.release).cmp(&(
                &b.component,
                &b.requirement,
                &b.release,
            ))
        });
        Ok(projections.into_boxed_slice())
    } else {
        Err(diagnostics)
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_instance(
    instance: &InstanceDecl,
    namespace: &CompilationModuleId,
    file: &str,
    aliases: &[ResolvedAlias],
    components: &BTreeMap<Key, (eqiora_lang::ComponentDecl, String)>,
    contracts: &BTreeMap<Key, Contract>,
    releases: &BTreeMap<Key, Release>,
    compositions: &BTreeMap<Key, Composition>,
    projections: &mut Vec<ResolvedPropertyBinding>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(component_key) = resolve_path(
        namespace,
        instance.definition(),
        aliases,
        components,
        |value| value.0.visibility(),
        file,
        diagnostics,
    ) else {
        return;
    };
    let (component, _) = &components[&component_key];
    let requirements = component
        .signature()
        .iter()
        .filter_map(|item| match item {
            eqiora_lang::SignatureItem::Property(value) => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    for requirement in requirements {
        let bindings = instance
            .bindings()
            .iter()
            .filter(|binding| binding.name() == requirement.name())
            .collect::<Vec<_>>();
        let [binding] = bindings.as_slice() else {
            diagnostics.push(error(
                file,
                instance.range(),
                format!(
                    "instance `{}` requires property `{}` exactly once",
                    instance.name(),
                    requirement.name()
                ),
            ));
            continue;
        };
        let path = match binding.value().kind() {
            eqiora_lang::ExprKind::Name(name) => {
                NamePath::from_segments([name.as_str()], binding.value().range())
                    .expect("parsed name")
            }
            eqiora_lang::ExprKind::Path(path) => path.clone(),
            _ => {
                diagnostics.push(error(
                    file,
                    binding.range(),
                    "property binding requires an exact release or composition member",
                ));
                continue;
            }
        };
        let Some((release_key, composition_key)) = resolve_property_value(
            namespace,
            &path,
            aliases,
            releases,
            compositions,
            file,
            diagnostics,
        ) else {
            continue;
        };
        let binding_range = binding.range();
        let contract_path = requirement.contract();
        let Some(required_contract) = resolve_path(
            &component_key.0,
            contract_path,
            aliases,
            contracts,
            |value| value.visibility,
            file,
            diagnostics,
        ) else {
            continue;
        };
        let release = &releases[&release_key];
        if release.contract != required_contract {
            diagnostics.push(error(
                file,
                binding_range,
                "property release implements a different nominal contract",
            ));
            continue;
        }
        projections.push(ResolvedPropertyBinding {
            composition: composition_key.as_ref().map(qualified),
            contract: qualified(&required_contract),
            release: qualified(&release_key),
            component: qualified(&component_key),
            requirement: requirement.name().to_owned(),
            meaning: release.meaning.clone(),
            validity: if matches!(&release.meaning, PropertyMeaning::Analytic(definition) if matches!(definition.nodes()[definition.root().index() as usize], eqiora_schema::kernel::pure_operator::CalculusNode::Require { .. })) { "checked" } else { "unconditional" },
            citation: release.citation.clone(),
            license: release.license.clone(),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_property_value(
    namespace: &CompilationModuleId,
    path: &NamePath,
    aliases: &[ResolvedAlias],
    releases: &BTreeMap<Key, Release>,
    compositions: &BTreeMap<Key, Composition>,
    file: &str,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<(Key, Option<Key>)> {
    let segments = path.segments().collect::<Vec<_>>();
    // A local composition owns its member namespace; qualified compositions retain
    // the ordinary import visibility check before their exact release is selected.
    let composition_member = match segments.as_slice() {
        [name, _] => compositions.contains_key(&(namespace.clone(), (*name).to_owned())),
        [_, _, _] => true,
        _ => false,
    };
    if composition_member {
        let prefix =
            NamePath::from_segments(segments[..segments.len() - 1].iter().copied(), path.range())
                .expect("parsed path");
        let key = resolve_path(
            namespace,
            &prefix,
            aliases,
            compositions,
            |value| value.visibility,
            file,
            diagnostics,
        )?;
        let member = segments[segments.len() - 1];
        let Some((_, release, _)) = compositions[&key]
            .properties
            .iter()
            .find(|(name, _, _)| name == member)
        else {
            diagnostics.push(error(
                file,
                path.range(),
                format!("material composition has no property `{member}`"),
            ));
            return None;
        };
        Some((release.clone(), Some(key)))
    } else {
        resolve_path(
            namespace,
            path,
            aliases,
            releases,
            |value| value.visibility,
            file,
            diagnostics,
        )
        .map(|key| (key, None))
    }
}

fn constant(file: &str, expression: &Expr) -> Result<f64, Diagnostic> {
    crate::hierarchy::closed_value(
        file,
        expression,
        eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            eqiora_core::DimExponents::DIMENSIONLESS,
        )
        .expect("admitted numeric scalar type"),
    )?
    .real_scalar_value()
    .map(|value| value.value())
    .ok_or_else(|| {
        error(
            file,
            expression.range(),
            "property normalization requires a real dimensionless scalar",
        )
    })
}

fn qualified(key: &Key) -> String {
    format!("{}::{}", key.0, key.1)
}

fn error(file: &str, range: TextRange, message: impl Into<String>) -> Diagnostic {
    source_error(codes::LANGUAGE_TYPE_ERROR, file, range, message)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::{
        CanonicalDeclarationKind, CompilationNamespaceId, ResolvedHierarchyInput,
        ResolvedSourceUnit, analyze_resolved_hierarchy,
    };

    #[test]
    fn rational_property_units_and_pure_operator_dimensions_share_exact_algebra() {
        let source = r#"
operator square(input x: scalar): scalar = component(x) * component(x);
property contract Amplitude(): m ^ (-1 / 2) { derivatives value_only; }
property release Reference: Amplitude {
  analytic { value = 8;
  source_unit: m ^ (-2 / 4) = 1 / 4; }
  validity unconditional; outside reject; branch single;
  citation org.example.measurement;
  license spdx.CC0_1_0;
}
component Wave(support body: volume(ambient_dimension = 1), property amplitude: Amplitude) {
  variable value: m ^ (-1 / 2) on body; initial { value = 0; }
  variable intensity: m ^ -1 on body; initial { intensity = 0; }
  relation law on body { value = amplitude; intensity = square(x=value); }
}
model Main() {
  domain interval = box(0, 1);
  instance wave: Wave(body = interval, amplitude = Reference);
}
"#;
        let input = |text: &str| {
            let root = CompilationNamespaceId::new(["root", "1.0.0", "dimension-test"]).unwrap();
            ResolvedHierarchyInput::new(
                root.clone(),
                vec![ResolvedSourceUnit::new(root, "src/main.eqi", text).unwrap()],
                vec![],
            )
        };
        let analyzed = analyze_resolved_hierarchy(input(source)).unwrap();
        // The existing conversion owner applies 8 * (1/4) once.
        assert_eq!(
            analyzed
                .property_bindings()
                .next()
                .unwrap()
                .5
                .constant_value()
                .expect("constant property")
                .component(0)
                .unwrap()
                .0,
            2.0
        );
        analyzed
            .validate_definitions()
            .unwrap()
            .compile_root("Main")
            .unwrap();
        let wrong_unit = source.replace("source_unit: m ^ (-2 / 4)", "source_unit: m ^ -1");
        assert!(analyze_resolved_hierarchy(input(&wrong_unit)).is_err());
        let wrong_output = source.replace(
            "intensity: m ^ -1 on body",
            "intensity: m ^ (-1 / 2) on body",
        );
        let invalid = analyze_resolved_hierarchy(input(&wrong_output)).unwrap();
        assert!(invalid.validate_definitions().is_err());
    }

    #[test]
    fn exact_scalar_property_elaborates_through_parameter_terms() {
        let root = CompilationNamespaceId::new(["root", "1.0.0", "semantic-digest"]).unwrap();
        let source = r#"
public property contract Diffusivity(): m ^ 2 / s { derivatives value_only; }
property release ReferenceDiffusivity: Diffusivity {
  analytic { value = 25;
  source_unit: m ^ 2 / s = 1 / 1000; }
  validity unconditional; outside reject; branch single;
  citation org.example.measurement;
  license spdx.CC0_1_0;
}
public component Diffusion(property diffusivity: Diffusivity) {
  relation law { diffusivity = 0; }
}
model Main() { instance domain: Diffusion(diffusivity = ReferenceDiffusivity); }
"#;
        let input = ResolvedHierarchyInput::new(
            root.clone(),
            vec![
                ResolvedSourceUnit::new(root.clone(), "src/main.eqi", source).expect("source path"),
            ],
            vec![],
        );
        let analyzed = analyze_resolved_hierarchy(input).expect("property graph analyzes");
        assert_eq!(analyzed.property_bindings().len(), 1);
        assert_eq!(
            analyzed
                .property_bindings()
                .next()
                .unwrap()
                .5
                .constant_value()
                .expect("constant property")
                .component(0)
                .unwrap()
                .0,
            0.025
        );
        for invalid in [
            "Diffusivity",
            "ReferenceDiffusivity + 1",
            "0.025[m ^ 2 / s]",
        ] {
            let source = source.replace(
                "diffusivity = ReferenceDiffusivity",
                &format!("diffusivity = {invalid}"),
            );
            let result = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
                root.clone(),
                vec![ResolvedSourceUnit::new(root.clone(), "src/main.eqi", source).unwrap()],
                vec![],
            ));
            assert!(
                result.is_err(),
                "property requires a nominal static release: {invalid}"
            );
        }
        let property_model = analyzed
            .validate_definitions()
            .expect("property definitions validate")
            .compile_root("Main")
            .expect("property model compiles");
        let direct = r#"
public component Diffusion(parameter diffusivity: m ^ 2 / s) {
  relation law { diffusivity = 0; }
}
model Main() { instance domain: Diffusion(diffusivity = 0.025[m ^ 2 / s]); }
"#;
        let direct_model = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
            root.clone(),
            vec![ResolvedSourceUnit::new(root, "src/main.eqi", direct).expect("source path")],
            vec![],
        ))
        .unwrap()
        .validate_definitions()
        .unwrap()
        .compile_root("Main")
        .unwrap();
        assert_eq!(
            property_model.symbols().get("domain.law"),
            direct_model.symbols().get("domain.law"),
            "property binding reuses the same effective scalar Law"
        );
    }

    #[test]
    fn material_composition_binds_multiple_properties_to_one_component_law() {
        let root = CompilationNamespaceId::new(["root", "1.0.0", "semantic-digest"]).unwrap();
        let source = r#"
public property contract Conductivity(): 1 { derivatives value_only; }
public property contract Capacity(): 1 { derivatives value_only; }
public property release ConductivityA: Conductivity {
  analytic { value = 2; source_unit: 1 = 1; } validity unconditional; outside reject; branch single;
  citation org.example.a; license spdx.CC0_1_0;
}
public property release CapacityA: Capacity {
  analytic { value = 4; source_unit: 1 = 1; } validity unconditional; outside reject; branch single;
  citation org.example.a; license spdx.CC0_1_0;
}
public material composition MaterialA {
  property capacity = CapacityA;
  property conductivity = ConductivityA;
}
public component DiffusionLaw(property conductivity: Conductivity, property capacity: Capacity) {
  relation law { conductivity / capacity = 0; }
}
model Main() { instance domain: DiffusionLaw(conductivity = MaterialA.conductivity, capacity = MaterialA.capacity); }
"#;
        let analyzed = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
            root.clone(),
            vec![ResolvedSourceUnit::new(root, "src/main.eqi", source).expect("source path")],
            vec![],
        ))
        .expect("material composition analyzes");
        let references = analyzed.resolved_references().collect::<Vec<_>>();
        // Two explicit member bindings each reference their composition.
        assert_eq!(references.len(), 9);
        assert!(references.windows(2).all(|pair| {
            (pair[0].1, pair[0].2.start(), pair[0].2.end())
                < (pair[1].1, pair[1].2.start(), pair[1].2.end())
        }));
        let reference_kinds = references.iter().fold(
            BTreeMap::<CanonicalDeclarationKind, usize>::new(),
            |mut counts, (target, _, _, _, _)| {
                *counts.entry(target.kind()).or_default() += 1;
                counts
            },
        );
        assert_eq!(
            reference_kinds[&CanonicalDeclarationKind::PropertyContract],
            4
        );
        assert_eq!(
            reference_kinds[&CanonicalDeclarationKind::PropertyRelease],
            2
        );
        assert_eq!(reference_kinds[&CanonicalDeclarationKind::Component], 1);
        assert_eq!(
            reference_kinds[&CanonicalDeclarationKind::MaterialComposition],
            2
        );
        for (_, _, range, _, _) in references {
            let spelling = &source
                [usize::try_from(range.start()).unwrap()..usize::try_from(range.end()).unwrap()];
            assert!(!spelling.contains(char::is_whitespace));
        }
        let bindings = analyzed.property_bindings().collect::<Vec<_>>();
        assert_eq!(bindings.len(), 2);
        assert!(bindings.iter().all(|binding| binding.0.is_some()));
        assert_eq!(bindings[0].0, bindings[1].0);
        analyzed
            .validate_definitions()
            .expect("composed definitions validate")
            .compile_root("Main")
            .expect("composed Law compiles");
    }

    #[test]
    fn incompatible_and_incomplete_property_bindings_fail_before_compilation() {
        let root = CompilationNamespaceId::new(["root", "1.0.0", "semantic-digest"]).unwrap();
        let wrong_dimension = r#"
property contract Diffusivity(): m ^ 2 / s { derivatives value_only; }
property release Wrong: Diffusivity {
  analytic { value = 1; source_unit: kg = 1; } validity unconditional; outside reject; branch single;
  citation org.example.measurement; license spdx.CC0_1_0;
}
model Main() {}
"#;
        let diagnostics = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
            root.clone(),
            vec![
                ResolvedSourceUnit::new(root.clone(), "src/main.eqi", wrong_dimension)
                    .expect("source path"),
            ],
            vec![],
        ))
        .unwrap_err();
        assert!(
            diagnostics
                .iter()
                .any(|value| value.message().contains("does not match"))
        );

        let missing = r#"
property contract Diffusivity(): m ^ 2 / s { derivatives value_only; }
component Diffusion(property diffusivity: Diffusivity) {
  relation law { diffusivity = 0; }
}
model Main() { instance domain: Diffusion(); }
"#;
        let diagnostics = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
            root.clone(),
            vec![ResolvedSourceUnit::new(root, "src/main.eqi", missing).expect("source path")],
            vec![],
        ))
        .unwrap_err();
        assert!(
            diagnostics
                .iter()
                .any(|value| value.message().contains("requires property"))
        );

        for (source, expected) in [
            (
                r#"
property contract A(): 1 { derivatives value_only; }
property release A1: A {
  analytic { value = 1; source_unit: 1 = 1; } validity unconditional; outside reject; branch single;
  citation org.example; license spdx.CC0_1_0;
}
material composition Duplicate {
  property value = A1;
  property value = A1;
}
component Law(property value: A) { relation law { value = 0; } }
model Main() { instance law: Law(value = Duplicate.value); }
"#,
                "duplicate material property",
            ),
            (
                r#"
property contract A(): 1 { derivatives value_only; }
property contract B(): 1 { derivatives value_only; }
property release B1: B {
  analytic { value = 1; source_unit: 1 = 1; } validity unconditional; outside reject; branch single;
  citation org.example; license spdx.CC0_1_0;
}
material composition Foreign { property value = B1; }
component Law(property value: A) { relation law { value = 0; } }
model Main() { instance law: Law(value = Foreign.value); }
"#,
                "different nominal contract",
            ),
            (
                r#"
property contract A(): 1 { derivatives value_only; }
property release A1: A {
  analytic { value = 1; source_unit: 1 = 1; } validity unconditional; outside reject; branch single;
  citation org.example; license spdx.CC0_1_0;
}
material composition EmptyForLaw { property other = A1; }
component Law(property value: A) { relation law { value = 0; } }
model Main() { instance law: Law(value = EmptyForLaw.value); }
"#,
                "has no property `value`",
            ),
            (
                r#"
property contract A(): 1 { derivatives value_only; }
property release A1: A {
  analytic { value = 1; source_unit: 1 = 1; } validity unconditional; outside reject; branch single;
  citation org.example; license spdx.CC0_1_0;
}
material composition MaterialA { property value = A1; }
component Law(property value: A) { relation law { value = 0; } }
model Main() {
  instance law: Law(value = MaterialA.value, value = A1);
}
"#,
                "exactly once",
            ),
        ] {
            let root = CompilationNamespaceId::new(["root", "1.0.0", "material-invalid"]).unwrap();
            let diagnostics = analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
                root.clone(),
                vec![ResolvedSourceUnit::new(root, "src/main.eqi", source).expect("source path")],
                vec![],
            ))
            .unwrap_err();
            assert!(
                diagnostics
                    .iter()
                    .any(|value| value.message().contains(expected)),
                "expected {expected:?} in {diagnostics:?}"
            );
        }
    }
}
