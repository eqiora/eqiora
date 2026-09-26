//! Unspecialized Component declarations reuse the ordinary declaration binder.
use super::{Candidate, Expected, Scope, reference};
use crate::hierarchy::{
    body_check::{component::declaration_symbols, scope::SymbolContract},
    field_slots::component_field_interface,
    parameters::{
        RecordContext, resolve_component_lets, resolve_component_parameters_symbolically,
    },
    preflight::Elaborator,
    supports::component_support_interface,
};
use eqiora_lang::{ComponentItem, SignatureItem};
use std::{collections::BTreeMap, sync::Arc};

pub(super) fn collect(
    elaborator: &Elaborator<'_>,
    is_cancelled: &mut impl FnMut() -> bool,
) -> Option<Vec<(String, Scope)>> {
    let mut scopes = Vec::new();
    for (_, definition) in elaborator.components() {
        if is_cancelled() {
            return None;
        }
        let supports = component_support_interface(definition.file, definition.declaration);
        if is_cancelled() {
            return None;
        }
        let Ok(supports) = supports else { continue };
        let values = resolve_component_parameters_symbolically(
            definition.file,
            definition.declaration,
            |name| {
                crate::hierarchy::clocks::component(definition.file, definition.declaration, name)
            },
            &RecordContext::component(elaborator, definition),
        );
        if is_cancelled() {
            return None;
        }
        let Ok(mut values) = values else { continue };
        let properties = elaborator.bind_symbolic_properties(
            &definition.namespace,
            definition.file,
            definition.declaration.signature(),
            &mut values,
        );
        if is_cancelled() {
            return None;
        }
        if properties.is_err() {
            continue;
        }
        let aliases = resolve_component_lets(
            definition.file,
            definition.declaration,
            &mut values,
            |name| {
                crate::hierarchy::clocks::component(definition.file, definition.declaration, name)
            },
        );
        if is_cancelled() {
            return None;
        }
        if aliases.is_err() {
            continue;
        }
        let fields =
            component_field_interface(definition.file, definition.declaration, &supports, &values);
        if is_cancelled() {
            return None;
        }
        let Ok(fields) = fields else { continue };
        let symbols = declaration_symbols(elaborator, definition, &values, &supports, &fields);
        if is_cancelled() {
            return None;
        }
        let Ok(mut symbols) = symbols else { continue };
        let mut candidates = BTreeMap::new();
        let mut declarations = BTreeMap::new();
        let mut contexts = Vec::new();
        let file: Arc<str> = definition.file.into();
        for item in definition.declaration.signature() {
            if is_cancelled() {
                return None;
            }
            let (name, range) = match item {
                SignatureItem::Clock(value)
                    if matches!(symbols.get(value.name()), Some(SymbolContract::Clock)) =>
                {
                    (value.name(), value.range())
                }
                SignatureItem::Field(value)
                    if matches!(symbols.get(value.name()), Some(SymbolContract::Field(..))) =>
                {
                    (value.name(), value.range())
                }
                _ => continue,
            };
            candidates.insert(name.to_owned(), Candidate::Requirement);
            declarations.insert(name.to_owned(), (file.clone(), range));
        }
        for item in definition.owned_items() {
            if is_cancelled() {
                return None;
            }
            let (name, range) = match item {
                ComponentItem::Field(value) => (value.name(), value.range()),
                ComponentItem::Parameter(value) => (value.name(), value.range()),
                ComponentItem::Port(value) => (value.name(), value.range()),
                ComponentItem::Event(value) => {
                    if matches!(symbols.get(value.name()), Some(SymbolContract::Event)) {
                        candidates.insert(value.name().to_owned(), Candidate::Event);
                        declarations.insert(value.name().to_owned(), (file.clone(), value.range()));
                    }
                    continue;
                }
                ComponentItem::Clock(value) => {
                    if let Ok((period, phase)) =
                        crate::units::lower_clock(definition.file, value.period(), value.phase())
                    {
                        candidates.insert(
                            value.name().to_owned(),
                            Candidate::Clock(period, phase, "Component"),
                        );
                        declarations.insert(value.name().to_owned(), (file.clone(), value.range()));
                    }
                    continue;
                }
                _ => continue,
            };
            let candidate = match symbols.remove(name) {
                Some(SymbolContract::Field(value, role, activation)) => {
                    Candidate::Field(Box::new((value, role, activation)))
                }
                Some(SymbolContract::Parameter(value)) => Candidate::Parameter(value.value_type),
                Some(SymbolContract::Port(value)) => Candidate::Port(Box::new(value)),
                _ => continue,
            };
            if let ComponentItem::Parameter(parameter) = item
                && let Some(value) = parameter.default().filter(|value| reference(value))
                && let Candidate::Parameter(expected) = &candidate
            {
                contexts.push((value.range(), Expected::Parameter(expected.clone())));
            }
            candidates.insert(name.to_owned(), candidate);
            declarations.insert(name.to_owned(), (file.clone(), range));
        }
        scopes.push((
            definition.file.to_owned(),
            Scope {
                range: definition.declaration.range(),
                candidates,
                declarations,
                contexts,
                exposed_signals: Default::default(),
            },
        ));
    }
    (!is_cancelled()).then_some(scopes)
}

#[cfg(test)]
mod tests {
    use crate::{
        CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit,
        analyze_resolved_hierarchy,
    };

    #[test]
    fn component_preparation_is_atomic_and_never_borrows_specialized_values() {
        for source in [
            "component C(parameter n:integer){variable values:array<1,n>;}",
            "component C(){parameter n:integer=3;let broken=1[m]+1[s];variable values:array<1,n>;}",
        ] {
            let owner = CompilationNamespaceId::new(["component_editor"]).unwrap();
            let unit = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap();
            let file = unit.diagnostic_file();
            let mut analysis =
                analyze_resolved_hierarchy(ResolvedHierarchyInput::new(owner, vec![unit], vec![]))
                    .unwrap();
            assert!(analysis.prepare_completion(|| false));
            let start = source.find("variable values").unwrap() as u32;
            let end = start + source[start as usize..].find(';').unwrap() as u32 + 1;
            assert!(
                analysis
                    .symbol_description(
                        &file,
                        start + 9,
                        "values",
                        (&file, eqiora_lang::TextRange::new(start, end))
                    )
                    .is_none()
            );
            assert!(
                analysis
                    .value_references(&eqiora_core::Span { file, start, end })
                    .is_none()
            );
        }
        let source = "component C(){let n:integer=1+2;variable values:array<1,n>;}";
        let owner = CompilationNamespaceId::new(["component_editor"]).unwrap();
        let unit = ResolvedSourceUnit::new(owner.clone(), "src/main.eqi", source).unwrap();
        let file = unit.diagnostic_file();
        let mut analysis =
            analyze_resolved_hierarchy(ResolvedHierarchyInput::new(owner, vec![unit], vec![]))
                .unwrap();
        let mut calls = 0_usize;
        assert!(analysis.prepare_completion(|| {
            calls += 1;
            false
        }));
        let start = source.find("variable values").unwrap() as u32;
        let range = eqiora_lang::TextRange::new(start, source.len() as u32 - 1);
        let before = analysis
            .symbol_description(&file, start + 9, "values", (&file, range))
            .unwrap();
        assert!(before.contains("shape [3]"));
        let mut remaining = calls - 1;
        assert!(!analysis.prepare_completion(|| {
            remaining = remaining.saturating_sub(1);
            remaining == 0
        }));
        assert_eq!(
            analysis.symbol_description(&file, start + 9, "values", (&file, range)),
            Some(before)
        );
    }
}
