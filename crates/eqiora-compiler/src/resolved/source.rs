use eqiora_core::Diagnostic;
use eqiora_lang::parse;

use super::{AnalyzedSourceUnit, ResolvedSourceUnit, resolved_error, resolved_source_label};

pub(super) fn analyze_source_unit(
    unit: ResolvedSourceUnit,
    max_provenance_path_bytes: usize,
) -> Result<AnalyzedSourceUnit, Vec<Diagnostic>> {
    if unit.file.is_empty() || unit.file.contains('\0') {
        return Err(vec![resolved_error(
            "resolved source paths must be nonempty and NUL-free",
        )]);
    }

    let parse_file = source_label(&unit);
    check_provenance_path(&parse_file, max_provenance_path_bytes)?;
    let document = match &unit.authored {
        Some(module) => module.document().clone(),
        None => parse(&parse_file, &unit.source).into_document()?,
    };
    let module = unit.module;
    let provenance_file = parse_file;
    check_provenance_path(&provenance_file, max_provenance_path_bytes)?;

    Ok(AnalyzedSourceUnit {
        native: unit.authored.map(std::sync::Arc::new),
        resolved_arrays: unit.resolved_arrays,
        resolved_documents: unit.resolved_documents,
        module,
        file: provenance_file,
        source_bytes: unit.input_bytes,
        authored_document: std::sync::Arc::new(document.clone()),
        document,
    })
}

pub(super) fn source_label(unit: &ResolvedSourceUnit) -> String {
    if unit
        .authored
        .as_ref()
        .is_some_and(|module| module.source_file().is_none())
    {
        format!("<module:{}>", unit.module())
    } else {
        resolved_source_label(unit.module(), &unit.file)
    }
}

pub(super) fn native_diagnostics(
    modules: &[(String, eqiora_lang::Module)],
    diagnostics: Vec<Diagnostic>,
) -> Vec<Diagnostic> {
    diagnostics
        .into_iter()
        .map(|diagnostic| {
            let module = diagnostic
                .source_span()
                .and_then(|span| modules.iter().find(|(file, _)| file == &span.file));
            module.map_or_else(
                || diagnostic.clone(),
                |(_, module)| crate::diagnostics::native_diagnostic(module, diagnostic.clone()),
            )
        })
        .collect()
}

fn check_provenance_path(path: &str, limit: usize) -> Result<(), Vec<Diagnostic>> {
    if path.len() > limit {
        return Err(vec![resolved_error(format!(
            "package-qualified source path requires {} bytes, exceeding the {limit} byte provenance-path limit",
            path.len()
        ))]);
    }
    Ok(())
}

/// A package namespace has one exact data closure across all its modules.
pub(super) fn validate_array_closures(units: &[AnalyzedSourceUnit]) -> Result<(), Diagnostic> {
    let mut closures = std::collections::BTreeMap::new();
    for unit in units {
        if let Some((arrays, documents)) = closures.insert(
            unit.module.owner(),
            (&unit.resolved_arrays, &unit.resolved_documents),
        ) && ((!std::sync::Arc::ptr_eq(arrays, &unit.resolved_arrays)
            && arrays != &unit.resolved_arrays)
            || (!std::sync::Arc::ptr_eq(documents, &unit.resolved_documents)
                && documents != &unit.resolved_documents))
        {
            return Err(resolved_error(
                "source units in one package namespace supply different exact asset closures",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod array_tests {
    use super::*;
    use crate::CompilationNamespaceId;
    use eqiora_schema::resolved_array::ResolvedF64Array;
    use std::{collections::BTreeMap, sync::Arc};
    fn owner(name: &str) -> CompilationNamespaceId {
        CompilationNamespaceId::new([name, "1.0.0", "exact"]).unwrap()
    }
    fn arrays(value: f64) -> Arc<BTreeMap<String, ResolvedF64Array>> {
        Arc::new(BTreeMap::from([(
            "curves.Cp".into(),
            ResolvedF64Array::new(vec![1], vec![value]).unwrap(),
        )]))
    }
    #[test]
    fn array_payloads_share_but_one_namespace_cannot_substitute_its_closure() {
        let shared = arrays(1.);
        let one = ResolvedSourceUnit::new(owner("one"), "src/main.eqi", "model Main() {}")
            .unwrap()
            .with_resolved_arrays(shared.clone())
            .unwrap();
        let two = ResolvedSourceUnit::new(owner("one"), "src/other.eqi", "model Other() {}")
            .unwrap()
            .with_resolved_arrays(shared.clone())
            .unwrap();
        assert!(Arc::ptr_eq(&one.resolved_arrays, &two.resolved_arrays));
        assert!(one.input_bytes() > "model Main() {}".len());
        let analyzed = [
            analyze_source_unit(one, 4096).unwrap(),
            analyze_source_unit(two, 4096).unwrap(),
        ];
        validate_array_closures(&analyzed).unwrap();
        let mut changed = analyzed.clone();
        changed[1].resolved_arrays = arrays(2.);
        assert!(validate_array_closures(&changed).is_err());
        changed[1].module = super::super::CompilationModuleId::main(owner("two"));
        validate_array_closures(&changed).unwrap();
    }
    #[test]
    fn invalid_asset_names_and_input_excess_reject_before_analysis() {
        let invalid = Arc::new(BTreeMap::from([(
            "../escape".into(),
            ResolvedF64Array::new(vec![1], vec![0.]).unwrap(),
        )]));
        assert!(
            ResolvedSourceUnit::new(owner("one"), "src/main.eqi", "")
                .unwrap()
                .with_resolved_arrays(invalid)
                .is_err()
        );
        let excessive = Arc::new(BTreeMap::from([(
            "values".into(),
            ResolvedF64Array::new(vec![2_097_153], vec![0.; 2_097_153]).unwrap(),
        )]));
        assert!(
            ResolvedSourceUnit::new(owner("one"), "src/main.eqi", "")
                .unwrap()
                .with_resolved_arrays(excessive)
                .is_err()
        );
    }
    #[test]
    fn attribution_payloads_are_shared_and_namespace_coherent() {
        let docs = Arc::new(BTreeMap::from([(
            "license".to_owned(),
            Arc::<[u8]>::from(b"license bytes".as_slice()),
        )]));
        let one = ResolvedSourceUnit::new(owner("one"), "src/main.eqi", "model Main() {}")
            .unwrap()
            .with_resolved_documents(docs.clone())
            .unwrap();
        assert_eq!(
            one.resolved_documents()["license"].as_ref(),
            b"license bytes"
        );
        let two = ResolvedSourceUnit::new(owner("one"), "src/other.eqi", "model Other() {}")
            .unwrap()
            .with_resolved_documents(docs)
            .unwrap();
        let mut analyzed = [
            analyze_source_unit(one, 4096).unwrap(),
            analyze_source_unit(two, 4096).unwrap(),
        ];
        validate_array_closures(&analyzed).unwrap();
        analyzed[1].resolved_documents = Arc::new(BTreeMap::new());
        assert!(validate_array_closures(&analyzed).is_err());
        let invalid = Arc::new(BTreeMap::from([(
            "license".into(),
            Arc::<[u8]>::from([255]),
        )]));
        assert!(
            ResolvedSourceUnit::new(owner("one"), "src/main.eqi", "")
                .unwrap()
                .with_resolved_documents(invalid)
                .is_err()
        );
    }
}
