use crate::diagnostics::{source_error, stable_sort};

use super::*;

/// Parse and globally analyze every source unit of an exact resolved graph.
///
/// # Errors
/// Returns all parser and global namespace diagnostics together. Analysis
/// creates no graph transaction and performs no package I/O.
pub fn analyze_resolved_hierarchy(
    input: ResolvedHierarchyInput,
) -> Result<AnalyzedResolvedHierarchy, Vec<Diagnostic>> {
    analyze_resolved_hierarchy_with_cancellation(input, || false)
        .map(|analysis| analysis.expect("non-cancellable analysis produces a result"))
}

pub(super) fn analyze_resolved_hierarchy_with_cancellation(
    input: ResolvedHierarchyInput,
    is_cancelled: impl FnMut() -> bool,
) -> Result<Option<AnalyzedResolvedHierarchy>, Vec<Diagnostic>> {
    let modules = input
        .units
        .iter()
        .filter_map(|unit| {
            unit.authored
                .as_ref()
                .filter(|module| module.source_file().is_none())
                .map(|module| (source::source_label(unit), module.clone()))
        })
        .collect::<Vec<_>>();
    analyze_inner(input, is_cancelled)
        .map_err(|errors| source::native_diagnostics(&modules, errors))
}

fn analyze_inner(
    input: ResolvedHierarchyInput,
    mut is_cancelled: impl FnMut() -> bool,
) -> Result<Option<AnalyzedResolvedHierarchy>, Vec<Diagnostic>> {
    if is_cancelled() {
        return Ok(None);
    }
    preflight_resolved_hierarchy(
        input.units.iter().map(ResolvedSourceUnit::input_bytes),
        input.dependencies.len(),
    )
    .map_err(|diagnostic| vec![diagnostic])?;

    let limits = HierarchyLimits::default();
    let mut diagnostics = Vec::new();
    let mut units = Vec::new();
    for unit in input.units {
        if is_cancelled() {
            return Ok(None);
        }
        match source::analyze_source_unit(unit, limits.provenance.max_source_path_bytes) {
            Ok(unit) => units.push(unit),
            Err(mut errors) => diagnostics.append(&mut errors),
        }
    }
    if is_cancelled() {
        return Ok(None);
    }

    source::validate_array_closures(&units).map_err(|diagnostic| vec![diagnostic])?;

    let authored_imports = units.iter().try_fold(0_usize, |count, unit| {
        count.checked_add(unit.document.imports().len())
    });
    let alias_count =
        authored_imports.and_then(|count| input.dependencies.len().checked_add(count));
    if alias_count.is_none_or(|count| count > MAX_ALIASES) {
        diagnostics.push(resolved_error(format!(
            "resolved hierarchy exceeds the {MAX_ALIASES} module-link limit"
        )));
        stable_sort(&mut diagnostics);
        return Err(diagnostics);
    }

    let mut module_index = BTreeMap::new();
    for unit in &units {
        let path = unit.module.import_path();
        if let Some(previous) = module_index.insert(path.clone(), unit.file.clone()) {
            diagnostics.push(resolved_error(format!(
                "canonical module identity `{path}` is supplied by both `{previous}` and `{}`",
                unit.file
            )));
        }
    }

    validate_dependencies(&units, &input.dependencies, &mut diagnostics);

    let mut aliases = Vec::new();
    aliases
        .try_reserve(authored_imports.expect("checked authored import count"))
        .map_err(|_| vec![resolved_error("cannot reserve authored module imports")])?;
    for unit in &units {
        if is_cancelled() {
            return Ok(None);
        }
        for (import_module, import_alias, import_range) in unit.document.imports() {
            let Some(target_file) = module_index.get(import_module.as_str()) else {
                diagnostics.push(source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    &unit.file,
                    import_range,
                    format!("unknown canonical module `{import_module}`"),
                ));
                continue;
            };
            let target = units
                .iter()
                .find(|candidate| candidate.file == *target_file)
                .map(|candidate| &candidate.module)
                .expect("module index refers to an analyzed source unit");
            if target.owner() != unit.module.owner()
                && !input.dependencies.iter().any(|dependency| {
                    dependency.declaring() == unit.module.owner()
                        && dependency.target() == target.owner()
                })
            {
                diagnostics.push(source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    &unit.file,
                    import_range,
                    format!(
                        "canonical module `{import_module}` is not in the current package or a direct dependency"
                    ),
                ));
                continue;
            }
            aliases.push(ResolvedAlias::authored_import(
                unit.module.clone(),
                import_alias,
                target.clone(),
                &unit.file,
                import_range,
            ));
        }
    }

    graph::validate_graph_shape(&input.root, &units, &aliases, &mut diagnostics);
    if is_cancelled() {
        return Ok(None);
    }
    if !diagnostics.is_empty() {
        stable_sort(&mut diagnostics);
        return Err(diagnostics);
    }
    let mut analysis = AnalyzedResolvedHierarchy {
        completion: Default::default(),
        root: input.root,
        units,
        aliases,
        canonical_declarations: Box::new([]),
        declaration_locations: Box::new([]),
        reference_locations: Box::new([]),
        property_bindings: Box::new([]),
        property_catalog: std::sync::Arc::new(Default::default()),
    };
    let canonical_units = analysis.units.clone();
    crate::enumeration::bind_resolved(&mut analysis.units, &analysis.aliases)?;
    crate::dimensions::bind_resolved(&mut analysis.units, &analysis.aliases)?;
    crate::nominal::bind_resolved(&mut analysis.units, &analysis.aliases)?;
    for unit in &mut analysis.units {
        unit.authored_document = std::sync::Arc::new(unit.document.clone());
    }
    analysis.property_catalog = std::sync::Arc::new(crate::property::catalog::build(
        &analysis.units,
        &analysis.aliases,
    )?);
    analysis.property_bindings = crate::property::validate_bindings(
        &analysis.units,
        &analysis.aliases,
        &analysis.property_catalog,
    )?;
    if is_cancelled() {
        return Ok(None);
    }
    crate::hierarchy::validate_resolved_hierarchy(&analysis, limits)?;
    if is_cancelled() {
        return Ok(None);
    }
    analysis.canonical_declarations =
        collect_canonical_declarations(&canonical_units, &analysis.aliases, &mut diagnostics)
            .into_boxed_slice();
    if is_cancelled() {
        return Ok(None);
    }
    analysis.declaration_locations = declaration::collect_declaration_locations(
        &canonical_units,
        &analysis.canonical_declarations,
    )
    .into_boxed_slice();
    analysis.reference_locations = declaration::collect_reference_locations(
        &canonical_units,
        &analysis.aliases,
        &analysis.canonical_declarations,
    )?
    .into_boxed_slice();
    if is_cancelled() {
        return Ok(None);
    }
    if diagnostics.is_empty() {
        Ok(Some(analysis))
    } else {
        stable_sort(&mut diagnostics);
        Err(diagnostics)
    }
}

fn validate_dependencies(
    units: &[AnalyzedSourceUnit],
    dependencies: &[ResolvedDependency],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let namespaces = units
        .iter()
        .map(|unit| unit.module.owner())
        .collect::<BTreeSet<_>>();
    let mut edges = BTreeSet::new();
    for dependency in dependencies {
        if dependency.declaring() == dependency.target() {
            diagnostics.push(resolved_error(format!(
                "package `{}` cannot depend on itself",
                dependency.declaring()
            )));
        }
        if !namespaces.contains(dependency.declaring()) {
            diagnostics.push(resolved_error(format!(
                "dependency has unknown declaring package `{}`",
                dependency.declaring()
            )));
        }
        if !namespaces.contains(dependency.target()) {
            diagnostics.push(resolved_error(format!(
                "dependency has unknown target package `{}`",
                dependency.target()
            )));
        }
        if !edges.insert((dependency.declaring(), dependency.target())) {
            diagnostics.push(resolved_error(format!(
                "duplicate direct dependency `{}` -> `{}`",
                dependency.declaring(),
                dependency.target()
            )));
        }
    }
}
