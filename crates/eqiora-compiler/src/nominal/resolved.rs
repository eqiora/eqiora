//! The existing lexical finite-type owner applied to explicit module imports.
use super::*;
use crate::resolved::{AnalyzedSourceUnit, ResolvedAlias};

pub(crate) fn resolved_spaces(
    units: &[AnalyzedSourceUnit],
    aliases: &[ResolvedAlias],
) -> Result<
    BTreeMap<crate::resolved::CompilationModuleId, BTreeMap<String, BoundFiniteSpace>>,
    Vec<Diagnostic>,
> {
    let mut definitions = BTreeMap::new();
    let mut public = BTreeMap::new();
    for unit in units.iter() {
        let namespace =
            crate::enumeration::resolved_namespace(&unit.module).map_err(|error| vec![error])?;
        spaces::check_names(&unit.file, &unit.document)?;
        definitions.insert(
            unit.module.clone(),
            spaces::declarations(&unit.file, &unit.document, &namespace, None, |name| {
                unit.native
                    .as_ref()
                    .and_then(|module| module.nominal_identity(name))
            })?,
        );
        public.insert(
            unit.module.clone(),
            unit.document
                .finite_spaces()
                .iter()
                .filter(|declaration| {
                    declaration.visibility() == eqiora_lang::VisibilitySyntax::Public
                })
                .map(|declaration| declaration.name().to_owned())
                .collect::<std::collections::BTreeSet<_>>(),
        );
    }
    let mut result = definitions.clone();
    for unit in units {
        let visible = visible_spaces(&unit.module, aliases, &definitions, &public);
        let namespace =
            crate::enumeration::resolved_namespace(&unit.module).map_err(|e| vec![e])?;
        let products = spaces::declarations(
            &unit.file,
            &unit.document,
            &namespace,
            Some(&visible),
            |name| {
                unit.native
                    .as_ref()
                    .and_then(|module| module.nominal_identity(name))
            },
        )?;
        result
            .get_mut(&unit.module)
            .expect("known module")
            .extend(products);
    }
    Ok(result)
}

fn visible_spaces(
    module: &crate::resolved::CompilationModuleId,
    aliases: &[ResolvedAlias],
    definitions: &BTreeMap<
        crate::resolved::CompilationModuleId,
        BTreeMap<String, BoundFiniteSpace>,
    >,
    public: &BTreeMap<crate::resolved::CompilationModuleId, std::collections::BTreeSet<String>>,
) -> BTreeMap<String, BoundFiniteSpace> {
    let mut visible = definitions[module].clone();
    for alias in aliases
        .iter()
        .filter(|alias| alias.declaring_module() == module)
    {
        for (name, value) in &definitions[alias.target_module()] {
            if public[alias.target_module()].contains(name) {
                visible.insert(format!("{}.{}", alias.alias(), name), value.clone());
            }
        }
    }
    visible
}

pub(crate) fn bind_resolved(
    units: &mut [AnalyzedSourceUnit],
    aliases: &[ResolvedAlias],
) -> Result<(), Vec<Diagnostic>> {
    let definitions = resolved_spaces(units, aliases)?;
    let public = units
        .iter()
        .map(|unit| {
            (
                unit.module.clone(),
                unit.document
                    .finite_spaces()
                    .iter()
                    .filter(|d| d.visibility() == eqiora_lang::VisibilitySyntax::Public)
                    .map(|d| d.name().to_owned())
                    .collect(),
            )
        })
        .collect();
    for unit in units {
        let visible = visible_spaces(&unit.module, aliases, &definitions, &public);
        bind_finite_types(&unit.file, &mut unit.document, &visible)?;
        bind_finite_expressions(&unit.file, &mut unit.document, &visible)?;
        let namespace =
            crate::enumeration::resolved_namespace(&unit.module).map_err(|error| vec![error])?;
        bind_local_index_types(&unit.file, &mut unit.document, &namespace, |name| {
            unit.native
                .as_ref()
                .and_then(|module| module.nominal_identity(name))
        })?;
    }
    Ok(())
}
