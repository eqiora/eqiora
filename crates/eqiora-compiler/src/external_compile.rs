//! One signature-directed public compilation API and exact Geometry checks.
use crate::diagnostics::source_error;
use crate::lower::CompiledModel;
use crate::resolved::ValidatedResolvedHierarchy;
use crate::source_identity::LocalSourceIdentityLimits;
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};
use eqiora_lang::TextRange;
use std::collections::BTreeMap;

pub(crate) fn validate_selected_bindings(
    file: &str,
    entry: &str,
    bindings: &[(&str, crate::StaticBindingValue<'_>)],
) -> Result<(), Vec<Diagnostic>> {
    let limits = LocalSourceIdentityLimits::default();
    if bindings.len() > limits.max_bindings_per_instance {
        return Err(vec![source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            TextRange::default(),
            "selected binding count exceeds resource limit",
        )]);
    }
    let mut total = 0;
    observe_external_name(file, "entry", entry, limits, &mut total)?;
    for (name, value) in bindings {
        observe_external_name(file, "binding", name, limits, &mut total)?;
        if let crate::StaticBindingValue::CompleteExterior {
            members, parent, ..
        } = value
        {
            if members.len()
                > crate::hierarchy::CompleteExteriorLimits::default().max_members_per_set
            {
                return Err(vec![source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    file,
                    TextRange::default(),
                    "complete-exterior member count exceeds resource limit",
                )]);
            }
            observe_external_name(file, "Geometry parent", parent.name(), limits, &mut total)?;
            for member in *members {
                observe_external_name(file, "Geometry member", member.name(), limits, &mut total)?;
            }
        }
        if let crate::StaticBindingValue::GeometrySupport {
            selection, parent, ..
        } = value
        {
            observe_external_name(
                file,
                "Geometry selection",
                selection.name(),
                limits,
                &mut total,
            )?;
            if let Some(parent) = parent {
                observe_external_name(file, "Geometry parent", parent.name(), limits, &mut total)?;
            }
        }
    }
    Ok(())
}

fn observe_external_name(
    file: &str,
    label: &str,
    name: &str,
    limits: LocalSourceIdentityLimits,
    total: &mut usize,
) -> Result<(), Vec<Diagnostic>> {
    if name.len() > limits.max_name_bytes {
        return Err(vec![source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            TextRange::default(),
            format!(
                "external {label} name requires {} UTF-8 bytes, exceeding the {} byte name limit",
                name.len(),
                limits.max_name_bytes
            ),
        )]);
    }
    *total = total.checked_add(name.len()).ok_or_else(|| {
        vec![source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            TextRange::default(),
            "external binding name bytes overflow usize",
        )]
    })?;
    if *total > limits.max_total_name_bytes {
        return Err(vec![source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            TextRange::default(),
            format!(
                "external binding names require {total} UTF-8 bytes, exceeding the {} byte aggregate name limit",
                limits.max_total_name_bytes
            ),
        )]);
    }
    Ok(())
}

#[allow(
    clippy::type_complexity,
    reason = "the closed tuple avoids a second public selection lifecycle"
)]
pub(crate) fn selected_geometry_parent<'a>(
    explicit: Option<&'a NamedEntitySet>,
    parent_slot: &str,
    bindings: &[(&str, crate::StaticBindingValue<'a>)],
) -> Option<&'a NamedEntitySet> {
    explicit.or_else(|| {
        bindings.iter().find_map(|(name, value)| match value {
            crate::StaticBindingValue::GeometrySupport { selection, .. }
                if *name == parent_slot =>
            {
                Some(*selection)
            }
            _ => None,
        })
    })
}

pub(crate) fn validate_geometry_bindings(
    file: &str,
    geometry: &CanonicalGeometryV1,
    supports: &[(&str, &NamedEntitySet, Option<(&str, &NamedEntitySet)>)],
) -> Result<(), Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let mut regions = BTreeMap::new();
    for (slot, selection, parent) in supports {
        if parent.is_none()
            && geometry.selection_dimension(selection) == Some(geometry.topological_dimension())
        {
            regions
                .entry(*slot)
                .and_modify(|selection| *selection = None)
                .or_insert(Some(*selection));
        }
    }
    for (slot, selection, parent) in supports {
        let Some(dimension) = geometry.selection_dimension(selection) else {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                TextRange::default(),
                format!("external support `{slot}` has a foreign or stale Geometry selection"),
            ));
            continue;
        };
        match parent {
            None if dimension != geometry.topological_dimension() => diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                TextRange::default(),
                format!(
                    "external region support `{slot}` has selection dimension {dimension}, expected {}",
                    geometry.topological_dimension(),
                ),
            )),
            None => {}
            Some((parent_slot, parent_selection)) => {
                if geometry.selection_dimension(parent_selection).is_none() {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        TextRange::default(),
                        format!(
                            "external boundary support `{slot}` has a foreign or stale parent selection"
                        ),
                    ));
                    continue;
                }
                let exact_parent = regions
                    .get(parent_slot)
                    .and_then(|selection| *selection)
                    .is_some_and(|selection| std::ptr::eq(selection, *parent_selection));
                if !exact_parent
                    || !geometry.selection_is_boundary_of(selection, parent_selection)
                {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        TextRange::default(),
                        format!(
                            "external boundary selection `{}` does not bind exact parent region slot `{parent_slot}` in the supplied Geometry revision",
                            selection.name(),
                        ),
                    ));
                }
            }
        }
    }
    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(diagnostics)
    }
}

impl CompiledModel {
    /// Compile one explicitly selected Model or public Component with named static arguments.
    ///
    /// # Errors
    /// Rejects unknown, duplicate, missing or incorrectly typed signature bindings.
    pub fn compile_selected(
        file: &str,
        source: &str,
        entry: &str,
        bindings: &[(&str, crate::StaticBindingValue<'_>)],
    ) -> Result<Self, Vec<Diagnostic>> {
        validate_selected_bindings(file, entry, bindings)?;
        crate::hierarchy::selected::local(file, source, entry, bindings)
    }
}
impl ValidatedResolvedHierarchy {
    /// Compile one selected root or directly imported definition with named static arguments.
    ///
    /// # Errors
    /// Returns signature binding, hierarchy, or mathematical lowering diagnostics.
    pub fn compile_selected(
        &self,
        entry: &str,
        bindings: &[(&str, crate::StaticBindingValue<'_>)],
    ) -> Result<CompiledModel, Vec<Diagnostic>> {
        validate_selected_bindings("<selected-entry>", entry, bindings)?;
        crate::hierarchy::selected::resolved(self, entry, bindings)
            .map(|model| self.analysis.retain_authored_provenance(model))
            .map_err(|errors| self.analysis.native_diagnostics(errors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unified_count_fails_before_source_or_binding_allocation() {
        let value = eqiora_lang::SourceAstFactory::expression(
            eqiora_lang::ExprKind::Number(
                eqiora_lang::DecimalLiteral::parse("1.0").expect("exact literal"),
            ),
            TextRange::default(),
        )
        .unwrap();
        let bindings = (0..=LocalSourceIdentityLimits::default().max_bindings_per_instance)
            .map(|_| ("value", crate::StaticBindingValue::Expression(&value)))
            .collect::<Vec<_>>();
        let errors =
            CompiledModel::compile_selected("oversized.eqi", "not valid source", "Law", &bindings)
                .unwrap_err();
        assert!(errors[0].message().contains("selected binding count"));
    }
    #[test]
    fn oversized_entry_fails_before_source_allocation() {
        let entry = "m".repeat(LocalSourceIdentityLimits::default().max_name_bytes + 1);
        let errors =
            CompiledModel::compile_selected("oversized.eqi", "not valid source", &entry, &[])
                .unwrap_err();
        assert!(errors[0].message().contains("byte name limit"));
    }
    #[test]
    fn aggregate_external_name_budget_fails_without_materializing_all_names() {
        let limits = LocalSourceIdentityLimits::default();
        let mut total = limits.max_total_name_bytes - 1;
        let errors = observe_external_name("aggregate.eqi", "binding", "ab", limits, &mut total)
            .unwrap_err();
        assert!(errors[0].message().contains("aggregate name limit"));
    }
}
