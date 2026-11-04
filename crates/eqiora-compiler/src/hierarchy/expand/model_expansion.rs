//! Expand the selected root through lexical values, child interfaces and materialization.
use super::*;

impl RootExpansion<'_, '_> {
    pub(in crate::hierarchy) fn expand(self) -> Result<ExpandedBlueprint, Vec<Diagnostic>> {
        self.expand_bound(&[], &[], &BTreeMap::new())
    }

    pub(in crate::hierarchy) fn expand_bound(
        mut self,
        supports: &[crate::external::ExternalSupportBinding],
        clocks: &[(String, eqiora_schema::kernel::ClockDomainDef)],
        properties: &BTreeMap<String, std::sync::Arc<eqiora_schema::kernel::PropertyRelease>>,
    ) -> Result<ExpandedBlueprint, Vec<Diagnostic>> {
        let model = self.model.clone();
        let mut root_scope = Scope::default();
        root_scope.extend_properties(properties);
        root_scope.record_context =
            super::parameters::RecordContext::model(self.elaborator, &model);
        root_scope.reduction_terms_limit = self.elaborator.limits.max_parameter_terms;
        root_scope.lexical_namespace = Some(model.namespace.clone());
        root_scope.set_pure_operators(self.elaborator.visible_pure_operators(&model.namespace));
        self.allocate_external_clocks(&mut root_scope, clocks)
            .map_err(one_diagnostic)?;
        self.allocate_external_supports(&mut root_scope, supports)
            .map_err(one_diagnostic)?;
        let identities = match self.allocate_model_scope(&mut root_scope) {
            Ok(value) => value,
            Err(error) => return Err(vec![error]),
        };

        self.allocate_model_expression_bindings(&mut root_scope, &model, true)?;
        self.expand_model_children(&model, &mut root_scope)?;

        self.allocate_model_expression_bindings(&mut root_scope, &model, false)?;

        if let Err(error) = self.materialize_model_items(&root_scope, &identities) {
            return Err(vec![error]);
        }
        self.record_index_dependencies(&root_scope)
            .map_err(one_diagnostic)?;
        if let Err(error) = self.finalize_physical_connections() {
            return Err(vec![error]);
        }
        self.expand_integral_partials().map_err(one_diagnostic)?;
        self.items.sort_by_key(FlatItemBlueprint::sort_key);
        Ok(ExpandedBlueprint::new(
            self.model.name().to_owned(),
            SourceLocation::new(self.model.file, self.model.range()),
            (self.model_key, self.model_full),
            self.items,
            self.display_symbols,
            self.physical_exposures,
            self.notation_specs,
        )
        .with_structural_dependencies(self.structural_dependencies))
    }
}
