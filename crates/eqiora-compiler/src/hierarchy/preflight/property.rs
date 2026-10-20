//! One property catalog with explicit caller/callee lexical namespaces.
use super::*;

impl Elaborator<'_> {
    pub(in crate::hierarchy) fn bind_instance_property_values(
        &self,
        component: &ComponentDefinition<'_>,
        instance: &eqiora_lang::InstanceDecl,
        caller: (&DefinitionNamespace, &str),
        parent: &super::super::parameters::SymbolicParameterMap,
        values: &mut super::super::parameters::SymbolicParameterMap,
    ) -> Result<(), Vec<Diagnostic>> {
        for item in component.declaration.signature() {
            let eqiora_lang::SignatureItem::Property(requirement) = item else {
                continue;
            };
            let contract = self
                .resolve_property_contract(
                    &component.namespace,
                    requirement.contract(),
                    component.file,
                )
                .map_err(|error| vec![error])?;
            if !contract.inputs.is_empty() {
                continue;
            }
            let Some(binding) = instance
                .bindings()
                .iter()
                .find(|binding| binding.name() == requirement.name())
            else {
                continue;
            };
            let value = if let eqiora_lang::ExprKind::Name(name) = binding.value().kind()
                && let Some(value) = parent.get(name)
            {
                value.clone()
            } else {
                let release = self.bind_property(
                    &component.namespace,
                    requirement,
                    caller.0,
                    binding.value(),
                    caller.1,
                )?;
                let Some(value) = release.meaning().constant_value() else {
                    continue;
                };
                super::super::parameters::SymbolicParameterValue {
                    value: Some(value.clone()),
                    value_type: value.value_type().clone(),
                    expression: None,
                    lineage: None,
                }
            };
            values.insert(requirement.name().to_owned(), value);
        }
        Ok(())
    }

    pub(in crate::hierarchy) fn bind_symbolic_properties(
        &self,
        namespace: &DefinitionNamespace,
        file: &str,
        signature: &[eqiora_lang::SignatureItem],
        values: &mut super::super::parameters::SymbolicParameterMap,
    ) -> Result<(), Vec<Diagnostic>> {
        for item in signature {
            let eqiora_lang::SignatureItem::Property(requirement) = item else {
                continue;
            };
            let contract = self
                .resolve_property_contract(namespace, requirement.contract(), file)
                .map_err(|error| vec![error])?;
            if !contract.inputs.is_empty() {
                continue;
            }
            let value_type = contract
                .result_type::<String>(None)
                .map_err(|error| vec![error])?;
            values.entry(requirement.name().to_owned()).or_insert(
                super::super::parameters::SymbolicParameterValue {
                    value: None,
                    value_type,
                    expression: None,
                    lineage: None,
                },
            );
        }
        Ok(())
    }

    fn property_namespace<'a>(
        &'a self,
        namespace: &'a DefinitionNamespace,
    ) -> Result<&'a crate::resolved::CompilationModuleId, Diagnostic> {
        match namespace {
            DefinitionNamespace::Resolved(module) => Ok(module),
            DefinitionNamespace::Local => self.property_local_module.as_ref().ok_or_else(|| {
                super::super::hierarchy_error("property scope has no exact local module")
            }),
        }
    }

    pub(in crate::hierarchy) fn resolve_property_contract(
        &self,
        namespace: &DefinitionNamespace,
        path: &eqiora_lang::NamePath,
        file: &str,
    ) -> Result<&crate::property::Contract, Diagnostic> {
        self.property_catalog.contract(
            self.property_namespace(namespace)?,
            path,
            &self.property_aliases,
            file,
        )
    }

    pub(in crate::hierarchy) fn property_contract_identity(
        &self,
        namespace: &DefinitionNamespace,
        path: &eqiora_lang::NamePath,
        file: &str,
    ) -> Result<String, Diagnostic> {
        self.property_catalog.contract_identity(
            self.property_namespace(namespace)?,
            path,
            &self.property_aliases,
            file,
        )
    }

    pub(in crate::hierarchy) fn bind_property(
        &self,
        requirement_namespace: &DefinitionNamespace,
        requirement: &eqiora_lang::ComponentPropertyDecl,
        binding_namespace: &DefinitionNamespace,
        value: &eqiora_lang::Expr,
        file: &str,
    ) -> Result<eqiora_schema::kernel::PropertyRelease, Vec<Diagnostic>> {
        self.property_catalog.bind(
            self.property_namespace(requirement_namespace)
                .map_err(|error| vec![error])?,
            requirement,
            self.property_namespace(binding_namespace)
                .map_err(|error| vec![error])?,
            value,
            &self.property_aliases,
            file,
        )
    }
}
