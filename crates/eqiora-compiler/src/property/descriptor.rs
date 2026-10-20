//! Read-only property signatures after the ordinary resolved-source admission.
use eqiora_core::{Diagnostic, ValueType};

/// Exact typed public property contract in an already admitted module graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyContractDescriptor {
    value_type: ValueType,
    inputs: Vec<(String, ValueType)>,
    derivatives: eqiora_schema::kernel::PropertyDerivatives,
    branch: Option<String>,
}

impl PropertyContractDescriptor {
    /// Complete result type owned by the common mathematical type vocabulary.
    #[must_use]
    pub const fn value_type(&self) -> &ValueType {
        &self.value_type
    }
    /// Named independent inputs in exact declaration order.
    #[must_use]
    pub fn inputs(&self) -> &[(String, ValueType)] {
        &self.inputs
    }
    /// Required ordinary first formal partials, holding all other inputs fixed.
    #[must_use]
    pub const fn derivatives(&self) -> eqiora_schema::kernel::PropertyDerivatives {
        self.derivatives
    }
    /// Exact declared phase branch.
    #[must_use]
    pub fn branch(&self) -> Option<&str> {
        self.branch.as_deref()
    }
}

impl crate::AnalyzedResolvedHierarchy {
    /// Inspect a public property contract in the exact selected root module.
    /// No release value, Model occurrence, or substitute material is constructed.
    ///
    /// # Errors
    /// Rejects missing/private contracts and unresolved or unsupported full types.
    pub fn property_contract_descriptor(
        &self,
        name: &str,
    ) -> Result<PropertyContractDescriptor, Diagnostic> {
        let path = eqiora_lang::NamePath::from_segments([name], eqiora_lang::TextRange::default())
            .map_err(|error| {
                super::error(
                    "<property-descriptor>",
                    eqiora_lang::TextRange::default(),
                    error.to_string(),
                )
            })?;
        let contract = self.property_catalog.contract(
            &self.root,
            &path,
            &self.aliases,
            "<property-descriptor>",
        )?;
        if contract.visibility != eqiora_lang::VisibilitySyntax::Public {
            return Err(super::error(
                &contract.file,
                path.range(),
                "imported property contract must be public",
            ));
        }
        let value_type = contract.result_type::<()>(None)?;
        let inputs = contract
            .inputs
            .iter()
            .map(|(name, value_type)| {
                crate::value_types::lower_signature_type::<()>(&contract.file, value_type, None)
                    .map(|value| (name.clone(), value))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PropertyContractDescriptor {
            value_type,
            inputs,
            derivatives: contract.derivatives,
            branch: contract.branch.clone(),
        })
    }
}
