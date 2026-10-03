//! Exact native nominal declaration lookup in the existing draft scope.

use super::*;

impl ModelDeclarations {
    pub(super) fn product_name(&self, basis: eqiora_core::FiniteBasis) -> Option<NamePath> {
        self.declarations
            .iter()
            .find_map(|declaration| match declaration {
                DraftDeclaration::FiniteSpace { name, definition }
                    if definition.basis() == basis =>
                {
                    Some(NamePath::single(name.clone(), TextRange::new(0, 0)))
                }
                _ => None,
            })
    }

    pub(super) fn validate_space(
        &self,
        name: &str,
        definition: &eqiora_schema::kernel::FiniteSpaceDef,
    ) -> Result<(), crate::AstConstructionError> {
        if let Some(labels) = definition.labels() {
            crate::SourceAstFactory::finite_space(
                crate::VisibilitySyntax::Private,
                name,
                labels.to_vec(),
                TextRange::new(0, 0),
            )?;
        }
        if let Some(factors) = definition.factors() {
            for factor in factors {
                if !self.declarations.iter().any(|declaration| matches!(declaration,
                    DraftDeclaration::FiniteSpace { definition, .. } if definition.basis() == factor)) {
                    return Err(crate::AstConstructionError::new("product factor references an omitted or foreign atomic declaration"));
                }
            }
        }
        Ok(())
    }

    pub(super) fn validate_enum_type(
        &self,
        value: &ValueType,
    ) -> Result<(), crate::AstConstructionError> {
        if let Some(id) = value.enum_definition() {
            let definition = self.enum_definition(id.erase()).ok_or_else(|| {
                crate::AstConstructionError::new(
                    "enum type references an omitted or foreign declaration",
                )
            })?;
            if &definition.value_type() != value {
                return Err(crate::AstConstructionError::new(
                    "enum type differs from its registered declaration",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn enum_definition(
        &self,
        id: eqiora_core::RawId,
    ) -> Option<&eqiora_schema::kernel::EnumDef> {
        self.declarations
            .iter()
            .find_map(|declaration| match declaration {
                DraftDeclaration::Enum { definition, .. } if definition.id().erase() == id => {
                    Some(definition)
                }
                _ => None,
            })
    }

    pub(super) fn nominal_name(&self, id: eqiora_core::RawId) -> Option<NamePath> {
        self.declarations
            .iter()
            .find_map(|declaration| match declaration {
                DraftDeclaration::Enum { name, definition } if definition.id().erase() == id => {
                    Some(NamePath::single(name.clone(), TextRange::new(0, 0)))
                }
                DraftDeclaration::FiniteSpace { name, definition }
                    if definition.id().erase() == id =>
                {
                    Some(NamePath::single(name.clone(), TextRange::new(0, 0)))
                }
                DraftDeclaration::IndexSet { name, definition }
                    if definition.id().erase() == id =>
                {
                    Some(NamePath::single(name.clone(), TextRange::new(0, 0)))
                }
                _ => None,
            })
    }
}

pub(super) fn validate_enum(
    name: &str,
    definition: &eqiora_schema::kernel::EnumDef,
) -> Result<(), crate::AstConstructionError> {
    let range = TextRange::new(0, 0);
    let tags = definition
        .members()
        .iter()
        .map(|tag| NamePath::from_segments([tag.as_str()], range))
        .collect::<Result<_, _>>()?;
    crate::SourceAstFactory::enumeration(crate::VisibilitySyntax::Private, name, tags, range)
        .map(|_| ())
}
