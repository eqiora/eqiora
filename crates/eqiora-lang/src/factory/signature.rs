//! Validation shared by Model and Component external signatures.
use super::*;
use crate::{ComponentPropertyDecl, SignatureItem};

pub(super) fn validate_signature(items: &[SignatureItem]) -> Result<(), AstConstructionError> {
    for item in items {
        checked_range(item.range())?;
        validate_identifier(item.name(), "signature name")?;
        match item {
            SignatureItem::Expression(value)
                if value.role() != crate::FieldRoleSyntax::Variable
                    || value.spatial_regularity()
                        != eqiora_schema::kernel::SpatialRegularity::Unspecified =>
            {
                return Err(AstConstructionError::new(
                    "expression slots declare value types and support, not Field roles or regularity",
                ));
            }
            SignatureItem::Input(value) | SignatureItem::Output(value)
                if value.role() != crate::FieldRoleSyntax::Variable =>
            {
                return Err(AstConstructionError::new(
                    "causal input/output payload must have algebraic value role",
                ));
            }
            SignatureItem::Port(value) if matches!(value.syntax(), PortSyntax::Signal { .. }) => {
                return Err(AstConstructionError::new(
                    "causal interfaces use input/output signature entries",
                ));
            }
            SignatureItem::Parameter(value) if value.visibility() != VisibilitySyntax::Public => {
                return Err(AstConstructionError::new(
                    "signature parameters are public requirements",
                ));
            }
            SignatureItem::Port(value) if value.visibility() != VisibilitySyntax::Public => {
                return Err(AstConstructionError::new(
                    "signature ports are exposed owned endpoints",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

impl SourceAstFactory {
    /// Construct a nominal property requirement in a shared signature.
    ///
    /// # Errors
    /// Rejects malformed names, paths, and ranges.
    pub fn property_requirement(
        name: impl Into<String>,
        contract: NamePath,
        range: TextRange,
    ) -> Result<ComponentPropertyDecl, AstConstructionError> {
        validate_name_path(&contract)?;
        Ok(ComponentPropertyDecl {
            comments: Default::default(),
            name: checked_identifier(name, "property requirement")?,
            contract,
            range: checked_range(range)?,
        })
    }
}
