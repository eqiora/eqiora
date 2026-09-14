//! Checked source construction for physical conservation Laws.

use super::*;
use crate::ast::{ConservationSyntax, RelationBody};

impl SourceAstFactory {
    /// Construct a fixed-domain conservation Law with an outward physical flux.
    ///
    /// Omit storage only for a steady balance. Supply source explicitly, using
    /// a typed zero where production is absent. Mathematical admission belongs
    /// to the compiler and kernel, independently of a numerical method.
    ///
    /// # Errors
    /// Rejects malformed names, term expressions, or source ranges.
    pub fn law(
        name: impl Into<String>,
        domain: impl Into<String>,
        storage: Option<Expr>,
        flux: Expr,
        source: Expr,
        range: TextRange,
    ) -> Result<RelationDecl, AstConstructionError> {
        if let Some(value) = &storage {
            validate_expression(value)?;
        }
        validate_expression(&flux)?;
        validate_expression(&source)?;
        Ok(RelationDecl {
            comments: Default::default(),
            name: checked_identifier(name, "Law")?,
            activation: ActivationSyntax::Continuous,
            domain: Some(checked_identifier(domain, "Law support")?),
            body: RelationBody::Conservation(Box::new(ConservationSyntax {
                storage,
                flux,
                source,
            })),
            range: checked_range(range)?,
        })
    }
}
