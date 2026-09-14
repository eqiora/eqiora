//! Physical conservation syntax, before any numerical method is selected.

use super::Expr;

/// Fixed-domain storage, outward flux, and volumetric production expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct ConservationSyntax {
    pub(crate) storage: Option<Expr>,
    pub(crate) flux: Expr,
    pub(crate) source: Expr,
}

impl ConservationSyntax {
    /// Stored expression; omission explicitly requests a steady Law.
    #[must_use]
    pub const fn storage(&self) -> Option<&Expr> {
        self.storage.as_ref()
    }

    /// Physical outward flux before divergence.
    #[must_use]
    pub const fn flux(&self) -> &Expr {
        &self.flux
    }

    /// Production term, including an explicitly authored zero.
    #[must_use]
    pub const fn source(&self) -> &Expr {
        &self.source
    }
}
