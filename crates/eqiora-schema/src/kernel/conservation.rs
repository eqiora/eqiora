//! Physical conservation terms retained by their owning Relation.
//!
//! Fixed-domain outward-flux convention: d(storage)/dt + div(flux) = source.
//! These identities name mathematical expressions, never numerical face fluxes.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use super::{ExprDag, ExprId, ExprNode};

/// Exact physical terms of one fixed-domain conservation Law.
///
/// No storage means a steady balance. Source is explicit, including a typed zero.
/// Support, parameter dependencies, and source identity belong to the Relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConservationTerms {
    storage: Option<(ExprId, ExprId)>,
    flux: ExprId,
    source: ExprId,
}

impl ConservationTerms {
    /// Name terms in the owning Relation's expression arena.
    ///
    /// Construction of the Relation checks the corresponding balance roots;
    /// semantic admission checks physical dimensions, support and accumulation.
    #[must_use]
    pub const fn new(storage: Option<(ExprId, ExprId)>, flux: ExprId, source: ExprId) -> Self {
        Self {
            storage,
            flux,
            source,
        }
    }

    /// Stored quantity and accumulation, absent for an explicitly steady Law.
    #[must_use]
    pub const fn storage(self) -> Option<(ExprId, ExprId)> {
        self.storage
    }

    /// Physical outward flux, before divergence or numerical realization.
    #[must_use]
    pub const fn flux(self) -> ExprId {
        self.flux
    }

    /// Physical production per unit volume and time.
    #[must_use]
    pub const fn source(self) -> ExprId {
        self.source
    }

    /// Check exact structural correspondence with the retained balance equation.
    ///
    /// This deliberately does not recognize algebraically similar residuals.
    ///
    /// # Errors
    /// Rejects missing term nodes, extra equations, changed source, reversed flux,
    /// or missing/doubled accumulation in the corresponding balance.
    pub fn validate_balance(self, expression: &ExprDag) -> Result<(), Diagnostic> {
        let invalid = || {
            Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "conservation Law requires exact d(storage)/dt + div(outward flux) = source roots",
            )
        };
        let [left, right] = expression.roots() else {
            return Err(invalid());
        };
        if *right != self.source
            || expression.node(self.source).is_none()
            || expression.node(self.flux).is_none()
        {
            return Err(invalid());
        }
        let divergence = if let Some((stored, stored_accumulation)) = self.storage {
            if expression.node(stored).is_none() || expression.node(stored_accumulation).is_none() {
                return Err(invalid());
            }
            match expression.node(*left) {
                Some(ExprNode::Add(accumulation, divergence))
                    if *accumulation == stored_accumulation =>
                {
                    *divergence
                }
                _ => return Err(invalid()),
            }
        } else {
            *left
        };
        match expression.node(divergence) {
            Some(ExprNode::Divergence(flux)) if *flux == self.flux => Ok(()),
            _ => Err(invalid()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{ExprDagBuilder, SymbolRef};
    use eqiora_core::{Id, entity::kinds};

    #[test]
    fn steady_balance_retains_exact_flux_and_source_identities() {
        let mut builder = ExprDagBuilder::new();
        let field = builder
            .symbol(SymbolRef::Field(Id::<kinds::Field>::new()))
            .unwrap();
        let gradient = builder.gradient(field).unwrap();
        let flux = builder.neg(gradient).unwrap();
        let divergence = builder.divergence(flux).unwrap();
        let source = builder
            .symbol(SymbolRef::Parameter(Id::<kinds::Parameter>::new()))
            .unwrap();
        let other_source = builder
            .symbol(SymbolRef::Parameter(Id::<kinds::Parameter>::new()))
            .unwrap();
        let expression = builder.finish([divergence, source]).unwrap();
        assert!(
            ConservationTerms::new(None, flux, source)
                .validate_balance(&expression)
                .is_ok()
        );
        assert!(
            ConservationTerms::new(None, gradient, source)
                .validate_balance(&expression)
                .is_err()
        );
        assert!(
            ConservationTerms::new(None, flux, other_source)
                .validate_balance(&expression)
                .is_err()
        );
    }

    #[test]
    fn transient_balance_rejects_missing_or_doubled_accumulation() {
        let mut builder = ExprDagBuilder::new();
        let field_id = Id::<kinds::Field>::new();
        let stored = builder.symbol(SymbolRef::Field(field_id)).unwrap();
        let accumulation = builder.symbol(SymbolRef::Derivative(field_id)).unwrap();
        let gradient = builder.gradient(stored).unwrap();
        let flux = builder.neg(gradient).unwrap();
        let divergence = builder.divergence(flux).unwrap();
        let left = builder.add(accumulation, divergence).unwrap();
        let doubled = builder.add(accumulation, accumulation).unwrap();
        let source = builder
            .symbol(SymbolRef::Parameter(Id::<kinds::Parameter>::new()))
            .unwrap();
        let expression = builder.finish([left, source]).unwrap();
        let terms = ConservationTerms::new(Some((stored, accumulation)), flux, source);
        assert!(terms.validate_balance(&expression).is_ok());
        assert!(
            ConservationTerms::new(None, flux, source)
                .validate_balance(&expression)
                .is_err()
        );
        assert!(
            ConservationTerms::new(Some((stored, doubled)), flux, source)
                .validate_balance(&expression)
                .is_err()
        );
    }
}
