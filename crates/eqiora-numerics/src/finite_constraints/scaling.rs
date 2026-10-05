//! Physical normalization of original equality rows, shared by residuals and AD.
use super::*;
use eqiora_realization::PositivePhysicalScale;
use std::collections::BTreeSet;

pub(super) fn normalize(value: f64, scale: f64) -> Result<f64, Diagnostic> {
    let normalized = value / scale;
    if !normalized.is_finite() || (value != 0.0 && normalized == 0.0) {
        return Err(invalid("scaled finite value is not representable"));
    }
    Ok(normalized)
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct EqualityScale {
    pub ordinal: u32,
    pub scale: PositivePhysicalScale,
    pub coordinates: usize,
}

impl RelationOperands {
    pub(super) fn row_scales(&self) -> impl Iterator<Item = f64> + '_ {
        self.equality_scales.iter().flat_map(|entry| {
            std::iter::repeat_n(entry.scale.quantity().value(), entry.coordinates)
        })
    }
}

impl FiniteConstraintProblem {
    pub(crate) fn residual_scales(&self) -> Vec<(ConstraintRef, PositivePhysicalScale)> {
        let mut entries: Vec<_> = self
            .relations
            .iter()
            .flat_map(|relation| {
                relation
                    .equality_scales
                    .iter()
                    .map(|entry| (ConstraintRef::new(relation.id, entry.ordinal), entry.scale))
            })
            .collect();
        entries.sort_by_key(|(reference, _)| *reference);
        entries
    }

    /// Explicit overrides use original condition identities and physical dimensions.
    /// Omitted equalities retain one coherent-SI unit per original operand dimension.
    pub(crate) fn with_residual_scales(
        mut self,
        scales: &[(ConstraintRef, PositivePhysicalScale)],
    ) -> Result<Self, Diagnostic> {
        if !self.is_nonlinear() && !scales.is_empty() {
            return Err(invalid("residual scaling requires finite Newton admission"));
        }
        let mut seen = BTreeSet::new();
        for (reference, scale) in scales {
            if !seen.insert(*reference) {
                return Err(invalid("residual scales repeat an exact equality"));
            }
            let entry = self.relations.iter_mut().find_map(|relation| {
                (relation.id == reference.relation())
                    .then(|| {
                        relation
                            .equality_scales
                            .iter_mut()
                            .find(|entry| entry.ordinal == reference.ordinal())
                    })
                    .flatten()
            });
            let Some(entry) = entry else {
                return Err(invalid("residual scale does not name an original equality"));
            };
            if entry.scale.quantity().dim() != scale.quantity().dim() {
                return Err(invalid(
                    "residual scale has incompatible physical dimensions",
                ));
            }
            entry.scale = *scale;
        }
        Ok(self)
    }
}
