//! Borrowed conservative scalar coefficients from the sole compiled equation row.
use super::*;

pub(crate) struct ScalarConservativeRow<'a, S: Coefficient> {
    row: &'a Row<S>,
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn scalar_conservative_row(
        &self,
    ) -> Result<ScalarConservativeRow<'_, S>, Diagnostic> {
        let [row] = self.rows.as_slice() else {
            return Err(invalid(
                "scalar conservative assembly requires one tested row",
            ));
        };
        let field_type = &self.roles.fields[&row.tested].1;
        if !field_type.shape().is_scalar()
            || field_type.array_rank() != 0
            || field_type.frame() != ValueFrame::Invariant
            || field_type.scalar_domain() != S::DOMAIN
            || !row.value_type.shape().is_scalar()
            || row.forcing.len() != 1
            || !row.dyadics.is_empty()
            || row.terms.iter().any(|term| {
                term.trial != row.tested
                    || term.derivative
                    || !matches!(term.pairing, Pairing::Value | Pairing::Gradient)
            })
        {
            return Err(invalid(
                "scalar conservative assembly requires a steady scalar diffusion/reaction row in its exact coefficient domain",
            ));
        }
        let gradients = row
            .terms
            .iter()
            .filter(|term| term.pairing == Pairing::Gradient);
        if row.flux.is_empty()
            || !gradients.eq(row.flux.iter().filter_map(|flux| match flux {
                flux::FluxTerm::Trial(term) => Some(term),
                flux::FluxTerm::Isotropic(_) => None,
            }))
            || row.flux.iter().any(|flux| {
                !matches!(flux,
                flux::FluxTerm::Trial(term) if term.pairing == Pairing::Gradient)
            })
        {
            return Err(invalid(
                "scalar conservative assembly requires its complete isotropic gradient flux",
            ));
        }
        Ok(ScalarConservativeRow { row })
    }
}

impl<S: Coefficient> ScalarConservativeRow<'_, S> {
    pub(crate) fn field(&self) -> RawId {
        self.row.tested
    }

    pub(crate) fn has_reaction(&self) -> bool {
        self.row
            .terms
            .iter()
            .any(|term| term.pairing == Pairing::Value)
    }

    pub(crate) fn has_positive_diffusion(&self) -> bool {
        self.row
            .terms
            .iter()
            .filter(|term| term.pairing == Pairing::Gradient)
            .all(|term| term.positive_diffusion)
    }

    pub(crate) fn diffusion(&self, point: &[f64]) -> Result<S, Diagnostic> {
        self.coefficient(Pairing::Gradient, point)
    }

    pub(crate) fn reaction(&self, point: &[f64]) -> Result<S, Diagnostic> {
        self.coefficient(Pairing::Value, point)
    }

    pub(crate) fn source(&self, point: &[f64]) -> Result<S, Diagnostic> {
        self.row.forcing[0].evaluate(point)
    }

    fn coefficient(&self, pairing: Pairing, point: &[f64]) -> Result<S, Diagnostic> {
        let mut value = <S as From<f64>>::from(0.0);
        for term in self.row.terms.iter().filter(|term| term.pairing == pairing) {
            value += term.require_coefficient(term.coefficient.evaluate(point)?)?;
        }
        if !value.is_finite() {
            return Err(invalid("scalar conservative coefficient sum is non-finite"));
        }
        Ok(value)
    }
}
