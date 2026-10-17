//! Borrowed conservative coefficients from the sole compiled equation row.
use super::*;

pub(crate) struct ConservativeRow<'a, S: Coefficient> {
    row: &'a Row<S>,
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn conservative_row(
        &self,
    ) -> Result<ConservativeRow<'_, S>, Diagnostic> {
        let [row] = self.rows.as_slice() else {
            return Err(invalid("conservative assembly requires one tested row"));
        };
        let field_type = &self.roles.fields[&row.tested].1;
        if field_type.scalar_domain() != S::DOMAIN
            || !same_component_type(field_type, &row.value_type)
            || row.forcing.len() != components(field_type, self.dimension)?
            || !row.dyadics.is_empty()
            || row.terms.iter().any(|term| {
                term.trial != row.tested
                    || term.derivative
                    || !matches!(
                        term.pairing,
                        Pairing::Value | Pairing::ComponentValue { .. } | Pairing::Gradient
                    )
            })
        {
            return Err(invalid(
                "conservative assembly requires a steady isotropic-diffusion/component-reaction row in its exact coefficient domain",
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
                "conservative assembly requires its complete isotropic gradient flux",
            ));
        }
        Ok(ConservativeRow { row })
    }
}

impl<S: Coefficient> ConservativeRow<'_, S> {
    pub(crate) fn field(&self) -> RawId {
        self.row.tested
    }

    pub(crate) fn has_reaction(&self) -> bool {
        self.row.terms.iter().any(|term| {
            matches!(
                term.pairing,
                Pairing::Value | Pairing::ComponentValue { .. }
            )
        })
    }

    pub(crate) fn has_positive_diffusion(&self) -> bool {
        self.row
            .terms
            .iter()
            .filter(|term| term.pairing == Pairing::Gradient)
            .all(|term| term.positive_diffusion)
    }

    pub(crate) fn diffusion(&self, point: &[f64]) -> Result<S, Diagnostic> {
        self.coefficient(|pairing| pairing == Pairing::Gradient, point)
    }

    pub(crate) fn reaction(
        &self,
        test: usize,
        trial: usize,
        point: &[f64],
    ) -> Result<S, Diagnostic> {
        if test >= self.components() || trial >= self.components() {
            return Err(invalid(
                "conservative reaction component is outside its exact Field",
            ));
        }
        self.coefficient(
            |pairing| {
                (test == trial && pairing == Pairing::Value)
                    || pairing == Pairing::ComponentValue { test, trial }
            },
            point,
        )
    }

    pub(crate) fn components(&self) -> usize {
        self.row.forcing.len()
    }

    pub(crate) fn source(&self, component: usize, point: &[f64]) -> Result<S, Diagnostic> {
        self.row
            .forcing
            .get(component)
            .ok_or_else(|| invalid("conservative source component is outside its exact Field"))?
            .evaluate(point)
    }

    fn coefficient(
        &self,
        select: impl Fn(Pairing) -> bool,
        point: &[f64],
    ) -> Result<S, Diagnostic> {
        let mut value = <S as From<f64>>::from(0.0);
        for term in self.row.terms.iter().filter(|term| select(term.pairing)) {
            value += term.require_coefficient(term.coefficient.evaluate(point)?)?;
        }
        if !value.is_finite() {
            return Err(invalid("conservative coefficient sum is non-finite"));
        }
        Ok(value)
    }
}
