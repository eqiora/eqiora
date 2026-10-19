//! One degree authority for preparation, execution, and portable Plan quadrature.
use super::*;
use eqiora_meshing::ReferenceCellFamily;
use eqiora_realization::SpaceFamily;

impl<S: Coefficient> BoundRegionForm<S> {
    /// Degree of the basis products in the bound weak residual and Jacobian.
    pub(super) fn required_polynomial_exactness(&self) -> usize {
        let mut exactness = if self
            .fields
            .iter()
            .any(|layout| layout.space.family() == SpaceFamily::SimplexP1Bubble)
        {
            2 * (self.form.dimension + 1)
        } else if self.reference.family() == ReferenceCellFamily::Hypercube {
            3
        } else {
            2
        };
        if self.form.rows.iter().any(|row| !row.dyadics.is_empty()) {
            let degree = if self
                .fields
                .iter()
                .any(|layout| layout.space.family() == SpaceFamily::SimplexP1Bubble)
            {
                3 * (self.form.dimension + 1) - 1
            } else if self.reference.family() == ReferenceCellFamily::Hypercube {
                3 * self.form.dimension - 1
            } else {
                2
            };
            exactness = exactness.max(degree);
        }
        let simplex = self.reference.family() == ReferenceCellFamily::Simplex;
        let basis = if self
            .fields
            .iter()
            .any(|field| field.space.family() == SpaceFamily::SimplexP1Bubble)
        {
            self.form.dimension + 1
        } else {
            1usize
        };
        let gradient = if simplex || self.form.dimension == 1 {
            basis - 1
        } else {
            basis
        };
        for row in &self.form.rows {
            let mut include = |data: &super::Data<S>, basis_degree: usize| {
                if let Some(degree) = data.polynomial_degree() {
                    exactness = exactness.max(degree.saturating_add(basis_degree));
                }
                // A non-polynomial datum has no finite exactness guarantee.
                // Retain the admitted basis rule; its accuracy needs a separate study.
            };
            for data in &row.forcing {
                include(data, basis);
            }
            for term in &row.terms {
                let product = match term.pairing {
                    super::Pairing::Value => 2 * basis,
                    super::Pairing::Components { test, trial } => {
                        let degree = |component: crate::form_compiler::bilinear::BasisComponent| {
                            if component.requires_gradient() {
                                gradient
                            } else {
                                basis
                            }
                        };
                        degree(test) + degree(trial)
                    }
                    super::Pairing::Gradient
                    | super::Pairing::SymmetricGradient
                    | super::Pairing::Curl
                    | super::Pairing::Divergence => 2 * gradient,
                    super::Pairing::TestGradientTrialValue(_)
                    | super::Pairing::SkewGradient(_)
                    | super::Pairing::TestDivergenceTrialValue
                    | super::Pairing::TestValueTrialDivergence => basis + gradient,
                };
                include(&term.coefficient, product);
            }
            for term in &row.dyadics {
                include(&term.coefficient, 2 * basis + gradient);
            }
            for term in &row.components {
                if let Some(degree) = term.value.polynomial_degree(basis, gradient) {
                    include(
                        &term.coefficient,
                        degree.saturating_add(if term.test.requires_gradient() {
                            gradient
                        } else {
                            basis
                        }),
                    );
                }
            }
            for flux in &row.flux {
                if let super::flux::FluxTerm::Isotropic(data)
                | super::flux::FluxTerm::Prescribed { value: data, .. } = flux
                {
                    include(data, gradient);
                }
            }
        }
        exactness
    }

    /// Gauss points per axis covering the same degree checked during preparation.
    pub(crate) fn minimum_gauss_points(&self) -> usize {
        // Tensor Gauss has degree 2n-1; the simplex Duffy Jacobian changes
        // the total-degree guarantee to 2n-d.
        let loss = if self.reference.family() == ReferenceCellFamily::Simplex {
            self.form.dimension
        } else {
            1
        };
        self.required_polynomial_exactness()
            .saturating_add(loss)
            .div_ceil(2)
    }
}
