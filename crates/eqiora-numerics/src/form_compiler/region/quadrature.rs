//! One degree authority for preparation, execution, and portable Plan quadrature.
use super::*;
use eqiora_meshing::ReferenceCellFamily;
use eqiora_realization::SpaceFamily;

impl<S: Coefficient> BoundRegionForm<S> {
    /// Degree of the basis products in the bound weak residual and Jacobian.
    pub(super) fn required_polynomial_exactness(&self) -> usize {
        let maximum_basis = self
            .fields
            .iter()
            .map(|field| self.field_basis_degree(field.field))
            .max()
            .unwrap_or(0);
        let mut exactness = if maximum_basis == 0 {
            0
        } else if maximum_basis == 1 {
            if self.reference.family() == ReferenceCellFamily::Hypercube {
                3
            } else {
                2
            }
        } else if self
            .fields
            .iter()
            .any(|field| field.space.family() == SpaceFamily::SimplexP1Bubble)
        {
            2 * (self.form.dimension + 1)
        } else {
            0
        };
        if self.form.rows.iter().any(|row| !row.dyadics.is_empty()) {
            let floor = if self
                .fields
                .iter()
                .any(|field| field.space.family() == SpaceFamily::SimplexP1Bubble)
            {
                3 * (self.form.dimension + 1) - 1
            } else if self.reference.family() == ReferenceCellFamily::Hypercube {
                3 * self.form.dimension - 1
            } else {
                2
            };
            exactness = exactness.max(floor);
        }
        for row in &self.form.rows {
            let basis = self.field_basis_degree(row.tested);
            let gradient = self.field_gradient_degree(row.tested);
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
                let trial_basis = self.field_basis_degree(term.trial);
                let trial_gradient = self.field_gradient_degree(term.trial);
                let product = match term.pairing {
                    super::Pairing::Value => basis + trial_basis,
                    super::Pairing::Components { test, trial } => {
                        let degree = |component: crate::form_compiler::bilinear::BasisComponent,
                                      value,
                                      derivative| {
                            if component.requires_gradient() {
                                derivative
                            } else {
                                value
                            }
                        };
                        degree(test, basis, gradient) + degree(trial, trial_basis, trial_gradient)
                    }
                    super::Pairing::Gradient
                    | super::Pairing::SymmetricGradient
                    | super::Pairing::Curl
                    | super::Pairing::Divergence => gradient + trial_gradient,
                    super::Pairing::TestGradientTrialValue(_)
                    | super::Pairing::TestDivergenceTrialValue => gradient + trial_basis,
                    super::Pairing::TestValueTrialDivergence => basis + trial_gradient,
                    super::Pairing::SkewGradient(_) => {
                        (gradient + trial_basis).max(basis + trial_gradient)
                    }
                };
                include(&term.coefficient, product);
            }
            for term in &row.dyadics {
                include(
                    &term.coefficient,
                    2 * self.field_basis_degree(term.field) + gradient,
                );
            }
            for term in &row.components {
                if let Some(degree) = term.value.polynomial_degree(|field, derivative| {
                    if derivative {
                        self.field_gradient_degree(field)
                    } else {
                        self.field_basis_degree(field)
                    }
                }) {
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

    pub(super) fn field_basis_degree(&self, field: eqiora_core::RawId) -> usize {
        let field = self.eliminations.get(&field).copied().unwrap_or(field);
        match self
            .fields
            .iter()
            .find(|layout| layout.field == field)
            .expect("bound Field degree")
            .space
            .family()
        {
            SpaceFamily::CellConstant => 0,
            SpaceFamily::ContinuousLagrange { order } => usize::from(order.get()),
            SpaceFamily::SimplexP1Bubble => self.form.dimension + 1,
            SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace => 1,
        }
    }

    fn field_gradient_degree(&self, field: eqiora_core::RawId) -> usize {
        let basis = self.field_basis_degree(field);
        if self.reference.family() == ReferenceCellFamily::Simplex || self.form.dimension == 1 {
            basis.saturating_sub(1)
        } else {
            basis
        }
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
