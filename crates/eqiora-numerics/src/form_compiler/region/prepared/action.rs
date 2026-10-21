//! Constitutive State/rate actions and their explicit time-coordinate pullbacks.
use super::super::field_point::{FieldPullback, FieldValues};
use super::*;

impl<S: Coefficient, P: Projection<S>> PreparedRegionCell<S, P> {
    pub(super) fn step_pullback(&self) -> FieldPullback<'static> {
        FieldPullback::Uniform {
            current: Some(1.),
            rate: self.step.map(f64::recip),
        }
    }

    /// Current columns are normalized; rate columns act on physical coefficients.
    pub(super) fn affine_matrix(&self, pullback: FieldPullback<'_>) -> Vec<P::Scalar> {
        let zero = <P::Scalar as From<f64>>::from(0.);
        let mut matrix = self
            .affine
            .matrix()
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                pullback
                    .weight(index % self.affine.columns(), false)
                    .map_or(zero, |weight| *entry * weight)
            })
            .collect::<Vec<_>>();
        if let Some(rate) = &self.rate {
            let count = self.affine.columns();
            for (row, rate_row) in matrix
                .chunks_exact_mut(count)
                .zip(rate.matrix().chunks_exact(count))
            {
                for (field, range) in self.fields.iter().zip(&self.coordinates) {
                    for column in range.clone() {
                        if let Some(weight) = pullback.weight(column, true) {
                            row[column] += rate_row[column] * (field.scale * weight);
                        }
                    }
                }
            }
        }
        matrix
    }
}

impl<S: Coefficient, P: Projection<S, Scalar = f64>> PreparedRegionCell<S, P> {
    pub(crate) fn linearize(
        &self,
        previous: &BTreeMap<RawId, Vec<f64>>,
        point: &[f64],
    ) -> Result<RegionLinearization, Diagnostic> {
        self.action(previous, point, true)
    }

    pub(crate) fn residual(
        &self,
        previous: &BTreeMap<RawId, Vec<f64>>,
        point: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        Ok(self.action(previous, point, false)?.residual)
    }

    fn action(
        &self,
        previous: &BTreeMap<RawId, Vec<f64>>,
        point: &[f64],
        derivative: bool,
    ) -> Result<RegionLinearization, Diagnostic> {
        let count = self.affine.rows();
        if point.len() != count || point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "nonlinear region requires a finite exact candidate point",
            ));
        }
        self.validate_previous(previous)?;
        let rates = self
            .step
            .map(|step| {
                let mut rates = self.physical_coefficients(point)?;
                for (field, range) in self.fields.iter().zip(&self.coordinates) {
                    if let Some(previous) = previous.get(&field.field) {
                        for (rate, old) in rates[range.clone()].iter_mut().zip(previous) {
                            // Subtract physical values first: unscaling history can invent a rate.
                            *rate = (*rate - old) / step;
                        }
                    } else {
                        rates[range.clone()].fill(0.);
                    }
                }
                Ok::<_, Diagnostic>(rates)
            })
            .transpose()?;
        let pullback = derivative.then(|| self.step_pullback());
        if self.independent_coordinates {
            return self.implicit_action(point, rates.as_deref(), pullback);
        }
        // Kinematic elimination and geometric inventory already carry explicit time
        // transformations. Their original history terms cannot be interpreted as R(u,v).
        let matrix = self.affine_matrix(self.step_pullback());
        let rhs = self.rhs(previous)?;
        let residual = matrix
            .chunks_exact(count)
            .zip(&rhs)
            .map(|(row, rhs)| row.iter().zip(point).map(|(a, b)| a * b).sum::<f64>() - rhs)
            .collect();
        let mut action = RegionLinearization {
            residual,
            jacobian: if derivative { matrix } else { Vec::new() },
        };
        self.add_physical_nonlinear(point, rates.as_deref(), pullback, &mut action)?;
        Ok(action)
    }

    /// Convert normalized scalar words to physical local coefficients, for values or rates.
    pub(crate) fn physical_coefficients(&self, point: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        if point.len() != self.affine.columns() {
            return Err(invalid(
                "physical coefficients require exact local coordinate coverage",
            ));
        }
        let mut physical = point.to_vec();
        for (field, range) in self.fields.iter().zip(&self.coordinates) {
            for value in &mut physical[range.clone()] {
                *value *= field.scale;
            }
        }
        if physical.iter().any(|value| !value.is_finite()) {
            return Err(invalid("physical local coefficients are non-finite"));
        }
        Ok(physical)
    }

    /// Original fixed-geometry residual and independent State/rate derivatives.
    /// Rates are physical coefficients; the State point uses normalized coordinates.
    pub(crate) fn implicit_action(
        &self,
        point: &[f64],
        rates: Option<&[f64]>,
        pullback: Option<FieldPullback<'_>>,
    ) -> Result<RegionLinearization, Diagnostic> {
        if !self.independent_coordinates {
            return Err(invalid(
                "independent State/rate actions require untransformed coordinates",
            ));
        }
        let count = self.affine.rows();
        if point.len() != count
            || point.iter().any(|v| !v.is_finite())
            || rates.is_some_and(|v| v.len() != count || v.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid(
                "implicit region action requires exact finite State and rate coordinates",
            ));
        }
        if let Some(pullback) = pullback {
            pullback.validate(count)?;
        }
        let mut residual = self
            .affine
            .matrix()
            .chunks_exact(count)
            .zip(self.affine.rhs())
            .map(|(row, rhs)| row.iter().zip(point).map(|(a, b)| a * b).sum::<f64>() - rhs)
            .collect::<Vec<_>>();
        if let Some(rate) = &self.rate {
            let rates =
                rates.ok_or_else(|| invalid("implicit region action lacks physical rates"))?;
            for (value, row) in residual.iter_mut().zip(rate.matrix().chunks_exact(count)) {
                *value += row.iter().zip(rates).map(|(a, b)| a * b).sum::<f64>();
            }
        }
        let mut action = RegionLinearization {
            residual,
            jacobian: pullback.map_or_else(Vec::new, |p| self.affine_matrix(p)),
        };
        self.add_physical_nonlinear(point, rates, pullback, &mut action)?;
        Ok(action)
    }

    fn add_physical_nonlinear(
        &self,
        point: &[f64],
        rates: Option<&[f64]>,
        pullback: Option<FieldPullback<'_>>,
        action: &mut RegionLinearization,
    ) -> Result<(), Diagnostic> {
        let physical = P::physical_point(&self.fields, &self.coordinates, point);
        let physical_rates =
            rates.map(|values| P::physical_point(&self.fields, &self.coordinates, values));
        self.add_nonlinear_action(
            &FieldValues {
                current: &physical,
                rates: physical_rates.as_deref(),
            },
            pullback,
            action,
        )
    }

    /// Nonlinear constitutive actions before a time discretization relates State and rate.
    pub(in crate::form_compiler::region) fn add_nonlinear_action(
        &self,
        point: &FieldValues<'_, S>,
        pullback: Option<FieldPullback<'_>>,
        action: &mut RegionLinearization,
    ) -> Result<(), Diagnostic> {
        let count = self.affine.rows();
        let physical_count: usize = self.fields.iter().map(|field| field.range.len()).sum();
        if action.residual.len() != count
            || action.jacobian.len() != if pullback.is_some() { count * count } else { 0 }
            || point.current.len() != physical_count
            || point.current.iter().any(|value| !value.is_finite())
            || point.rates.is_some_and(|rates| {
                rates.len() != physical_count || rates.iter().any(|value| !value.is_finite())
            })
        {
            return Err(invalid(
                "nonlinear action requires exact finite State and rate coordinates",
            ));
        }
        if let Some(pullback) = pullback {
            pullback.validate(count)?;
        }
        for sample in &self.samples {
            for term in &self.components {
                self.add_component_action(term, sample, point, pullback, action)?;
            }
            self.add_dyadic_actions(sample, point.current, pullback, action)?;
        }
        if action
            .residual
            .iter()
            .chain(&action.jacobian)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("nonlinear region action is non-finite"));
        }
        Ok(())
    }
}
