//! Original-pencil validation and selection of typed spectral candidates.
use super::*;
use eqiora_core::{Id, ValueLiteral, entity::kinds};
use eqiora_solver::HermitianEigenproblem;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct EigenResult {
    pub candidates: Vec<(ValueLiteral, ValueLiteral)>,
    pub mode_fields: Vec<Vec<(Id<kinds::Field>, ValueLiteral)>>,
    pub defects: Vec<(f64, f64)>,
    pub selected: Vec<usize>,
    pub converged: bool,
}

impl EigenResult {
    pub(super) fn accept(
        plan: &crate::CommonEigenPlan,
        candidates: Vec<(ValueLiteral, ValueLiteral)>,
    ) -> Result<Self, Diagnostic> {
        let pencil = plan.admitted_problem()?;
        if candidates.len() > pencil.dimension() {
            return Err(invalid(
                "spectral provider returned more modes than the space dimension",
            ));
        }
        let request = plan.request();
        let lifted = candidates
            .iter()
            .map(|(_, mode)| plan.lift_mode(mode))
            .collect::<Result<Vec<_>, _>>()?;
        let defects = candidates
            .iter()
            .zip(&lifted)
            .map(|((lambda, _), (fields, coordinate_residual))| {
                let (residual, normalization) = HermitianEigenproblem::original_eigenpair_defects(
                    plan.operator(),
                    plan.metric(),
                    lambda,
                    &fields[0].1,
                )?;
                Ok((residual.max(*coordinate_residual), normalization))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let mode_fields = lifted
            .into_iter()
            .map(|(fields, _)| fields)
            .collect::<Vec<_>>();
        let accepted = |index: usize| {
            defects[index].0 <= request.residual_tolerance()
                && defects[index].1 <= request.normalization_tolerance()
        };
        // All individually accepted candidates must be mutually B-orthonormal.
        // Duplicate vectors cannot masquerade as a complete spectrum.
        let modes = mode_fields
            .iter()
            .enumerate()
            .filter(|(i, _)| accepted(*i))
            .map(|(_, fields)| fields[0].1.clone())
            .collect::<Vec<_>>();
        if !modes.is_empty() {
            HermitianEigenproblem::original_metric_projector(
                plan.operator(),
                plan.metric(),
                &modes,
                request.normalization_tolerance(),
            )?;
        }
        let value = |i: usize| candidates[i].0.component(0).expect("validated scalar").0;
        let mut selected = (0..candidates.len())
            .filter(|&i| {
                request
                    .interval()
                    .is_none_or(|[lo, hi]| value(i) >= lo.value() && value(i) <= hi.value())
            })
            .collect::<Vec<_>>();
        selected.sort_by(|&a, &b| {
            let order = request
                .target()
                .map(|target| {
                    let left = (value(a) - target.value()).abs();
                    let right = (value(b) - target.value()).abs();
                    if left.is_infinite() && right.is_infinite() {
                        // Only rescale overflowing distances; preserve small separations.
                        let distance = |i| (value(i) * 0.5 - target.value() * 0.5).abs();
                        distance(a).total_cmp(&distance(b))
                    } else {
                        left.total_cmp(&right)
                    }
                })
                .unwrap_or(std::cmp::Ordering::Equal);
            order.then_with(|| value(a).total_cmp(&value(b)))
        });
        selected.truncate(request.count().get());
        // Do not replace a failed requested mode with a different successful mode.
        selected.retain(|&i| accepted(i));
        let converged = candidates.len() == pencil.dimension()
            && defects.iter().enumerate().all(|(i, _)| accepted(i))
            && selected.len() == request.count().get();
        Ok(Self {
            candidates,
            mode_fields,
            defects,
            selected,
            converged,
        })
    }
}

impl CommonResult {
    pub(crate) fn from_eigen(
        plan: &crate::CommonEigenPlan,
        candidates: Vec<(ValueLiteral, ValueLiteral)>,
    ) -> Result<Self, Diagnostic> {
        let spectral = EigenResult::accept(plan, candidates)?;
        Self {
            plan: ResolvedCommonPlan::Eigen(Box::new(plan.clone())),
            family: CommonResultFamily::Eigen,
            elapsed_seconds: 0.,
            identity: String::new(),
            payload: CommonResultPayload::Eigen(Box::new(spectral)),
        }
        .refresh_identity()
    }

    /// Number of accepted selected eigenpairs, including partially converged runs.
    pub fn eigenpair_count(&self) -> usize {
        match &self.payload {
            CommonResultPayload::Eigen(value) => value.selected.len(),
            _ => 0,
        }
    }

    /// Typed eigenvalue and original mode with the maximum dimensionless source
    /// residual (pencil and coordinate equalities) and the B-norm defect.
    pub fn eigenpair(&self, index: usize) -> Option<(&ValueLiteral, &ValueLiteral, f64, f64)> {
        let CommonResultPayload::Eigen(value) = &self.payload else {
            return None;
        };
        let i = *value.selected.get(index)?;
        let (lambda, _) = &value.candidates[i];
        let mode = &value.mode_fields[i][0].1;
        let (residual, normalization) = value.defects[i];
        Some((lambda, mode, residual, normalization))
    }

    /// Typed values for every source mode/coordinate Field of a selected
    /// eigenpair, ordered from the physical mode through its coordinate chain.
    /// Values are reconstructed and checked against the original equalities.
    pub fn eigenmode_fields(&self, index: usize) -> Option<&[(Id<kinds::Field>, ValueLiteral)]> {
        let CommonResultPayload::Eigen(value) = &self.payload else {
            return None;
        };
        Some(&value.mode_fields[*value.selected.get(index)?])
    }

    /// Metric projector for a selection of accepted modes. Phase and mode order
    /// do not change this operator; repeated eigenspaces may rotate their basis.
    pub fn eigenprojector(&self, indices: &[usize]) -> Result<ValueLiteral, Diagnostic> {
        let plan = self
            .plan
            .as_eigen()
            .ok_or_else(|| invalid("eigenprojector requires a spectral Result"))?;
        let modes = indices
            .iter()
            .map(|&i| {
                self.eigenpair(i)
                    .map(|(_, mode, _, _)| mode.clone())
                    .ok_or_else(|| invalid("eigenprojector index is outside the accepted modes"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        HermitianEigenproblem::original_metric_projector(
            plan.operator(),
            plan.metric(),
            &modes,
            plan.request().normalization_tolerance(),
        )
    }

    /// Dense spectral status: converged, partial, or not-converged.
    /// A short interval, missing candidates or failed tolerances cannot report convergence.
    pub fn eigen_convergence(&self) -> Option<&'static str> {
        let CommonResultPayload::Eigen(value) = &self.payload else {
            return None;
        };
        Some(if value.converged {
            "converged"
        } else if value.selected.is_empty() {
            "not-converged"
        } else {
            "partial"
        })
    }

    /// Number of provider candidates and number failing the requested tolerances.
    pub fn eigen_candidate_counts(&self) -> Option<(usize, usize)> {
        let CommonResultPayload::Eigen(value) = &self.payload else {
            return None;
        };
        let request = self.plan.as_eigen()?.request();
        Some((
            value.candidates.len(),
            value
                .defects
                .iter()
                .filter(|(r, n)| {
                    *r > request.residual_tolerance() || *n > request.normalization_tolerance()
                })
                .count(),
        ))
    }
}
