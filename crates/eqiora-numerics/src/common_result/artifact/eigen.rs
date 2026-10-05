use super::*;
use crate::common_result::eigen::EigenResult;
use eqiora_core::ValueLiteral;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireEigenResult {
    candidates: Vec<WireEigenpair>,
    selected: Vec<usize>,
    converged: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireEigenpair {
    eigenvalue: f64,
    coordinates: Vec<(f64, f64)>,
    residual: f64,
    normalization_defect: f64,
}

impl WireEigenResult {
    pub(super) fn from_native(value: &EigenResult) -> Self {
        Self {
            candidates: value
                .candidates
                .iter()
                .zip(&value.defects)
                .map(|((lambda, mode), &(residual, normalization_defect))| {
                    let count = mode
                        .value_type()
                        .shape()
                        .component_count()
                        .expect("validated finite shape");
                    WireEigenpair {
                        eigenvalue: lambda.component(0).expect("validated scalar").0,
                        coordinates: (0..count)
                            .map(|i| mode.component(i).expect("validated shape"))
                            .collect(),
                        residual,
                        normalization_defect,
                    }
                })
                .collect(),
            selected: value.selected.clone(),
            converged: value.converged,
        }
    }

    pub(super) fn replay(&self, plan: &ResolvedCommonPlan) -> Result<EigenResult, Diagnostic> {
        let plan = plan
            .as_eigen()
            .ok_or_else(|| invalid("spectral Result requires its exact spectral Plan"))?;
        let problem = plan.admitted_problem()?;
        let candidates = self
            .candidates
            .iter()
            .map(|pair| {
                let lambda = ValueLiteral::new(
                    problem.eigenvalue_type().clone(),
                    vec![(pair.eigenvalue, 0.)],
                )
                .map_err(|error| invalid(error.to_string()))?;
                let mode = ValueLiteral::new(problem.mode_type().clone(), pair.coordinates.clone())
                    .map_err(|error| invalid(error.to_string()))?;
                Ok((lambda, mode))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let result = EigenResult::accept(plan, candidates)?;
        if Self::from_native(&result) != *self {
            return Err(invalid(
                "spectral Result selection, convergence or original-pencil evidence is inconsistent",
            ));
        }
        Ok(result)
    }
}
