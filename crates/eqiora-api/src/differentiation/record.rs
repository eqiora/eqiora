//! Immutable accepted primal content, reopened through the native point validator.

use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

use super::evidence::{Acceptance, PrimalEvidence};
use super::*;

const SCHEMA: &str = "eqiora.differentiable-evaluation/v3";

/// Read-only accepted primal with the candidate coordinates required for reopening.
///
/// Capturing a record explicitly retains these coordinates; ordinary JVP/VJP
/// results do not. The content identity is independent of execution occurrences.
/// Reopening checks mathematical acceptance, not the authenticity of a producer
/// or its historical iteration trace. It creates no live derivative executor.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedDifferentiableEvaluation {
    identity: String,
    result: DifferentiableResult,
    state: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRecord<V = Vec<f64>, S = Vec<Vec<f64>>, P = Vec<PrimalEvidence>> {
    schema: String,
    program: String,
    point: V,
    state: S,
    output: V,
    primal: P,
}

impl DifferentiableEvaluation {
    /// Capture this accepted primal for explicit storage without a live solver.
    pub fn record(&self) -> Result<RecordedDifferentiableEvaluation, Diagnostic> {
        RecordedDifferentiableEvaluation::new(
            self.primal(),
            self.native
                .points()
                .iter()
                .map(|point| {
                    point
                        .linearization()
                        .relation()
                        .accepted_unknowns()
                        .to_vec()
                })
                .collect(),
        )
    }
}

impl RecordedDifferentiableEvaluation {
    fn new(result: DifferentiableResult, state: Vec<Vec<f64>>) -> Result<Self, Diagnostic> {
        let mut record = Self {
            identity: String::new(),
            result,
            state,
        };
        record.identity =
            eqiora_artifact::ArtifactDigest::compute(SCHEMA.as_bytes(), &record.to_bytes()?)
                .as_str()
                .to_owned();
        Ok(record)
    }

    /// Canonical content identity; repeated capture does not mint a new occurrence.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Typed outputs, input point and owned historical provenance.
    #[must_use]
    pub const fn result(&self) -> &DifferentiableResult {
        &self.result
    }

    /// Encode bounded canonical content, including the exact static Program reference.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        let evidence = self.result.evidence();
        eqiora_artifact::JsonLimits::EXECUTION
            .encode(&WireRecord {
                schema: SCHEMA.to_owned(),
                program: STANDARD.encode(evidence.identity().to_bytes()?),
                point: evidence.point().coordinates(),
                state: self.state.as_slice(),
                output: self.result.output().coordinates(),
                primal: evidence
                    .steps()
                    .iter()
                    .map(|step| &step.primal)
                    .collect::<Vec<_>>(),
            })
            .map_err(|error| invalid(format!("cannot encode evaluation record: {error}")))
    }

    /// Reopen against the exact admitted Plan and optional initial State.
    ///
    /// Uses original-equation and regularity acceptance without solving, drawing
    /// samples, or executing derivative actions. Rejects foreign references,
    /// altered outputs, inconsistent provenance and noncanonical or oversized bytes.
    pub fn from_bytes(
        bytes: &[u8],
        plan: &ResolvedCommonPlan,
        initial: Option<&CommonInitialState<'_>>,
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::from_context_bytes(bytes, &Context::stationary(plan, initial))
    }

    /// Reopen a complete trajectory evaluation against its exact ordinary Run request, without solving.
    pub fn from_run_bytes(
        bytes: &[u8],
        request: &crate::RunRequest,
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::from_context_bytes(bytes, &Context::Run(std::borrow::Cow::Borrowed(request)))
    }

    pub(crate) fn from_context_bytes(
        bytes: &[u8],
        context: &Context<'_>,
    ) -> Result<Self, Vec<Diagnostic>> {
        let plan = context.plan();
        let initial = context.initial();
        let initial = initial.as_ref();
        eqiora_artifact::JsonLimits::EXECUTION
            .check(bytes)
            .map_err(single)?;
        let wire: WireRecord = serde_json::from_slice(bytes)
            .map_err(|error| single(invalid(format!("invalid evaluation record: {error}"))))?;
        if wire.schema != SCHEMA {
            return Err(single(invalid("unsupported evaluation record schema")));
        }
        let identity_bytes = STANDARD
            .decode(&wire.program)
            .map_err(|error| single(invalid(format!("invalid Program encoding: {error}"))))?;
        let identity = DifferentiableProgramIdentity::from_context_bytes(&identity_bytes, context)?;
        let point = DifferentiableValues::from_coordinates(identity.inputs.clone(), wire.point)
            .map_err(single)?;
        let selected = identity
            .inputs()
            .iter()
            .map(DifferentiableValueSignature::parameter)
            .collect::<Vec<_>>();
        let outputs = identity
            .outputs()
            .iter()
            .map(DifferentiableValueSignature::entity)
            .collect::<Vec<_>>();
        let linearizations = if let Some(request) = context.run_request() {
            plan.as_spatial()
                .ok_or_else(|| single(invalid("Run record requires its spatial Plan")))?
                .validate_differentiated_trajectory_values(
                    request.native(),
                    &selected,
                    Some(point.coordinates()),
                    &outputs,
                    identity.observation_quadratures(),
                    &wire.state,
                )
        } else {
            if wire.state.len() != 1 {
                return Err(single(invalid(
                    "stationary record requires exactly one accepted point",
                )));
            }
            match plan {
                ResolvedCommonPlan::Spatial(spatial) => spatial.validate_differentiated_values(
                    program::spatial_initial(initial).map_err(single)?,
                    &selected,
                    Some(point.coordinates()),
                    &outputs,
                    identity.observation_quadratures(),
                    &wire.state[0],
                ),
                ResolvedCommonPlan::Algebraic(algebraic) => {
                    let canonical;
                    let initial = match program::algebraic_initial(initial).map_err(single)? {
                        Some(initial) => initial,
                        None => {
                            canonical = algebraic.initial_state(&[]).map_err(single)?;
                            &canonical
                        }
                    };
                    algebraic.validate_differentiated_values(
                        initial,
                        &selected,
                        Some(point.coordinates()),
                        &outputs,
                        &wire.state[0],
                    )
                }
                _ => return Err(single(invalid("unsupported evaluation record Plan"))),
            }
            .map(|point| vec![point])
        };
        let linearizations = linearizations.map_err(single)?;
        let projection = OutputProjection::new(&identity, context.run_request()).map_err(single)?;
        let output_values = projection
            .gather(
                &linearizations
                    .iter()
                    .map(CommonDifferentiationLinearization::output_values)
                    .collect::<Vec<_>>(),
            )
            .map_err(single)?;
        let implementation = DerivativeImplementation::from_linearization(&linearizations[0]);
        if !same_coordinates(&wire.output, &output_values)
            || linearizations.iter().any(|linearization| {
                !same_coordinates(
                    point.coordinates(),
                    linearization.relation().design_values(),
                )
            })
        {
            return Err(single(invalid(
                "evaluation record differs from its accepted point or outputs",
            )));
        }
        if wire.primal.len() != linearizations.len() {
            return Err(single(invalid(
                "evaluation record requires every step's primal evidence",
            )));
        }
        let mut acceptance = Vec::with_capacity(linearizations.len());
        for (index, (primal, linearization)) in wire.primal.iter().zip(&linearizations).enumerate()
        {
            // Subsequent seeds are the preceding reaccepted physical history;
            // only the first step may name the external initial State.
            primal
                .validate(plan, if index == 0 { initial } else { None }, linearization)
                .map_err(single)?;
            acceptance.push(Acceptance::new(&identity, linearization).map_err(single)?);
        }
        let output = DifferentiableValues::from_coordinates(identity.outputs.clone(), wire.output)
            .map_err(single)?;
        let record = Self::new(
            DifferentiableResult {
                output,
                derivative: (),
                evidence: DifferentiationEvidence::from_record(
                    identity,
                    point,
                    implementation,
                    acceptance,
                    wire.primal,
                ),
            },
            wire.state,
        )
        .map_err(single)?;
        if record.to_bytes().map_err(single)? != bytes {
            return Err(single(invalid("evaluation record bytes are not canonical")));
        }
        Ok(record)
    }
}

fn same_coordinates(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

impl PrimalEvidence {
    fn validate(
        &self,
        plan: &ResolvedCommonPlan,
        initial: Option<&CommonInitialState<'_>>,
        linearization: &CommonDifferentiationLinearization,
    ) -> Result<(), Diagnostic> {
        match self {
            Self::Linear(solve) => {
                if linearization.initial_residual_norm().is_some()
                    || solve.orientation() != eqiora_solver::LinearOperatorOrientation::Normal
                    || solve.residual_target().to_bits()
                        != linearization.residual_target().to_bits()
                {
                    return Err(invalid(
                        "evaluation primal solve differs from native acceptance",
                    ));
                }
                solve.validate_plan(plan)
            }
            Self::Nonlinear {
                initial_state_identity,
                iterations,
                initial_residual_norm,
                linear_solves,
            } => {
                let Some(eqiora_numerics::CommonSolvePolicy::Newton { nonlinear, .. }) =
                    plan.canonical_solve_request()
                else {
                    return Err(invalid(
                        "nonlinear evaluation record requires a Newton Plan",
                    ));
                };
                if initial.map(CommonInitialState::identity) != initial_state_identity.as_deref()
                    || *iterations > nonlinear.maximum_iterations().get()
                    || linearization.initial_residual_norm().map(f64::to_bits)
                        != Some(initial_residual_norm.to_bits())
                {
                    return Err(invalid(
                        "nonlinear evaluation record differs from its initial State or iteration policy",
                    ));
                }
                linearization.validate_nonlinear_update_count(*iterations)?;
                if linear_solves.len() != *iterations {
                    return Err(invalid(
                        "nonlinear evaluation must retain each update's solve report",
                    ));
                }
                for solve in linear_solves {
                    if solve.orientation() != eqiora_solver::LinearOperatorOrientation::Normal {
                        return Err(invalid("nonlinear primal update must use a normal solve"));
                    }
                    solve.validate_plan(plan)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests;
