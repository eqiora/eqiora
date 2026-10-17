//! Immutable accepted primal content, reopened through the native point validator.

use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

use super::evidence::{Acceptance, PrimalEvidence};
use super::*;

const SCHEMA: &str = "eqiora.differentiable-evaluation/v1";

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
    state: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRecord<V = Vec<f64>, P = PrimalEvidence> {
    schema: String,
    program: String,
    point: V,
    state: V,
    output: V,
    primal: P,
}

impl DifferentiableEvaluation {
    /// Capture this accepted primal for explicit storage without a live solver.
    pub fn record(&self) -> Result<RecordedDifferentiableEvaluation, Diagnostic> {
        RecordedDifferentiableEvaluation::new(
            self.primal(),
            self.native
                .linearization()
                .relation()
                .accepted_unknowns()
                .to_vec(),
        )
    }
}

impl RecordedDifferentiableEvaluation {
    fn new(result: DifferentiableResult, state: Vec<f64>) -> Result<Self, Diagnostic> {
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
                primal: &evidence.primal,
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
        initial: Option<&CommonAlgebraicState>,
    ) -> Result<Self, Vec<Diagnostic>> {
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
        let identity = DifferentiableProgramIdentity::from_bytes(&identity_bytes, plan, initial)?;
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
        let linearization = match plan {
            ResolvedCommonPlan::Spatial(spatial) => spatial.validate_differentiated_values(
                &selected,
                Some(point.coordinates()),
                &wire.state,
            ),
            ResolvedCommonPlan::Algebraic(algebraic) => {
                let canonical;
                let initial = match initial {
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
                    &wire.state,
                )
            }
            _ => return Err(single(invalid("unsupported evaluation record Plan"))),
        };
        let linearization = linearization.map_err(single)?;
        let implementation = DerivativeImplementation::from_linearization(&linearization);
        if !same_coordinates(&wire.output, linearization.output_values())
            || !same_coordinates(
                point.coordinates(),
                linearization.relation().design_values(),
            )
        {
            return Err(single(invalid(
                "evaluation record differs from its accepted point or outputs",
            )));
        }
        wire.primal
            .validate(plan, initial, &linearization)
            .map_err(single)?;
        let acceptance = Acceptance::new(&identity, &linearization).map_err(single)?;
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
        initial: Option<&CommonAlgebraicState>,
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
            } => {
                let nonlinear = plan
                    .as_algebraic()
                    .and_then(|plan| plan.nonlinear())
                    .ok_or_else(|| invalid("nonlinear evaluation record requires a Newton Plan"))?;
                let initial = initial.ok_or_else(|| {
                    invalid("nonlinear evaluation record requires its initial State")
                })?;
                if initial.identity() != initial_state_identity
                    || *iterations > nonlinear.maximum_iterations().get()
                    || linearization.initial_residual_norm().map(f64::to_bits)
                        != Some(initial_residual_norm.to_bits())
                {
                    return Err(invalid(
                        "nonlinear evaluation record differs from its initial State or iteration policy",
                    ));
                }
                initial.validate_nonlinear_update_count(
                    linearization.relation().accepted_unknowns(),
                    *initial_residual_norm,
                    linearization.residual_target(),
                    *iterations,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests;
