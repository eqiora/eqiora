//! Bounded ordered composition of immutable accepted evaluations.

use crate::{
    DifferentiableEvaluation, DifferentiableProgram, DifferentiableProgramIdentity,
    DifferentiableValues, SampledParameterPoint,
};
use eqiora_core::{Diagnostic, Id, diagnostic::codes, entity::kinds};
use std::sync::Arc;

type Parameter = Id<kinds::Parameter>;

mod axes;
mod estimator;
pub use estimator::{EvaluationMapEstimator, MomentEstimatorConvention};
mod outcomes;
mod partition;
mod policy;
mod products;
mod resources;
mod schedule;

pub use outcomes::{CompleteEvaluationMap, EvaluationMapOccurrence, EvaluationMapTerminalReport};
pub use policy::{EvaluationMapExecutionPolicy, EvaluationMapRetention};
pub use products::{EvaluationMapJvp, EvaluationMapProducts, EvaluationMapVjp};

/// An admitted ordered inventory of complete inputs for one immutable program.
///
/// Position identifies an occurrence. Equal points remain separate: no sorting,
/// deduplication, default anchor, retry or sampling occurs during execution.
#[derive(Debug, Clone)]
pub struct EvaluationMapPlan {
    program: Arc<DifferentiableProgram>,
    points: Arc<[DifferentiableValues]>,
    shared_inputs: Arc<[Parameter]>,
    samples: Option<Arc<[SampledParameterPoint]>>,
    policy: EvaluationMapExecutionPolicy,
    estimated_storage_bytes: usize,
}

impl EvaluationMapPlan {
    /// Admit complete points, an explicit shared-input selection and execution policy.
    ///
    /// Shared inputs retain their selected order and must be bit-identical across
    /// all points. An empty selection makes every input occurrence-local.
    ///
    /// The storage estimate includes every planned point and indexed terminal
    /// evidence, the policy's resident numerical buffers and bounded ordering
    /// buffers, plus the explicit shared-input ID inventory. Deployment metadata
    /// uses its encoded-size charge, not a heap upper bound. Shared Program storage,
    /// allocator overhead, solver scratch,
    /// OS thread stacks, diagnostic strings and callback-owned allocations are
    /// excluded: this is neither a process-memory nor peak-memory limit.
    ///
    /// # Errors
    /// Rejects foreign Models, roles, types or input order, unsupported providers and checked
    /// count/storage overflow before any member executes or points are copied.
    pub fn new(
        program: Arc<DifferentiableProgram>,
        points: &[DifferentiableValues],
        shared_inputs: &[Parameter],
        policy: EvaluationMapExecutionPolicy,
    ) -> Result<Self, Diagnostic> {
        for point in points {
            program.validate_point(point)?;
        }
        partition::admit_shared(program.identity(), points.iter(), shared_inputs)?;
        program.validate_map_execution()?;
        let estimate = resources::estimate(&program, points.len(), policy, 0, shared_inputs.len())?;
        Ok(Self {
            points: points
                .iter()
                .map(|values| {
                    program
                        .point_from_coordinates(values.coordinates())
                        .expect("validated native point")
                })
                .collect::<Vec<_>>()
                .into(),
            program,
            samples: None,
            shared_inputs: shared_inputs.into(),
            policy,
            estimated_storage_bytes: estimate,
        })
    }

    /// Admit already frozen sampled points with their full sampling identity.
    ///
    /// No sampler is retained or called. Repeated IDs and equal points keep their
    /// separate request positions; chunking and recomputation cannot change the
    /// generator, master stream, coupling, lineage or frozen input.
    ///
    /// # Errors
    /// Rejects a foreign Program or point/input order and the same structural,
    /// provider and storage limits as [Self::new], including sample metadata.
    pub fn from_samples(
        program: Arc<DifferentiableProgram>,
        samples: &[SampledParameterPoint],
        shared_inputs: &[Parameter],
        policy: EvaluationMapExecutionPolicy,
    ) -> Result<Self, Diagnostic> {
        for sample in samples {
            if sample.program_identity() != program.identity() {
                return Err(invalid(
                    "evaluation map sample belongs to a foreign Program or input order",
                ));
            }
            program.validate_point(sample.point())?;
        }
        partition::admit_shared(
            program.identity(),
            samples.iter().map(SampledParameterPoint::point),
            shared_inputs,
        )?;
        program.validate_map_execution()?;
        let sample_bytes = resources::sample_bytes(samples)?;
        let estimate = resources::estimate(
            &program,
            samples.len(),
            policy,
            sample_bytes,
            shared_inputs.len(),
        )?;
        Ok(Self {
            points: samples
                .iter()
                .map(|sample| sample.point().clone())
                .collect::<Vec<_>>()
                .into(),
            program,
            samples: Some(samples.to_vec().into()),
            shared_inputs: shared_inputs.into(),
            policy,
            estimated_storage_bytes: estimate,
        })
    }

    /// Exact Model, Plan, ordered inputs and output authority.
    #[must_use]
    pub fn program_identity(&self) -> &DifferentiableProgramIdentity {
        self.program.identity()
    }
    /// Explicit globally shared inputs, in their declared order.
    ///
    /// Equal values never imply sharing, and derivative views cannot redefine it.
    #[must_use]
    pub fn shared_inputs(&self) -> &[Parameter] {
        &self.shared_inputs
    }
    /// Frozen complete points in request order, including duplicates.
    #[must_use]
    pub fn points(&self) -> &[DifferentiableValues] {
        &self.points
    }
    /// Full frozen sampling association, when constructed from samples.
    #[must_use]
    pub fn samples(&self) -> Option<&[SampledParameterPoint]> {
        self.samples.as_deref()
    }
    /// Explicit scheduling and accepted-state lifetime.
    #[must_use]
    pub const fn policy(&self) -> EvaluationMapExecutionPolicy {
        self.policy
    }
    /// Admitted defined storage charge, not actual heap or peak process memory.
    #[must_use]
    pub const fn estimated_storage_bytes(&self) -> usize {
        self.estimated_storage_bytes
    }

    /// Execute with this plan's bounded host scheduling and retention policy.
    ///
    /// # Errors
    /// Returns indexed partial outcomes, never a fake complete collection.
    pub fn execute(&self) -> Result<CompleteEvaluationMap, EvaluationMapTerminalReport> {
        self.execute_with_cancellation(|| false)
    }

    /// Execute with cooperative cancellation before dispatching occurrences.
    ///
    /// In-flight members finish ordinary acceptance; no cancellation poll occurs
    /// within a member or after all occurrences are accepted. Empty maps are
    /// already complete. A failed member stays failed even if cancellation also
    /// arrives; independently accepted in-flight members remain inspectable.
    ///
    /// # Errors
    /// Returns indexed failure/cancelled/not-started states on incomplete work.
    pub fn execute_with_cancellation<C: FnMut() -> bool>(
        &self,
        should_cancel: C,
    ) -> Result<CompleteEvaluationMap, EvaluationMapTerminalReport> {
        self.execute_with_delivery(should_cancel, |_, _| {})
    }

    /// Deliver accepted members in request order, at bounded chunk boundaries.
    ///
    /// The callback only borrows each evaluation. Under Recompute, its primal
    /// and derivative buffers are released immediately after delivery. Retained
    /// callback copies are caller-owned, outside this plan's storage charge.
    /// Accepted members of an incomplete chunk may also be delivered; only the
    /// final return type establishes complete membership. No reduction, failure
    /// filtering or normalization is implicit in delivery.
    ///
    /// # Errors
    /// Preserves every member failure and cancellation in an indexed report.
    pub fn execute_with_delivery<C, D>(
        &self,
        mut should_cancel: C,
        mut deliver: D,
    ) -> Result<CompleteEvaluationMap, EvaluationMapTerminalReport>
    where
        C: FnMut() -> bool,
        D: FnMut(usize, &DifferentiableEvaluation),
    {
        schedule::execute(
            self,
            &|program, point| program.evaluate(point),
            &mut should_cancel,
            &mut deliver,
            &mut |_| {},
        )
    }

    #[cfg(test)]
    fn execute_with_evaluator<E, C>(
        &self,
        evaluator: &E,
        should_cancel: &mut C,
    ) -> Result<CompleteEvaluationMap, EvaluationMapTerminalReport>
    where
        E: Fn(
                &DifferentiableProgram,
                &DifferentiableValues,
            ) -> Result<DifferentiableEvaluation, Vec<Diagnostic>>
            + Sync,
        C: FnMut() -> bool,
    {
        schedule::execute(self, evaluator, should_cancel, &mut |_, _| {}, &mut |_| {})
    }
}

impl PartialEq for EvaluationMapPlan {
    fn eq(&self, other: &Self) -> bool {
        // Scheduling and retention never change mathematical identity.
        self.program_identity() == other.program_identity()
            && self.shared_inputs == other.shared_inputs
            && self.samples == other.samples
            && self.points.len() == other.points.len()
            && self.points.iter().zip(other.points.iter()).all(|(a, b)| {
                a.signatures() == b.signatures() && exact_values(a.coordinates(), b.coordinates())
            })
    }
}

fn validate_member(
    plan: &EvaluationMapPlan,
    point: &DifferentiableValues,
    member: &DifferentiableEvaluation,
) -> Result<(), Diagnostic> {
    if member.identity() != plan.program_identity() {
        return Err(invalid(
            "evaluation map member belongs to a foreign program or Plan",
        ));
    }
    if member.point().signatures() != point.signatures()
        || !exact_values(member.point().coordinates(), point.coordinates())
    {
        return Err(invalid(
            "evaluation map member differs from its exact requested point",
        ));
    }
    Ok(())
}
fn exact_values(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}

#[cfg(test)]
mod bounded_tests;
#[cfg(test)]
mod tests;
