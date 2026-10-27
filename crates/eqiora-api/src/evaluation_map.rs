//! Bounded ordered composition of immutable accepted evaluations.

use crate::{
    DifferentiableEvaluation, DifferentiableProgram, DifferentiableProgramIdentity,
    DifferentiableValues, SampledParameterPoint,
};
use eqiora_core::{Diagnostic, Id, diagnostic::codes, entity::kinds};
use std::sync::Arc;

type Parameter = Id<kinds::Parameter>;

mod axes;
mod collection;
pub use collection::{EvaluationCollection, EvaluationCollectionGroup, RecordedEvaluationOutcome};
mod occurrence_id;
pub use occurrence_id::EvaluationMapOccurrenceId;
mod outcomes;
mod parameters;
pub(crate) mod partition;
mod policy;
mod products;
mod resources;
mod schedule;
mod time;

pub use outcomes::{CompleteEvaluationMap, EvaluationMapOccurrence, EvaluationMapTerminalReport};
pub use policy::{EvaluationMapExecutionPolicy, EvaluationMapRetention};
pub use products::{EvaluationMapProducts, EvaluationMapResult};

/// An admitted ordered inventory of complete inputs for one immutable Program or an ordinary Run template.
///
/// Position identifies an occurrence within one execution. Equal inputs remain separate: no sorting,
/// deduplication, default anchor, retry or sampling occurs during execution.
#[derive(Debug, Clone)]
pub struct EvaluationMapPlan {
    inputs: Arc<Inputs>,
    policy: EvaluationMapExecutionPolicy,
    estimated_storage_bytes: usize,
}

#[derive(Debug, Clone)]
enum Inputs {
    Evaluations {
        program: Arc<DifferentiableProgram>,
        points: Arc<[DifferentiableValues]>,
        shared_inputs: Arc<[Parameter]>,
        samples: Option<Arc<[SampledParameterPoint]>>,
    },
    Time(time::TimeInputs),
    Parameters(parameters::ParameterInputs),
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
            inputs: Arc::new(Inputs::Evaluations {
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
            }),
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
            inputs: Arc::new(Inputs::Evaluations {
                points: samples
                    .iter()
                    .map(|sample| sample.point().clone())
                    .collect::<Vec<_>>()
                    .into(),
                program,
                samples: Some(samples.to_vec().into()),
                shared_inputs: shared_inputs.into(),
            }),
            policy,
            estimated_storage_bytes: estimate,
        })
    }

    pub(crate) fn program(&self) -> Result<&DifferentiableProgram, Diagnostic> {
        match self.inputs.as_ref() {
            Inputs::Evaluations { program, .. } => Ok(program),
            Inputs::Time(_) | Inputs::Parameters(_) => Err(invalid(
                "ordinary Run maps have no differentiable steady program",
            )),
        }
    }

    /// Exact original Model that supplies the requested occurrence inputs.
    #[must_use]
    pub fn model(&self) -> &eqiora_artifact::ModelArtifactReference {
        match self.inputs.as_ref() {
            Inputs::Evaluations { program, .. } => program.identity().model(),
            Inputs::Time(inputs) => &inputs.model,
            Inputs::Parameters(inputs) => &inputs.model,
        }
    }

    /// Complete occurrence count, including repeated and unstarted requests.
    #[must_use]
    pub fn len(&self) -> usize {
        match self.inputs.as_ref() {
            Inputs::Evaluations { points, .. } => points.len(),
            Inputs::Time(inputs) => inputs.requests.len(),
            Inputs::Parameters(inputs) => inputs.points.len(),
        }
    }

    /// Whether this admitted inventory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Exact steady program identity; time maps reject this derivative projection.
    pub fn program_identity(&self) -> Result<&DifferentiableProgramIdentity, Diagnostic> {
        self.program().map(DifferentiableProgram::identity)
    }

    /// Explicit shared steady inputs; equal values never imply sharing.
    pub fn shared_inputs(&self) -> Result<&[Parameter], Diagnostic> {
        match self.inputs.as_ref() {
            Inputs::Evaluations { shared_inputs, .. } => Ok(shared_inputs),
            Inputs::Time(_) | Inputs::Parameters(_) => Err(invalid(
                "ordinary Run maps have no shared derivative inputs",
            )),
        }
    }

    /// Complete frozen steady points; time maps retain their original Run requests instead.
    pub fn points(&self) -> Result<&[DifferentiableValues], Diagnostic> {
        match self.inputs.as_ref() {
            Inputs::Evaluations { points, .. } => Ok(points),
            Inputs::Time(_) | Inputs::Parameters(_) => Err(invalid(
                "ordinary Run maps do not contain differentiable points",
            )),
        }
    }

    /// Full frozen parameter-sampling association, when constructed from samples.
    /// Time sampling identities remain on the original time requests.
    #[must_use]
    pub fn samples(&self) -> Option<&[SampledParameterPoint]> {
        match self.inputs.as_ref() {
            Inputs::Evaluations { samples, .. } => samples.as_deref(),
            Inputs::Time(_) | Inputs::Parameters(_) => None,
        }
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
    /// The occurrence ID binds delivery to this exact execution and request position.
    /// A fresh execution mints a fresh identity, including a retry of equal inputs.
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
        D: FnMut(EvaluationMapOccurrenceId, EvaluationMapOccurrence<'_>),
    {
        if let Inputs::Time(inputs) = self.inputs.as_ref() {
            return time::execute(self, inputs, &mut should_cancel, &mut deliver);
        }
        if let Inputs::Parameters(inputs) = self.inputs.as_ref() {
            return time::execute_common(
                self,
                &|index| inputs.evaluate(index),
                &mut should_cancel,
                &mut deliver,
            );
        }
        schedule::execute(
            self,
            &|program, point| program.evaluate(point),
            &mut should_cancel,
            &mut |index, member| deliver(index, EvaluationMapOccurrence::Accepted(member)),
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
        // Scheduling and retention do not change the complete requested inventory.
        match (self.inputs.as_ref(), other.inputs.as_ref()) {
            (
                Inputs::Evaluations {
                    program: a,
                    points: ap,
                    shared_inputs: ai,
                    samples: sa,
                },
                Inputs::Evaluations {
                    program: b,
                    points: bp,
                    shared_inputs: bi,
                    samples: sb,
                },
            ) => {
                a.identity() == b.identity()
                    && ai == bi
                    && sa == sb
                    && ap.len() == bp.len()
                    && ap.iter().zip(bp.iter()).all(|(a, b)| {
                        a.signatures() == b.signatures()
                            && exact_values(a.coordinates(), b.coordinates())
                    })
            }
            (Inputs::Time(a), Inputs::Time(b)) => a.same_inputs(b),
            (Inputs::Parameters(a), Inputs::Parameters(b)) => a.same_inputs(b),
            _ => false,
        }
    }
}

fn validate_member(
    plan: &EvaluationMapPlan,
    point: &DifferentiableValues,
    member: &DifferentiableEvaluation,
) -> Result<(), Diagnostic> {
    if member.identity() != plan.program_identity()? {
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
pub(crate) fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}

#[cfg(test)]
mod bounded_tests;
#[cfg(test)]
mod tests;
