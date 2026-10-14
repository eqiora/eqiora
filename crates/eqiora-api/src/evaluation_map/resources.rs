use super::{EvaluationMapExecutionPolicy, EvaluationMapRetention, invalid, outcomes::Outcome};
use crate::{
    DifferentiableEvaluation, DifferentiableParameterPoint, DifferentiableProgram,
    SampledParameterPoint, SamplingCoupling,
};
use eqiora_core::Diagnostic;

pub(super) fn estimate(
    program: &DifferentiableProgram,
    count: usize,
    policy: EvaluationMapExecutionPolicy,
    samples: usize,
) -> Result<usize, Diagnostic> {
    let chunk = count.min(policy.chunk_size());
    let workers = chunk.min(policy.workers());
    let point = add(
        size_of::<DifferentiableParameterPoint>(),
        multiply(program.identity().input_dimension(), size_of::<f64>())?,
    )?;
    // Charge full indexed outcomes and complete-collection conversion buffers;
    // no accepted-prefix shortcut drops this O(n) membership charge.
    let records = add(
        size_of::<Outcome>(),
        add(
            program.map_evidence_bytes()?,
            size_of::<DifferentiableEvaluation>(),
        )?,
    )?;
    let inventory = multiply(count, add(point, records)?)?;
    let numerical_count = match policy.retention() {
        EvaluationMapRetention::Retain => add(count, workers)?,
        EvaluationMapRetention::Recompute => {
            if count == 0 {
                0
            } else {
                add(chunk, 1)?
            }
        }
    };
    let numerical = multiply(numerical_count, program.map_occurrence_bytes()?)?;
    // Canonical metadata size is a deterministic charge, not a heap estimate.
    // At most one temporary deployment copy per concurrently running member;
    // accepted receipts share the Program's exactly equal binding afterward.
    let binding = if workers == 0 {
        0
    } else {
        multiply(workers, program.map_binding_metadata_bytes()?)?
    };
    let queues = multiply(workers, add(size_of::<Outcome>(), 4 * size_of::<usize>())?)?;
    let total = [inventory, numerical, binding, queues, samples]
        .into_iter()
        .try_fold(0, add)?;
    if total > policy.storage_bytes_limit() || total > isize::MAX as usize {
        return Err(invalid(format!(
            "evaluation map storage estimate {total} exceeds limit {} or addressable storage",
            policy.storage_bytes_limit()
        )));
    }
    Ok(total)
}

pub(super) fn sample_bytes(samples: &[SampledParameterPoint]) -> Result<usize, Diagnostic> {
    samples.iter().try_fold(0usize, |total, sample| {
        let identity = sample.identity();
        let coupling = match identity.coupling() {
            SamplingCoupling::IndependentReplicate => 0,
            SamplingCoupling::CommonRandomNumbers(bytes) => bytes.len(),
        };
        let inputs = multiply(sample.point().values().len(), size_of::<f64>())?;
        [
            size_of::<SampledParameterPoint>(),
            identity.sample_id().len(),
            identity.channel().len(),
            multiply(identity.lineage().len(), size_of::<u64>())?,
            coupling,
            inputs,
            sample.program_identity().plan_identity().len(),
            64,
        ]
        .into_iter()
        .try_fold(total, add)
    })
}

pub(super) fn multiply(a: usize, b: usize) -> Result<usize, Diagnostic> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn add(a: usize, b: usize) -> Result<usize, Diagnostic> {
    a.checked_add(b).ok_or_else(overflow)
}
fn overflow() -> Diagnostic {
    invalid("evaluation map storage estimate overflows usize")
}
