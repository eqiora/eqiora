use super::*;
use crate::ParameterSampler;
use eqiora_numerics::SamplingIdentity;
use eqiora_solver::{
    LinearProblem, LinearSolution, LinearSolverBackend, REFERENCE_LINEAR_SOLVER,
    ReplicatedLinearExecution, SolverCapabilities, SolverPlan, SolverProvider,
};
use eqiora_time::ImplicitMidpointTimeBackend;
use std::num::NonZeroUsize;

#[derive(Debug)]
struct NoSolve;
impl LinearSolverBackend for NoSolve {
    fn provider(&self) -> SolverProvider {
        REFERENCE_LINEAR_SOLVER.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        REFERENCE_LINEAR_SOLVER.capabilities()
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        panic!("collection reopen must not solve")
    }
}

fn fixture(retention: EvaluationMapRetention) -> EvaluationMapPlan {
    let (_, program) = super::super::tests::fixture();
    let program = Arc::new(program);
    let fixed = DifferentiableValues::from_components(&[], &[]).unwrap();
    let sampler = ParameterSampler::new(
        program.clone(),
        [7; 32],
        &fixed,
        &[[1., 2.], [0.5, 1.], [-0.5, 0.5]],
    )
    .unwrap();
    let samples = [2_u8, 1, 2, 3]
        .iter()
        .map(|id| {
            sampler
                .sample(&SamplingIdentity::new(&[*id], b"collection", &[11], None).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    EvaluationMapPlan::from_samples(
        program,
        &samples,
        &[],
        EvaluationMapExecutionPolicy::new(
            1,
            eqiora_realization::Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            retention,
            1 << 28,
        )
        .unwrap(),
    )
    .unwrap()
}

fn reopen(bytes: &[u8]) -> Result<EvaluationCollection, Vec<Diagnostic>> {
    crate::sampling::without_draws(|| {
        EvaluationCollection::from_bytes(
            bytes,
            &NoSolve,
            &[ImplicitMidpointTimeBackend::CAPABILITIES],
        )
    })
}

#[test]
fn collection_roundtrip_preserves_duplicate_samples_and_separates_retry_occurrences() {
    let plan = fixture(EvaluationMapRetention::Retain);
    let mut delivered = Vec::new();
    let complete = plan
        .execute_with_delivery(|| false, |id, _| delivered.push(id))
        .unwrap();
    assert_eq!(
        delivered,
        (0..4)
            .map(|i| complete.occurrence_id(i).unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(complete.clone().execution_id(), complete.execution_id());
    let saved = complete.record().unwrap();
    assert_eq!(saved, complete.record().unwrap());
    let bytes = saved.to_bytes().unwrap();
    let reopened = reopen(&bytes).unwrap();
    assert_eq!(saved, reopened);
    assert!(reopened.is_complete());
    let group = &reopened.groups()[0];
    assert_eq!(group.samples().unwrap()[0], group.samples().unwrap()[2]);
    assert_ne!(
        group.occurrence(0).unwrap().0,
        group.occurrence(2).unwrap().0
    );
    let retry = plan.execute_recorded(|| false).unwrap();
    assert_eq!(
        group.request_identity().unwrap(),
        retry.groups()[0].request_identity().unwrap()
    );
    assert_ne!(group.execution_id(), retry.groups()[0].execution_id());
    let combined = EvaluationCollection::combine(vec![saved.clone(), retry]).unwrap();
    assert_eq!(combined, reopen(&combined.to_bytes().unwrap()).unwrap());
    assert!(EvaluationCollection::combine(vec![saved.clone(), saved]).is_err());
}

#[test]
fn collection_records_released_and_interrupted_members_without_implicit_recomputation() {
    for retention in [
        EvaluationMapRetention::Retain,
        EvaluationMapRetention::Recompute,
    ] {
        let plan = fixture(retention);
        let mut checks = 0;
        let partial = plan
            .execute_recorded(|| {
                checks += 1;
                checks == 3
            })
            .unwrap();
        assert!(!partial.is_complete());
        assert_eq!(partial.groups()[0].len(), 4);
        assert!(matches!(
            partial.groups()[0].occurrence(0).unwrap().1,
            RecordedEvaluationOutcome::Accepted(_)
        ));
        assert!(matches!(
            partial.groups()[0].occurrence(2).unwrap().1,
            RecordedEvaluationOutcome::Cancelled(None)
        ));
        assert!(matches!(
            partial.groups()[0].occurrence(3).unwrap().1,
            RecordedEvaluationOutcome::NotStarted
        ));
        assert_eq!(partial, reopen(&partial.to_bytes().unwrap()).unwrap());
        let complete = plan.execute_recorded(|| false).unwrap();
        assert!(complete.is_complete());
        assert_eq!(complete, reopen(&complete.to_bytes().unwrap()).unwrap());
        if retention == EvaluationMapRetention::Recompute {
            assert!(
                plan.execute()
                    .unwrap()
                    .record()
                    .unwrap_err()
                    .message()
                    .contains("originally delivered")
            );
        }
    }
}

#[test]
fn collection_rejects_reordered_duplicate_missing_foreign_and_false_complete_membership() {
    let plan = fixture(EvaluationMapRetention::Retain);
    let collection = plan.execute_recorded(|| false).unwrap();
    let bytes = collection.to_bytes().unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut swapped = original.clone();
    swapped["groups"][0]["members"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    let mut duplicated = original.clone();
    duplicated["groups"][0]["members"][1]["occurrence"] =
        duplicated["groups"][0]["members"][0]["occurrence"].clone();
    let mut missing = original.clone();
    missing["groups"][0]["records"] = serde_json::json!({});
    let mut wrong_reference = original.clone();
    wrong_reference["groups"][0]["members"][0]["outcome"]["reference"] =
        original["groups"][0]["members"][1]["outcome"]["reference"].clone();
    let mut foreign_plan = original.clone();
    foreign_plan["groups"][0]["members"][0]["outcome"]["plan"] = serde_json::json!("foreign");
    let mut truncated_inventory = original.clone();
    truncated_inventory["groups"][0]["members"]
        .as_array_mut()
        .unwrap()
        .pop();
    for (bad, expected) in [
        (swapped, "attribution"),
        (duplicated, "attribution"),
        (missing, "reference is missing"),
        (wrong_reference, "requested Program or point"),
        (truncated_inventory, "missing planned occurrences"),
        (foreign_plan, "cannot carry an ordinary Run Plan"),
    ] {
        let errors = reopen(&serde_json::to_vec(&bad).unwrap()).unwrap_err();
        assert!(errors[0].message().contains(expected), "{errors:?}");
    }
    let partial = plan.execute_recorded(|| true).unwrap();
    let mut bad: serde_json::Value = serde_json::from_slice(&partial.to_bytes().unwrap()).unwrap();
    bad["complete"] = serde_json::json!(true);
    assert!(
        reopen(&serde_json::to_vec(&bad).unwrap()).unwrap_err()[0]
            .message()
            .contains("completion label")
    );
    for bad in [
        bytes[..bytes.len() - 1].to_vec(),
        [bytes.as_slice(), b" "].concat(),
        vec![b'['; 65],
    ] {
        assert!(reopen(&bad).is_err());
    }
}

#[test]
fn occurrence_identity_is_canonical_and_bound_even_for_empty_or_cancelled_execution() {
    let plan = fixture(EvaluationMapRetention::Retain);
    let report = plan.execute_with_cancellation(|| true).unwrap_err();
    for i in 0..4 {
        let id = report.occurrence_id(i).unwrap();
        assert_eq!(
            id.to_string().parse::<EvaluationMapOccurrenceId>().unwrap(),
            id
        );
        assert_eq!(id.execution_id(), report.execution_id());
        assert_eq!(id.index(), i);
    }
    assert!(report.occurrence_id(4).is_none());
    assert!(
        format!("{}:00", report.execution_id())
            .parse::<EvaluationMapOccurrenceId>()
            .is_err()
    );
    let program = Arc::new(plan.program().unwrap().clone());
    let empty = EvaluationMapPlan::new(
        program,
        &[],
        &[],
        EvaluationMapExecutionPolicy::retained(16),
    )
    .unwrap();
    assert_eq!(empty.estimated_storage_bytes(), 16);
    let a = empty.execute_recorded(|| true).unwrap();
    let b = empty.execute_recorded(|| true).unwrap();
    assert!(a.is_complete());
    assert_ne!(a.groups()[0].execution_id(), b.groups()[0].execution_id());
    assert_eq!(a, reopen(&a.to_bytes().unwrap()).unwrap());
}

#[test]
fn repeated_content_references_share_payload_without_merging_occurrences() {
    let mut collection = fixture(EvaluationMapRetention::Retain)
        .execute_recorded(|| false)
        .unwrap();
    // Two independent occurrences may explicitly refer to the same immutable
    // accepted content. Wall-clock metadata need not coincide across fresh solves.
    let group = &mut collection.groups[0];
    group.outcomes[2] = group.outcomes[0].clone();
    let collection = EvaluationCollection::new(collection.groups).unwrap();
    let reopened = reopen(&collection.to_bytes().unwrap()).unwrap();
    let group = &reopened.groups()[0];
    let (first_id, RecordedEvaluationOutcome::Accepted(first)) = group.occurrence(0).unwrap()
    else {
        panic!("accepted")
    };
    let (repeated_id, RecordedEvaluationOutcome::Accepted(repeated)) = group.occurrence(2).unwrap()
    else {
        panic!("accepted")
    };
    assert_ne!(first_id, repeated_id);
    assert!(Arc::ptr_eq(first, repeated));
}

mod sampling;
