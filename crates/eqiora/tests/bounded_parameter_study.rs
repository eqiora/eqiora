mod support;

use std::sync::Arc;

use eqiora::api::{
    CompleteEvaluationMap, DifferentiableEvaluation, DifferentiableProgram,
    EvaluationMapOccurrence, EvaluationMapPlan, ModelDocument,
};
use eqiora::diagnostic::codes;
use eqiora_numerics::CommonScalarPlan;
use support::common_scalar_plan::document_and_plans;

const P0: [f64; 3] = [1.0, 1.0, 0.0];
const P1: [f64; 3] = [2.0, 0.75, 0.5];
const P2: [f64; 3] = [3.0, 1.25, -0.25];
const BYTE_LIMIT: usize = 1 << 30;

#[test]
fn bounded_public_policies_deliver_ordered_members_and_recompute_exact_receipts() {
    use eqiora::api::{EvaluationMapExecutionPolicy, EvaluationMapRetention};
    use eqiora::realization::Target;
    use std::num::NonZeroUsize;

    for program in programs() {
        let expected = [&P2, &P1, &P2].map(|point| program.evaluate(point).unwrap());
        for (chunk, workers) in [(1, 1), (2, 1), (2, 2), (3, 2)] {
            for retention in [
                EvaluationMapRetention::Retain,
                EvaluationMapRetention::Recompute,
            ] {
                let policy = EvaluationMapExecutionPolicy::new(
                    chunk,
                    Target::HostCpu {
                        threads: NonZeroUsize::new(workers).unwrap(),
                    },
                    retention,
                    BYTE_LIMIT,
                )
                .unwrap();
                let plan =
                    EvaluationMapPlan::new(program.clone(), &[&P2, &P1, &P2], policy).unwrap();
                let mut indices = Vec::new();
                let complete = plan
                    .execute_with_delivery(
                        || false,
                        |index, member| {
                            indices.push(index);
                            assert_evaluation_matches(member, &expected[index]);
                        },
                    )
                    .unwrap();
                assert_eq!(indices, [0, 1, 2]);
                assert_eq!(complete.len(), 3);
                for (index, expected) in expected.iter().enumerate() {
                    assert_eq!(
                        complete.receipt(index),
                        expected.primal().evidence().receipt()
                    );
                    match retention {
                        EvaluationMapRetention::Retain => {
                            assert_evaluation_matches(complete.evaluation(index).unwrap(), expected)
                        }
                        EvaluationMapRetention::Recompute => {
                            assert!(complete.members().is_none());
                            assert_evaluation_matches(
                                &complete.recompute(index).unwrap(),
                                expected,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ordered_duplicates_match_independent_q1_and_tpfa_evaluations() {
    for program in programs() {
        // Reference calls use the unchanged pointwise owner outside the map.
        let expected = [&P2, &P1, &P2].map(|point| program.evaluate(point).unwrap());
        let original_default = program.default_point().values().to_vec();
        let plan = EvaluationMapPlan::new(
            program.clone(),
            &[&P2, &P1, &P2],
            eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT),
        )
        .unwrap();
        assert_eq!(plan.program_identity(), program.identity());
        assert_eq!(plan.points().len(), 3);
        for (point, values) in plan.points().iter().zip([P2, P1, P2]) {
            assert_eq!(point.inputs(), program.identity().inputs());
            assert_point_bits(point.values(), &values);
        }
        assert_complete_matches(&plan.execute().unwrap(), &plan, &expected);
        assert_point_bits(program.default_point().values(), &original_default);
    }
}

#[test]
fn request_permutation_changes_membership_positions_without_sorting() {
    for program in programs() {
        let original = EvaluationMapPlan::new(
            program.clone(),
            &[&P2, &P1, &P2],
            eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT),
        )
        .unwrap();
        let permuted = EvaluationMapPlan::new(
            program.clone(),
            &[&P1, &P2, &P2],
            eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT),
        )
        .unwrap();
        assert_ne!(original, permuted);
        let expected = [&P1, &P2, &P2].map(|point| program.evaluate(point).unwrap());
        assert_complete_matches(&permuted.execute().unwrap(), &permuted, &expected);
        assert_eq!(
            original,
            EvaluationMapPlan::new(
                program,
                &[&P2, &P1, &P2],
                eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT)
            )
            .unwrap()
        );
    }
}

#[test]
fn retained_primal_jvp_and_vjp_survive_other_points_and_repeated_occurrences() {
    for program in programs() {
        let original = program.evaluate(&P0).unwrap();
        let direction = [0.25, -0.125, 0.5];
        let cotangent = vec![1.0; program.identity().output_dimension()];
        let expected_jvp = original.jvp(&direction).unwrap();
        let expected_vjp = original.vjp(&cotangent).unwrap();
        let plan = EvaluationMapPlan::new(
            program.clone(),
            &[&P0, &P1, &P0],
            eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT),
        )
        .unwrap();
        let complete = plan.execute().unwrap();
        let first = complete.evaluation(0).unwrap();
        let repeated = complete.evaluation(2).unwrap();
        assert_evaluation_matches(first, &original);
        assert_evaluation_matches(repeated, &original);
        let alternate = program.evaluate(&P2).unwrap();
        assert_ne!(alternate.point().values(), original.point().values());
        for retained in [&original, first, repeated] {
            assert_evaluation_matches(retained, &original);
            let jvp = retained.jvp(&direction).unwrap();
            let vjp = retained.vjp(&cotangent).unwrap();
            assert_point_bits(jvp.tangent(), expected_jvp.tangent());
            assert_point_bits(vjp.input_cotangent(), expected_vjp.input_cotangent());
            assert_eq!(jvp.evidence(), expected_jvp.evidence());
            assert_eq!(vjp.evidence(), expected_vjp.evidence());
        }
    }
}

#[test]
fn numerical_failure_retains_exact_accepted_prefix_and_indexed_terminal_states() {
    for program in programs() {
        let invalid = [1.0, -1.0, 0.0];
        let expected_failure = program.evaluate(&invalid).unwrap_err();
        let expected_first = program.evaluate(&P2).unwrap();
        let plan = EvaluationMapPlan::new(
            program,
            &[&P2, &invalid, &P1],
            eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT),
        )
        .unwrap();
        let report = plan.execute().unwrap_err();
        assert_eq!(report.plan(), &plan);
        assert!(!report.is_cancelled());
        assert_eq!(report.stopped_index(), 1);
        assert_eq!(report.diagnostics(), expected_failure);
        assert_evaluation_matches(report.accepted_members().next().unwrap().1, &expected_first);
        assert!(matches!(
            report.occurrence(0),
            Some(EvaluationMapOccurrence::Accepted(_))
        ));
        assert!(
            matches!(report.occurrence(1), Some(EvaluationMapOccurrence::Failed(d)) if d == expected_failure)
        );
        assert!(matches!(
            report.occurrence(2),
            Some(EvaluationMapOccurrence::NotStarted)
        ));
        assert!(report.occurrence(3).is_none());
        assert!(
            report
                .accepted_members()
                .next()
                .unwrap()
                .1
                .jvp(&[1.0, 0.0, 0.0])
                .is_ok()
        );
    }
}

#[test]
fn cancellation_preserves_prefix_and_final_completion() {
    for program in programs() {
        let plan = EvaluationMapPlan::new(
            program,
            &[&P2, &P1, &P2],
            eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT),
        )
        .unwrap();
        for after in [0, 1, 3] {
            let mut polls = 0;
            let outcome = plan.execute_with_cancellation(|| {
                let cancel = polls == after;
                polls += 1;
                cancel
            });
            if after == 3 {
                assert_eq!(outcome.unwrap().members().unwrap().len(), 3);
                assert_eq!(polls, 3);
            } else {
                let report = outcome.unwrap_err();
                assert!(report.is_cancelled());
                assert_eq!(report.stopped_index(), after);
                assert_eq!(report.accepted_members().count(), after);
                assert_eq!(report.diagnostics()[0].code(), codes::EXECUTION_CANCELLED);
                assert!(matches!(
                    report.occurrence(after),
                    Some(EvaluationMapOccurrence::Cancelled)
                ));
            }
        }
    }
}

#[test]
fn empty_singleton_and_structural_admission_use_the_exact_static_signature() {
    for program in programs() {
        let empty = EvaluationMapPlan::new(
            program.clone(),
            &[],
            eqiora::api::EvaluationMapExecutionPolicy::retained(0),
        )
        .unwrap();
        let complete = empty
            .execute_with_cancellation(|| panic!("empty map is complete"))
            .unwrap();
        assert!(complete.members().unwrap().is_empty());
        assert_eq!(complete.plan().program_identity(), program.identity());
        assert_eq!(empty.estimated_storage_bytes(), 0);
        let singleton = EvaluationMapPlan::new(
            program.clone(),
            &[&P2],
            eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT),
        )
        .unwrap();
        let expected = program.evaluate(&P2).unwrap();
        assert_evaluation_matches(
            singleton.execute().unwrap().evaluation(0).unwrap(),
            &expected,
        );
        assert!(
            EvaluationMapPlan::new(
                program.clone(),
                &[&P1, &[1.0]],
                eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT)
            )
            .is_err()
        );
        assert!(
            EvaluationMapPlan::new(
                program.clone(),
                &[&[1.0, f64::NAN, 0.0]],
                eqiora::api::EvaluationMapExecutionPolicy::retained(BYTE_LIMIT)
            )
            .is_err()
        );
        assert!(
            EvaluationMapPlan::new(
                program.clone(),
                &[&P1],
                eqiora::api::EvaluationMapExecutionPolicy::retained(0)
            )
            .is_err()
        );
        let required = singleton.estimated_storage_bytes();
        assert!(
            EvaluationMapPlan::new(
                program.clone(),
                &[&P2],
                eqiora::api::EvaluationMapExecutionPolicy::retained(required)
            )
            .is_ok()
        );
        assert!(
            EvaluationMapPlan::new(
                program,
                &[&P2],
                eqiora::api::EvaluationMapExecutionPolicy::retained(required - 1)
            )
            .is_err()
        );
    }
}

fn programs() -> [Arc<DifferentiableProgram>; 2] {
    let (document, q1, tpfa) = document_and_plans();
    [q1, tpfa].map(|plan| Arc::new(program_for(&document, plan)))
}

fn program_for(document: &ModelDocument, plan: CommonScalarPlan) -> DifferentiableProgram {
    let inputs = [
        document.parameter_ref("source_scale").unwrap(),
        document.parameter_ref("diffusion").unwrap(),
        document.parameter_ref("boundary_offset").unwrap(),
    ];
    let output = document
        .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
        .unwrap();
    DifferentiableProgram::compile(
        eqiora_numerics::ResolvedCommonPlan::Scalar(Box::new(plan)),
        &inputs,
        &output,
        None,
        &eqiora_solver::REFERENCE_LINEAR_SOLVER,
    )
    .unwrap()
}

fn assert_complete_matches(
    complete: &CompleteEvaluationMap,
    plan: &EvaluationMapPlan,
    expected: &[DifferentiableEvaluation],
) {
    assert_eq!(complete.plan(), plan);
    assert_eq!(complete.members().unwrap().len(), expected.len());
    for (index, (actual, expected)) in complete.members().unwrap().iter().zip(expected).enumerate()
    {
        assert_evaluation_matches(actual, expected);
        assert_evaluation_matches(complete.evaluation(index).unwrap(), expected);
    }
    assert!(complete.evaluation(expected.len()).is_none());
}

fn assert_evaluation_matches(
    actual: &DifferentiableEvaluation,
    expected: &DifferentiableEvaluation,
) {
    assert!(actual.identity() == expected.identity());
    assert_eq!(actual.point().inputs(), expected.point().inputs());
    assert_point_bits(actual.point().values(), expected.point().values());

    let actual_primal = actual.primal();
    let expected_primal = expected.primal();
    assert_point_bits(actual_primal.output(), expected_primal.output());
    assert_eq!(
        actual_primal.evidence().state_system(),
        expected_primal.evidence().state_system()
    );
    assert_eq!(
        actual_primal.evidence().receipt().unwrap().output(),
        expected_primal.evidence().receipt().unwrap().output()
    );
    assert_eq!(
        actual_primal.evidence().primal_solve(),
        expected_primal.evidence().primal_solve()
    );
    assert_eq!(actual_primal.evidence(), expected_primal.evidence());
}

fn assert_point_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    );
}
