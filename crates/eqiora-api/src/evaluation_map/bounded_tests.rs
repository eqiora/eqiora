use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_numerics::{CommonSpatialPolicy, SamplingIdentity};
use eqiora_realization::{RealizationRevision, Target};

use super::products::tests::FixtureProducts;
use super::*;
use crate::ParameterSampler;

const LIMIT: usize = 1 << 30;
const POINTS: [[f64; 3]; 5] = [
    [3.0, 1.25, -0.25],
    [2.0, 0.75, 0.5],
    [3.0, 1.25, -0.25],
    [1.0, 2.0, 0.0],
    [2.0, 0.75, 0.5],
];

pub(super) fn registered_bounded_falsifiers() {
    staged_input_and_delivered_output_alias_mutations_cannot_retarget_accepted_members();
    real_numerical_failure_at_a_chunk_edge_does_not_contaminate_inflight_members();
    resident_payload_instrumentation_is_chunk_bounded_and_metadata_scales_with_inventory();
    partitions_parallel_and_recompute_match_independently_accepted_q1_and_tpfa();
    forced_out_of_order_completion_preserves_duplicate_occurrences_and_worker_bound();
    failed_and_cancelled_chunks_keep_indexed_inflight_success_without_retry();
    storage_admission_bounds_chunks_metadata_and_overflow_before_execution();
    frozen_sampling_identity_is_retained_across_chunks_and_recompute();
    outcomes::verify_recompute_evidence_falsifier();
}

#[test]
fn staged_input_and_delivered_output_alias_mutations_cannot_retarget_accepted_members() {
    let (_, program) = tests::fixture();
    let program = Arc::new(program);
    let mut caller = POINTS[0].to_vec();
    let plan = EvaluationMapPlan::new(
        program.clone(),
        &crate::evaluation_map::tests::typed_points(&program, &[&caller, &caller]),
        &[],
        policy(2, 2, EvaluationMapRetention::Retain),
    )
    .unwrap();
    caller.fill(9.0);
    let expected = program
        .evaluate(
            &program
                .point_from_coordinates(&POINTS[0])
                .expect("validated native point"),
        )
        .unwrap();
    let mut outputs = Vec::new();
    let complete = plan
        .execute_with_delivery(
            || false,
            |_, member| {
                let EvaluationMapOccurrence::Accepted(member) = member else {
                    panic!("expected a steady accepted member")
                };
                outputs.push(member.primal().output().coordinates().to_vec());
            },
        )
        .unwrap();
    outputs[0].fill(99.0);
    assert_eq!(outputs[1], expected.primal().output().coordinates());
    assert_member(complete.evaluation(0).unwrap(), &expected);
    assert_member(complete.evaluation(1).unwrap(), &expected);
    // Simulate an unsafe adapter accidentally reusing the previous accepted
    // buffer/linearization. Equal length and provider do not authorize reuse.
    let plan = EvaluationMapPlan::new(
        program.clone(),
        &crate::evaluation_map::tests::typed_points(&program, &[&POINTS[0], &POINTS[1]]),
        &[],
        policy(2, 1, EvaluationMapRetention::Retain),
    )
    .unwrap();
    let previous = Mutex::new(None);
    let report = plan
        .execute_with_evaluator(
            &|program, point| {
                let mut previous = previous.lock().unwrap();
                if previous.is_none() {
                    *previous = Some(program.evaluate(point).unwrap());
                }
                Ok(previous.as_ref().unwrap().clone())
            },
            &mut || false,
        )
        .unwrap_err();
    assert_eq!(report.stopped_index(), 1);
    assert!(
        report.diagnostics()[0]
            .message()
            .contains("exact requested point")
    );
    assert_member(report.accepted_members().next().unwrap().1, &expected);
}

#[test]
fn real_numerical_failure_at_a_chunk_edge_does_not_contaminate_inflight_members() {
    let (_, program) = tests::fixture();
    let program = Arc::new(program);
    let invalid = [1.0, -1.0, 0.0];
    let original = program
        .evaluate(
            &program
                .point_from_coordinates(&invalid)
                .expect("validated native point"),
        )
        .unwrap_err();
    let plan = EvaluationMapPlan::new(
        program.clone(),
        &crate::evaluation_map::tests::typed_points(
            &program,
            &[&POINTS[0], &POINTS[1], &invalid, &POINTS[3], &POINTS[4]],
        ),
        &[],
        policy(2, 2, EvaluationMapRetention::Recompute),
    )
    .unwrap();
    let report = plan.execute().unwrap_err();
    assert_eq!(report.stopped_index(), 2);
    assert_eq!(report.diagnostics(), original);
    for index in [0, 1, 3] {
        assert_member(
            &report.recompute(index).unwrap(),
            &program
                .evaluate(
                    &program
                        .point_from_coordinates(&POINTS[index])
                        .expect("validated native point"),
                )
                .unwrap(),
        );
    }
    assert!(matches!(
        report.occurrence(4),
        Some(EvaluationMapOccurrence::NotStarted)
    ));
}

#[test]
fn resident_payload_instrumentation_is_chunk_bounded_and_metadata_scales_with_inventory() {
    let (_, program) = tests::fixture();
    let program = Arc::new(program);
    let points = vec![POINTS[0].as_slice(); 17];
    for chunk in [1, 2, 5] {
        let plan = EvaluationMapPlan::new(
            program.clone(),
            &crate::evaluation_map::tests::typed_points(&program, &points),
            &[],
            policy(chunk, chunk.min(2), EvaluationMapRetention::Recompute),
        )
        .unwrap();
        let mut observed = Vec::new();
        let complete = schedule::execute(
            &plan,
            &|program, point| program.evaluate(point),
            &mut || false,
            &mut |_, _| {},
            &mut |count| observed.push(count),
        )
        .unwrap();
        let expected = points
            .chunks(chunk)
            .flat_map(|chunk| [chunk.len(), 0])
            .collect::<Vec<_>>();
        assert_eq!(
            observed, expected,
            "finished numerical buffers are released at every delivery boundary"
        );
        assert!(complete.members().is_none());
    }
    let first = EvaluationMapPlan::new(
        program.clone(),
        &crate::evaluation_map::tests::typed_points(&program, &points),
        &[],
        policy(2, 2, EvaluationMapRetention::Recompute),
    )
    .unwrap();
    let second = EvaluationMapPlan::new(
        program.clone(),
        &crate::evaluation_map::tests::typed_points(&program, &vec![POINTS[0].as_slice(); 34]),
        &[],
        policy(2, 2, EvaluationMapRetention::Recompute),
    )
    .unwrap();
    // Doubling the inventory must charge exactly the extra fixed membership and
    // frozen point records, not double the bounded numerical chunk payload.
    // Each point owns three f64 values; its Arc signature inventory is shared
    // with the Program and already included in the inline point record.
    // Derive the owned solve-provenance payload from the declared providers,
    // independently of the implementation's storage_bytes calculation.
    let provider_payload =
        |id: &str, version: &str, libraries: &[eqiora_solver::ProviderLibrary]| {
            id.len()
                + version.len()
                + libraries.len() * size_of::<(String, String)>()
                + libraries
                    .iter()
                    .map(|library| library.name().len() + library.version().len())
                    .sum::<usize>()
        };
    let solver = eqiora_solver::REFERENCE_SOLVER_PROVIDER;
    let execution = eqiora_solver::SERIAL_EXECUTION_PROVIDER;
    let solve_provenance = size_of::<eqiora_numerics::CommonSolveEvidence>()
        + provider_payload(
            solver.id().as_str(),
            solver.implementation_version(),
            solver.libraries(),
        )
        + 2 * provider_payload(
            execution.id().as_str(),
            execution.implementation_version(),
            execution.libraries(),
        )
        + 2 * eqiora_solver::ExecutionReport::host_serial()
            .adapter()
            .as_str()
            .len();
    let one_record = size_of::<DifferentiableValues>()
        + 3 * size_of::<f64>()
        + size_of::<outcomes::Outcome>()
        + size_of::<crate::DifferentiationEvidence>()
        + size_of::<crate::DifferentiationStepEvidence>()
        // This stationary fixture retains one separately allocated primal group
        // per occurrence; the step's inline Vec owns only its header.
        + size_of::<crate::differentiation::DifferentiationPrimalEvidence>()
        + 3 * size_of::<f64>() // evidence's frozen Parameter point
        + first.program_identity().unwrap().plan_identity().len()
        + 64 // canonical Model digest
        + solve_provenance
        + size_of::<DifferentiableEvaluation>();
    assert_eq!(
        second.estimated_storage_bytes() - first.estimated_storage_bytes(),
        17 * one_record
    );
}

fn policy(
    chunk: usize,
    workers: usize,
    retention: EvaluationMapRetention,
) -> EvaluationMapExecutionPolicy {
    EvaluationMapExecutionPolicy::new(
        chunk,
        Target::HostCpu {
            threads: NonZeroUsize::new(workers).unwrap(),
        },
        retention,
        LIMIT,
    )
    .unwrap()
}

#[test]
fn partitions_parallel_and_recompute_match_independently_accepted_q1_and_tpfa() {
    let (document, _) = tests::fixture();
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let program = Arc::new(tests::program_for(
            &document,
            spatial,
            RealizationRevision::new(21),
            &["source_scale", "diffusion", "boundary_offset"],
        ));
        let expected = POINTS
            .iter()
            .map(|point| {
                program
                    .evaluate(
                        &program
                            .point_from_coordinates(point)
                            .expect("validated native point"),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let points = POINTS
            .iter()
            .map(|point| point.as_slice())
            .collect::<Vec<_>>();
        for (chunk, workers) in [(1, 1), (2, 1), (3, 1), (2, 2), (5, 3)] {
            for retention in [
                EvaluationMapRetention::Retain,
                EvaluationMapRetention::Recompute,
            ] {
                let plan = EvaluationMapPlan::new(
                    program.clone(),
                    &crate::evaluation_map::tests::typed_points(&program, &points),
                    &[],
                    policy(chunk, workers, retention),
                )
                .unwrap();
                let mut delivered = Vec::new();
                let complete = plan
                    .execute_with_delivery(
                        || false,
                        |occurrence, evaluation| {
                            let index = occurrence.index();
                            delivered.push(index);
                            let EvaluationMapOccurrence::Accepted(evaluation) = evaluation else {
                                panic!("expected a steady accepted member")
                            };
                            assert_member(evaluation, &expected[index]);
                        },
                    )
                    .unwrap();
                assert_eq!(delivered, [0, 1, 2, 3, 4]);
                assert_eq!(complete.len(), 5);
                for (index, expected) in expected.iter().enumerate() {
                    assert_eq!(
                        *complete.evidence(index).unwrap(),
                        expected.primal_evidence()
                    );
                    match retention {
                        EvaluationMapRetention::Retain => {
                            assert_member(complete.evaluation(index).unwrap(), expected);
                            assert!(complete.recompute(index).is_err());
                        }
                        EvaluationMapRetention::Recompute => {
                            assert!(complete.members().is_none());
                            assert_member(&complete.recompute(index).unwrap(), expected);
                        }
                    }
                }
                let products =
                    EvaluationMapProducts::new(&complete, &[5], &[2], &[1], LIMIT).unwrap();
                let tangents = vec![0.25; 5 * 2 * 3];
                let jvp = products.fixture_jvp(&[], &tangents).unwrap();
                for (point, expected) in expected.iter().enumerate() {
                    for seed in 0..2 {
                        assert_eq!(
                            jvp.products()[seed * 5 + point],
                            expected
                                .jvp(
                                    &expected
                                        .identity()
                                        .tangent_from_coordinates(&[0.25; 3])
                                        .unwrap()
                                )
                                .unwrap()
                        );
                    }
                }
                let cotangents = vec![0.5; 5 * 2 * program.identity().output_dimension()];
                let vjp = products.fixture_vjp(&cotangents).unwrap();
                for (point, expected) in expected.iter().enumerate() {
                    for seed in 0..2 {
                        assert_eq!(
                            vjp.products()[seed * 5 + point],
                            expected
                                .vjp(
                                    &expected
                                        .identity()
                                        .cotangent_from_coordinates(&vec![
                                            0.5;
                                            program
                                                .identity()
                                                .output_dimension()
                                        ])
                                        .unwrap()
                                )
                                .unwrap()
                        );
                    }
                }
            }
        }
    }
}

fn assert_member(actual: &DifferentiableEvaluation, expected: &DifferentiableEvaluation) {
    assert_eq!(actual.identity(), expected.identity());
    assert_eq!(actual.point().signatures(), expected.point().signatures());
    assert!(exact_values(
        actual.point().coordinates(),
        expected.point().coordinates()
    ));
    assert_eq!(actual.primal(), expected.primal());
    assert_eq!(
        actual
            .jvp(
                &actual
                    .identity()
                    .tangent_from_coordinates(&[0.25, -0.5, 0.125])
                    .unwrap()
            )
            .unwrap(),
        expected
            .jvp(
                &expected
                    .identity()
                    .tangent_from_coordinates(&[0.25, -0.5, 0.125])
                    .unwrap()
            )
            .unwrap()
    );
    assert_eq!(
        actual
            .vjp(
                &actual
                    .identity()
                    .cotangent_from_coordinates(&vec![1.0; actual.identity().output_dimension()])
                    .unwrap()
            )
            .unwrap(),
        expected
            .vjp(
                &expected
                    .identity()
                    .cotangent_from_coordinates(&vec![1.0; expected.identity().output_dimension()])
                    .unwrap()
            )
            .unwrap()
    );
}

#[test]
fn forced_out_of_order_completion_preserves_duplicate_occurrences_and_worker_bound() {
    let (_, program) = tests::fixture();
    let plan = EvaluationMapPlan::new(
        Arc::new(program.clone()),
        &crate::evaluation_map::tests::typed_points(
            &program,
            &[&POINTS[0], &POINTS[1], &POINTS[0]],
        ),
        &[],
        policy(2, 2, EvaluationMapRetention::Retain),
    )
    .unwrap();
    let completed = Mutex::new(Vec::new());
    let condition = Condvar::new();
    let active = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let complete = plan
        .execute_with_evaluator(
            &|program, point| {
                let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(count, Ordering::SeqCst);
                if point.coordinates() == POINTS[0] {
                    let mut done = completed.lock().unwrap();
                    while done.is_empty() {
                        done = condition.wait(done).unwrap();
                    }
                }
                let value = program.evaluate(point);
                completed.lock().unwrap().push(point.coordinates()[0]);
                active.fetch_sub(1, Ordering::SeqCst);
                condition.notify_all();
                value
            },
            &mut || false,
        )
        .unwrap();
    assert_eq!(*completed.lock().unwrap(), [2.0, 3.0, 3.0]);
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert_eq!(active.load(Ordering::SeqCst), 0);
    for (index, point) in [POINTS[0], POINTS[1], POINTS[0]].iter().enumerate() {
        assert_eq!(
            complete.evaluation(index).unwrap().point().coordinates(),
            point
        );
    }
}

#[test]
fn failed_and_cancelled_chunks_keep_indexed_inflight_success_without_retry() {
    let (_, program) = tests::fixture();
    let program = Arc::new(program);
    for retention in [
        EvaluationMapRetention::Retain,
        EvaluationMapRetention::Recompute,
    ] {
        let plan = EvaluationMapPlan::new(
            program.clone(),
            &crate::evaluation_map::tests::typed_points(
                &program,
                &[&POINTS[0], &POINTS[1], &POINTS[2], &POINTS[3]],
            ),
            &[],
            policy(2, 2, retention),
        )
        .unwrap();
        let calls = AtomicUsize::new(0);
        let barrier = std::sync::Barrier::new(2);
        let original = Diagnostic::error(
            codes::INVALID_LINEARIZATION,
            "independent failed occurrence",
        );
        let report = plan
            .execute_with_evaluator(
                &|program, point| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    barrier.wait();
                    if point.coordinates() == POINTS[0] {
                        Err(vec![original.clone()])
                    } else {
                        program.evaluate(point)
                    }
                },
                &mut || false,
            )
            .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(report.stopped_index(), 0);
        assert_eq!(report.diagnostics(), [original]);
        assert!(matches!(
            report.occurrence(1),
            Some(EvaluationMapOccurrence::Accepted(_) | EvaluationMapOccurrence::Released(_))
        ));
        assert!(matches!(
            report.occurrence(2),
            Some(EvaluationMapOccurrence::NotStarted)
        ));
        assert!(report.recompute(0).is_err());
        if retention == EvaluationMapRetention::Recompute {
            assert_member(
                &report.recompute(1).unwrap(),
                &program
                    .evaluate(
                        &program
                            .point_from_coordinates(&POINTS[1])
                            .expect("validated native point"),
                    )
                    .unwrap(),
            );
        }
        for after in [0, 1, 2, 3, 4] {
            let mut polls = 0;
            let result = plan.execute_with_cancellation(|| {
                let cancel = polls == after;
                polls += 1;
                cancel
            });
            if after == 4 {
                assert_eq!(result.unwrap().len(), 4);
                assert_eq!(polls, 4);
            } else {
                let report = result.unwrap_err();
                assert_eq!(report.stopped_index(), after);
                assert!(report.is_cancelled());
                assert!(matches!(
                    report.occurrence(after),
                    Some(EvaluationMapOccurrence::Cancelled)
                ));
                for index in 0..after {
                    assert!(matches!(
                        report.occurrence(index),
                        Some(
                            EvaluationMapOccurrence::Accepted(_)
                                | EvaluationMapOccurrence::Released(_)
                        )
                    ));
                }
                for index in after + 1..4 {
                    assert!(matches!(
                        report.occurrence(index),
                        Some(EvaluationMapOccurrence::NotStarted)
                    ));
                }
            }
        }
    }
}

#[test]
fn storage_admission_bounds_chunks_metadata_and_overflow_before_execution() {
    let (_, program) = tests::fixture();
    let program = Arc::new(program);
    assert!(
        EvaluationMapExecutionPolicy::new(
            0,
            Target::HostCpu {
                threads: NonZeroUsize::MIN
            },
            EvaluationMapRetention::Recompute,
            LIMIT
        )
        .is_err()
    );
    assert!(
        EvaluationMapExecutionPolicy::new(
            1,
            Target::HostCpu {
                threads: NonZeroUsize::new(2).unwrap()
            },
            EvaluationMapRetention::Recompute,
            LIMIT
        )
        .is_err()
    );
    assert!(
        EvaluationMapExecutionPolicy::new(
            usize::MAX,
            Target::HostCpu {
                threads: NonZeroUsize::MIN
            },
            EvaluationMapRetention::Recompute,
            LIMIT
        )
        .is_err()
    );
    assert!(
        resources::estimate(
            &program,
            usize::MAX,
            policy(2, 2, EvaluationMapRetention::Recompute),
            0,
            0
        )
        .is_err()
    );
    let points = vec![POINTS[0].as_slice(); 16];
    let retained = EvaluationMapPlan::new(
        program.clone(),
        &crate::evaluation_map::tests::typed_points(&program, &points),
        &[],
        policy(2, 2, EvaluationMapRetention::Retain),
    )
    .unwrap();
    let released = EvaluationMapPlan::new(
        program.clone(),
        &crate::evaluation_map::tests::typed_points(&program, &points),
        &[],
        policy(2, 2, EvaluationMapRetention::Recompute),
    )
    .unwrap();
    assert!(retained.estimated_storage_bytes() > released.estimated_storage_bytes());
    let limit = released.estimated_storage_bytes();
    let admitted = EvaluationMapExecutionPolicy::new(
        2,
        Target::HostCpu {
            threads: NonZeroUsize::new(2).unwrap(),
        },
        EvaluationMapRetention::Recompute,
        limit,
    )
    .unwrap();
    assert!(
        EvaluationMapPlan::new(
            program.clone(),
            &crate::evaluation_map::tests::typed_points(&program, &points),
            &[],
            admitted
        )
        .is_ok()
    );
    let denied = EvaluationMapExecutionPolicy::new(
        2,
        Target::HostCpu {
            threads: NonZeroUsize::new(2).unwrap(),
        },
        EvaluationMapRetention::Recompute,
        limit - 1,
    )
    .unwrap();
    assert!(
        EvaluationMapPlan::new(
            program.clone(),
            &crate::evaluation_map::tests::typed_points(&program, &points),
            &[],
            denied
        )
        .is_err()
    );
    for retention in [
        EvaluationMapRetention::Retain,
        EvaluationMapRetention::Recompute,
    ] {
        let empty = EvaluationMapPlan::new(
            program.clone(),
            &[],
            &[],
            EvaluationMapExecutionPolicy::new(
                2,
                Target::HostCpu {
                    threads: NonZeroUsize::new(2).unwrap(),
                },
                retention,
                // Even an empty execution retains its 128-bit occurrence namespace.
                128 / 8,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(empty.estimated_storage_bytes(), 128 / 8);
        assert!(
            EvaluationMapPlan::new(
                program.clone(),
                &[],
                &[],
                EvaluationMapExecutionPolicy::new(
                    2,
                    Target::HostCpu {
                        threads: NonZeroUsize::new(2).unwrap()
                    },
                    retention,
                    (128 / 8) - 1,
                )
                .unwrap(),
            )
            .is_err()
        );
        assert!(
            empty
                .execute_with_cancellation(|| panic!("already complete"))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            EvaluationMapPlan::new(
                program.clone(),
                &crate::evaluation_map::tests::typed_points(&program, &[&POINTS[0]]),
                &[],
                policy(2, 2, retention)
            )
            .unwrap()
            .execute()
            .unwrap()
            .len(),
            1
        );
    }
}

#[test]
fn frozen_sampling_identity_is_retained_across_chunks_and_recompute() {
    let (document, program) = tests::fixture();
    let program = Arc::new(program);
    let sampler = ParameterSampler::new(
        program.clone(),
        [3; 32],
        &crate::DifferentiableValues::from_components(&[], &[]).unwrap(),
        &[[1.0, 3.0], [0.75, 1.25], [-0.25, 0.5]],
    )
    .unwrap();
    let id = |label: &[u8], lineage: &[u64]| {
        SamplingIdentity::new(label, b"forcing", lineage, Some(b"same-noise")).unwrap()
    };
    let samples = [
        sampler.sample(&id(b"second", &[8, 2])).unwrap(),
        sampler.sample(&id(b"first", &[8, 1])).unwrap(),
        sampler.sample(&id(b"second", &[8, 2])).unwrap(),
    ];
    assert_eq!(samples[0].point(), samples[1].point());
    assert_ne!(samples[0].identity(), samples[1].identity());
    let plan = EvaluationMapPlan::from_samples(
        program.clone(),
        &samples,
        &[program.identity().inputs()[0].parameter()],
        policy(2, 2, EvaluationMapRetention::Recompute),
    )
    .unwrap();
    assert_eq!(plan.samples().unwrap(), samples);
    let complete = plan.execute().unwrap();
    for (index, sample) in samples.iter().enumerate() {
        assert_member(
            &complete.recompute(index).unwrap(),
            &program.evaluate(sample.point()).unwrap(),
        );
        assert_eq!(&complete.plan().samples().unwrap()[index], sample);
    }
    assert_eq!(
        complete.plan().shared_inputs().unwrap(),
        &[program.identity().inputs()[0].parameter()]
    );
    let products = EvaluationMapProducts::new(&complete, &[3], &[], &[0], LIMIT).unwrap();
    let forward = products.fixture_jvp(&[1.0], &[0.0; 6]).unwrap();
    for (actual, sample) in forward.products().iter().zip(&samples) {
        let point = program.evaluate(sample.point()).unwrap();
        let tangent = point
            .identity()
            .tangent_from_coordinates(&[1.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(*actual, point.jvp(&tangent).unwrap());
    }
    for inputs in [
        ["source_scale", "diffusion", "boundary_offset"],
        ["diffusion", "source_scale", "boundary_offset"],
    ] {
        let foreign = Arc::new(tests::program_for(
            &document,
            CommonSpatialPolicy::Q1,
            RealizationRevision::new(22),
            &inputs,
        ));
        assert!(
            EvaluationMapPlan::from_samples(
                foreign,
                &samples,
                &[],
                policy(2, 2, EvaluationMapRetention::Recompute)
            )
            .is_err()
        );
    }
    let values = samples
        .iter()
        .map(|sample| sample.point().clone())
        .collect::<Vec<_>>();
    let raw = EvaluationMapPlan::new(
        program.clone(),
        &values,
        &[program.identity().inputs()[0].parameter()],
        policy(2, 2, EvaluationMapRetention::Recompute),
    )
    .unwrap();
    assert!(plan.estimated_storage_bytes() > raw.estimated_storage_bytes());
    assert_ne!(plan, raw);
}
