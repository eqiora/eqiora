mod support;

use std::sync::Arc;

use eqiora::api::{
    DifferentiableProgram, EvaluationMapPlan, ParameterSampler, SamplingCoupling,
    SamplingGenerator, SamplingIdentity,
};
use eqiora_numerics::CommonSpatialPolicy;
use support::common_scalar_plan::document_and_plan;

fn program() -> Arc<DifferentiableProgram> {
    let (document, plan) = document_and_plan(CommonSpatialPolicy::Q1);
    let inputs = [document.parameter_ref("diffusion").unwrap()];
    let output = document
        .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
        .unwrap();
    Arc::new(
        DifferentiableProgram::compile(
            eqiora_numerics::ResolvedCommonPlan::Linear(Box::new(plan)),
            &inputs,
            &output,
            None,
            &eqiora_solver::REFERENCE_LINEAR_SOLVER,
        )
        .unwrap(),
    )
}

fn request(id: u8, lineage: &[u64]) -> SamplingIdentity {
    SamplingIdentity::new(
        &[id],
        b"diffusion",
        lineage,
        SamplingCoupling::IndependentReplicate,
    )
    .unwrap()
}

#[test]
fn client_sampling_replays_through_typed_scalar_and_mapped_evaluation() {
    let program = program();
    let sampler = ParameterSampler::new(
        program.clone(),
        SamplingGenerator::Sha256UniformV1,
        [42; 32],
        &[[1.0, 2.0]],
    )
    .unwrap();
    let samples = (0..3)
        .map(|id| sampler.sample(&request(id, &[u64::from(id)])).unwrap())
        .collect::<Vec<_>>();
    let scalar = samples
        .iter()
        .map(|sample| program.evaluate(sample.point().values()).unwrap())
        .collect::<Vec<_>>();
    for order in [[0, 1, 2], [2, 0, 1]] {
        for chunk_size in [1, 2, 3] {
            for (chunk_index, chunk) in order.chunks(chunk_size).enumerate() {
                let drawn = chunk
                    .iter()
                    .enumerate()
                    .map(|(position, &id)| {
                        let sample = sampler
                            .sample(&request(id as u8, &[chunk_index as u64, position as u64]))
                            .unwrap();
                        assert_eq!(sample.point(), samples[id].point());
                        assert_eq!(sample.stream_digest(), samples[id].stream_digest());
                        assert_eq!(
                            sample.identity().lineage(),
                            &[chunk_index as u64, position as u64]
                        );
                        assert_eq!(sample.program_identity(), program.identity());
                        sample
                    })
                    .collect::<Vec<_>>();
                let points = drawn
                    .iter()
                    .map(|sample| sample.point().values())
                    .collect::<Vec<_>>();
                let mapped = EvaluationMapPlan::new(
                    program.clone(),
                    &points,
                    eqiora::api::EvaluationMapExecutionPolicy::retained(1 << 30),
                )
                .unwrap()
                .execute()
                .unwrap();
                for (&id, member) in chunk.iter().zip(mapped.members().unwrap()) {
                    assert_eq!(member.point(), scalar[id].point());
                    assert_eq!(member.primal().output(), scalar[id].primal().output());
                }
            }
        }
    }
    // Evaluation and derivative calls consume no stream state; explicit retry
    // returns the same frozen input even after arbitrary accepted operations.
    let evaluation = &scalar[0];
    let jvp = evaluation.jvp(&[1.0]).unwrap();
    let cotangent = vec![1.0; evaluation.primal().output().len()];
    let vjp = evaluation.vjp(&cotangent).unwrap();
    assert_eq!(sampler.sample(&request(0, &[0])).unwrap(), samples[0]);
    assert_eq!(evaluation.jvp(&[1.0]).unwrap().tangent(), jvp.tangent());
    assert_eq!(
        evaluation.vjp(&cotangent).unwrap().input_cotangent(),
        vjp.input_cotangent()
    );
    let parallel = std::thread::scope(|scope| {
        let jobs = (0..3)
            .map(|id| {
                let sampler = &sampler;
                scope.spawn(move || sampler.sample(&request(id, &[777])).unwrap())
            })
            .collect::<Vec<_>>();
        jobs.into_iter()
            .map(|job| job.join().unwrap())
            .collect::<Vec<_>>()
    });
    for (sample, expected) in parallel.iter().zip(&samples) {
        assert_eq!(sample.point(), expected.point());
        assert_eq!(sample.stream_digest(), expected.stream_digest());
    }
}

#[test]
fn duplicate_numerical_points_do_not_merge_replicates_and_crn_is_explicit() {
    let program = program();
    // Adjacent f64 endpoints force every draw onto the single admitted value.
    let sampler = ParameterSampler::new(
        program.clone(),
        SamplingGenerator::Sha256UniformV1,
        [0; 32],
        &[[1.0, 1.0f64.next_up()]],
    )
    .unwrap();
    let first = sampler.sample(&request(1, &[])).unwrap();
    let second = sampler.sample(&request(2, &[])).unwrap();
    assert_eq!(first.point(), second.point());
    assert_ne!(first.identity(), second.identity());
    assert_ne!(first.stream_digest(), second.stream_digest());
    let crn = |id| {
        SamplingIdentity::new(
            &[id],
            b"diffusion",
            &[],
            SamplingCoupling::CommonRandomNumbers(b"paired-comparison".to_vec()),
        )
        .unwrap()
    };
    let first = sampler.sample(&crn(1)).unwrap();
    let second = sampler.sample(&crn(2)).unwrap();
    assert_eq!(first.stream_digest(), second.stream_digest());
    assert_ne!(first.identity().sample_id(), second.identity().sample_id());
    assert_eq!(first.point().inputs(), program.identity().inputs());
}

#[test]
fn sampler_bounds_use_existing_input_admission_and_evaluation_remains_authoritative() {
    let program = program();
    for bounds in [
        vec![],
        vec![[0.0, 1.0]; 257],
        vec![[0.0, 1.0]; 2],
        vec![[1.0, 1.0]],
        vec![[f64::NAN, 1.0]],
        vec![[0.0, f64::INFINITY]],
        vec![[-f64::MAX, f64::MAX]],
    ] {
        assert!(
            ParameterSampler::new(
                program.clone(),
                SamplingGenerator::Sha256UniformV1,
                [0; 32],
                &bounds
            )
            .is_err()
        );
    }
    // A finite typed input can still fail physical diffusion admission. Drawing
    // is not acceptance of the residual, primal solve or derivative.
    let sampler = ParameterSampler::new(
        program.clone(),
        SamplingGenerator::Sha256UniformV1,
        [0; 32],
        &[[-2.0, -1.0]],
    )
    .unwrap();
    let sample = sampler.sample(&request(0, &[])).unwrap();
    assert!(program.evaluate(sample.point().values()).is_err());
    assert_eq!(sampler.sample(&request(0, &[])).unwrap(), sample);
}
