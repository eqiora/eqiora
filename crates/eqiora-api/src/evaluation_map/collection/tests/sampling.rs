use super::*;
use crate::{SampledParameterPoint, sampling::without_draws};

#[test]
fn frozen_sample_reopens_without_drawing_and_rejects_foreign_or_out_of_bounds_data() {
    let (_, program) = super::super::super::tests::fixture();
    let program = Arc::new(program);
    let fixed = DifferentiableValues::from_coordinates(
        program.identity().inputs()[..1].to_vec().into(),
        vec![1.25],
    )
    .unwrap();
    let sampler =
        ParameterSampler::new(program.clone(), [13; 32], &fixed, &[[0.5, 1.], [-0.5, 0.5]])
            .unwrap();
    let id = SamplingIdentity::new(b"frozen", b"record", &[4, 9], None).unwrap();
    let sample = sampler.sample(&id).unwrap();
    let bytes = sample.to_bytes().unwrap();
    let restored =
        without_draws(|| SampledParameterPoint::from_bytes(&bytes, program.identity())).unwrap();
    assert_eq!(restored, sample);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            without_draws(|| sampler.sample(&id).unwrap())
        }))
        .is_err()
    );
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut changed_fixed = original.clone();
    changed_fixed["point"][0] = serde_json::json!(2.);
    let mut upper_endpoint = original.clone();
    upper_endpoint["point"][1] = serde_json::json!(1.);
    for changed in [changed_fixed, upper_endpoint] {
        let error = without_draws(|| {
            SampledParameterPoint::from_bytes(
                &serde_json::to_vec(&changed).unwrap(),
                program.identity(),
            )
        })
        .unwrap_err();
        assert!(
            error.message().contains("fixed inputs or intervals"),
            "{error:?}"
        );
    }
    let mut foreign = original.clone();
    foreign["fixed_parameters"][0] = serde_json::json!("foreign");
    assert!(
        SampledParameterPoint::from_bytes(
            &serde_json::to_vec(&foreign).unwrap(),
            program.identity()
        )
        .unwrap_err()
        .message()
        .contains("absent from its Program")
    );
    let mut stream = original;
    stream["stream"][0] = serde_json::json!((sample.stream_digest()[0] as u16 + 1) % 256);
    assert!(
        SampledParameterPoint::from_bytes(
            &serde_json::to_vec(&stream).unwrap(),
            program.identity()
        )
        .unwrap_err()
        .message()
        .contains("Program or stream lineage")
    );
    for end in [0, bytes.len() / 2, bytes.len() - 1] {
        assert!(SampledParameterPoint::from_bytes(&bytes[..end], program.identity()).is_err());
    }
    let oversized = vec![0; eqiora_artifact::JsonDecoderLimits::default().max_bytes + 1];
    assert!(SampledParameterPoint::from_bytes(&oversized, program.identity()).is_err());
}

#[test]
fn collection_rejects_oversized_input_before_parsing_or_resolving_any_plan() {
    let oversized = vec![0; MAX_EXECUTION_ARTIFACT_BYTES + 1];
    assert!(reopen(&oversized).is_err());
}
