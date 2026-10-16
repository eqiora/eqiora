use super::*;
use crate::{CommonTimePlan, ResolvedCommonPlan};
use eqiora_artifact::ModelEnvelope;

const WIENER: &str = r#"model Process() {
    noise W: wiener(covariance=1);
    state x: 1;
    initial { x=2; }
    law evolution for x { calculus ito; drift -x * 1[1/s]; diffusion W=1[1/s^(1/2)]; }
}"#;
const POISSON: &str = r#"model Process() {
    noise N: poisson(intensity=2[1/s]);
    state x: 1;
    initial { x=2; }
    law evolution for x { jump N=1; }
}"#;

#[test]
fn ordinary_time_plans_replay_both_process_profiles_and_complete_physical_states() {
    for source in [WIENER, POISSON] {
        let kernel = compile_process(source);
        let model = ModelEnvelope::from_program(&kernel).unwrap();
        let plan = if source == WIENER {
            CommonTimePlan::resolve_euler_maruyama(&model, &kernel, 8).unwrap()
        } else {
            CommonTimePlan::resolve_poisson_events(&model, &kernel, 64).unwrap()
        };
        let state = plan.initial_state(3.).unwrap();
        assert_eq!(state.values(), [2.]);
        assert_eq!(state.time_s(), 3.);
        assert_eq!(state.state_space_identity(), plan.state_space_identity());
        assert_eq!(
            plan.state_from_bytes(&state.to_bytes().unwrap()).unwrap(),
            state
        );
        let resolved = ResolvedCommonPlan::Time(Box::new(plan));
        assert!(resolved.as_ode().is_none());
        assert!(resolved.mesh_digest().is_none());
        let bytes = resolved.to_bytes().unwrap();
        let reopened = ResolvedCommonPlan::from_bytes(
            &bytes,
            &eqiora_solver::REFERENCE_LINEAR_SOLVER,
            eqiora_time::TimeBackendCapabilities::new(
                eqiora_time::TimeBackendIdentity::new("unused.ode.provider", "1"),
                &[],
                &[],
            ),
        )
        .unwrap();
        assert_eq!(resolved, reopened);
        assert_eq!(reopened.to_bytes().unwrap(), bytes);
        assert_eq!(
            reopened.as_time().unwrap().initial_state(3.).unwrap(),
            state
        );
        let mut noncanonical = bytes.clone();
        noncanonical.push(b'\n');
        assert!(
            ResolvedCommonPlan::from_bytes(
                &noncanonical,
                &eqiora_solver::REFERENCE_LINEAR_SOLVER,
                eqiora_time::TimeBackendCapabilities::new(
                    eqiora_time::TimeBackendIdentity::new("unused.ode.provider", "1"),
                    &[],
                    &[]
                )
            )
            .is_err()
        );
    }
}

#[test]
fn method_controls_change_plan_identity_without_changing_the_physical_state_space() {
    let kernel = compile_process(WIENER);
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let coarse = CommonTimePlan::resolve_euler_maruyama(&model, &kernel, 8).unwrap();
    let fine = CommonTimePlan::resolve_euler_maruyama(&model, &kernel, 12).unwrap();
    assert_ne!(coarse.identity(), fine.identity());
    assert_eq!(coarse.state_space_identity(), fine.state_space_identity());
    let state = coarse.initial_state(0.).unwrap();
    assert_eq!(
        fine.state_from_bytes(&state.to_bytes().unwrap()).unwrap(),
        state
    );
    assert!(CommonTimePlan::resolve_euler_maruyama(&model, &kernel, 53).is_err());
    assert!(CommonTimePlan::resolve_poisson_events(&model, &kernel, 64).is_err());
    let foreign = ModelEnvelope::from_program(&compile_process(POISSON)).unwrap();
    assert!(CommonTimePlan::resolve_euler_maruyama(&foreign, &kernel, 8).is_err());
}

#[test]
fn plan_admission_rejects_unsupported_calculus_and_dynamic_intensity() {
    let kernel = compile_process(&WIENER.replace("calculus ito", "calculus stratonovich"));
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    assert!(CommonTimePlan::resolve_euler_maruyama(&model, &kernel, 8).is_err());
    let kernel = compile_process(&POISSON.replace("intensity=2[1/s]", "intensity=x*2[1/s]"));
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let error = CommonTimePlan::resolve_poisson_events(&model, &kernel, 64).unwrap_err();
    assert!(error.message().contains("Model-constant"));
}

#[test]
fn ordinary_time_requests_bind_sample_and_output_controls_and_replay_the_same_owner() {
    use crate::CommonTimeRunRequest;
    use crate::common_time::request_artifact::WireTimeRequest;
    for source in [WIENER, POISSON] {
        let kernel = compile_process(source);
        let model = ModelEnvelope::from_program(&kernel).unwrap();
        let plan = if source == WIENER {
            CommonTimePlan::resolve_euler_maruyama(&model, &kernel, 8).unwrap()
        } else {
            CommonTimePlan::resolve_poisson_events(&model, &kernel, 64).unwrap()
        };
        let state = plan.initial_state(3.).unwrap();
        let make = |outputs, sample, lineage, master| {
            CommonTimeRunRequest::sampled(
                plan.clone(),
                state.clone(),
                4.,
                outputs,
                master,
                identity(sample, b"process", lineage),
            )
            .unwrap()
        };
        let request = make(vec![3.5], 1, &[4], [7; 32]);
        assert_eq!(request.execution_times_s, [3.5, 4.]);
        assert_eq!(request.plan(), &plan);
        assert_eq!(request.state(), &state);
        assert!(request.ode_plan().is_err());
        assert!(request.problem().is_err());
        assert!(request.time_plan().is_err());
        assert!(request.require_continuation().is_ok());
        for other in [
            make(vec![3.5, 4.], 1, &[4], [7; 32]),
            make(vec![3.5], 2, &[4], [7; 32]),
            make(vec![3.5], 1, &[5], [7; 32]),
            make(vec![3.5], 1, &[4], [8; 32]),
        ] {
            assert_ne!(request.identity(), other.identity());
            assert_eq!(request.plan(), other.plan());
            assert_eq!(request.state(), other.state());
        }
        let wire = WireTimeRequest::from_request(&request).unwrap();
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded: WireTimeRequest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.replay(&plan).unwrap(), request);
        let mut changed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        changed["output_times_s"] = serde_json::json!([3.25]);
        assert!(
            serde_json::from_value::<WireTimeRequest>(changed)
                .unwrap()
                .replay(&plan)
                .is_err()
        );
        assert!(CommonTimeRunRequest::new(plan.clone(), state, 4., vec![3.5]).is_err());
    }
}

#[test]
fn process_request_admission_never_rounds_an_output_onto_the_path_grid() {
    use crate::CommonTimeRunRequest;
    let kernel = compile_process(WIENER);
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let plan = CommonTimePlan::resolve_euler_maruyama(&model, &kernel, 8).unwrap();
    let make = |start, until, outputs| {
        CommonTimeRunRequest::sampled(
            plan.clone(),
            plan.initial_state(start).unwrap(),
            until,
            outputs,
            [7; 32],
            identity(1, b"process", &[]),
        )
    };
    for outputs in [
        vec![],
        vec![0.],
        vec![1.1],
        vec![0.5, 0.5],
        vec![0.5, 0.25],
        vec![f64::NAN],
    ] {
        assert!(
            make(0., 1., outputs)
                .unwrap_err()
                .message()
                .contains("output_times_s")
        );
    }
    assert!(
        make(0., 1., vec![0.1])
            .unwrap_err()
            .message()
            .contains("dyadic grid")
    );
    assert!(
        make(0., f64::INFINITY, vec![0.5])
            .unwrap_err()
            .message()
            .contains("until_s")
    );
    // The native path currently has a binary64 duration. Do not replace the
    // exact authored endpoints with a rounded subtraction when binding it.
    assert!(
        make(0.1, 1., vec![1.])
            .unwrap_err()
            .message()
            .contains("exactly representable")
    );
    assert!(make(3., 4., vec![3.25, 3.5, 4.]).is_ok());
}
