//! The readable source components consume ordinary time execution.
use eqiora::{api::ModelDocument, artifact::ModelEnvelope};
use eqiora_numerics::{CommonTimePlan, CommonTimeRunRequest, SamplingIdentity};
use std::ops::ControlFlow;

const COMPONENTS: &str =
    include_str!("../../eqiora-api/packages/Eqiora.Stochastic.Processes/src/processes.eqi");

#[test]
fn source_components_recover_their_zero_diffusion_recurrences() {
    // h=1/8. OU: x_n=mu+(x_0-mu)(1-lambda*h)^n.
    // GBM without noise: x_n=x_0(1+a*h)^n. These expectations
    // follow from the authored recurrences, independently of the executor.
    for (instance, expected) in [
        (
            "OrnsteinUhlenbeck(initial_value=1,mean=0.5,reversion_rate=2[1/s],volatility=0[1/s^(1/2)])",
            0.5 + 0.5 * 0.75_f64.powi(8),
        ),
        (
            "GeometricBrownianMotion(initial_value=1,growth_rate=0.25[1/s],volatility=0[1/s^(1/2)])",
            (1.0_f64 + 0.25 / 8.0).powi(8),
        ),
    ] {
        let source = format!("{COMPONENTS}\nmodel Main(){{instance process:{instance};}}");
        let document = ModelDocument::compile("processes.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let plan = CommonTimePlan::resolve_euler_maruyama(&model, document.program(), 3).unwrap();
        let request = CommonTimeRunRequest::sampled(
            plan.clone(),
            plan.initial_state(0.).unwrap(),
            1.,
            vec![1.],
            [0; 32],
            SamplingIdentity::new(b"reference", b"path", &[], None).unwrap(),
        )
        .unwrap();
        let trajectory = request
            .advance_process(None, |_, _| ControlFlow::Continue(()))
            .unwrap()
            .continue_value()
            .unwrap();
        let value = trajectory.time_states().unwrap()[0].values()[0];
        assert!((value - expected).abs() <= 4. * f64::EPSILON);
    }
}
