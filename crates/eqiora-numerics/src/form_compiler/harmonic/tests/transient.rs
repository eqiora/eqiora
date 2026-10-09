use super::*;
use crate::{CommonOdePlan, CommonOdePolicy, CommonOdeRunRequest, CommonTimeTolerance};
use eqiora_time::{ImplicitMidpointTimeBackend, TimeMethod};

pub(super) fn rc_harmonic_response_matches_settled_time_solution_without_claiming_initial_equivalence()
 {
    let times = vec![0.0005, 0.001, 0.005, 0.01, 0.02];
    let seconds = eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let mut identities = Vec::new();
    for initial_voltage in [0., 1.] {
        // Eliminate I between the original real laws I=C*dV/dt and Vs-V=R*I.
        // The existing scalar time IR admits sine: express the same cosine drive as
        // sin(omega*t+pi/2), with pi/2 rounded to binary64 in this bounded fixture.
        let time_source = format!("model TransientRC() {{
            parameter resistance:Ohm=1000[Ohm];
            parameter capacitance:F=1e-6[F];
            parameter omega:1/s=1000[1/s];
            state voltage:V;
            initial {{voltage={initial_voltage}[V];}}
            relation balance {{ derivative(voltage)=(1[V]*math.sin(omega*time()+1.5707963267948966)-voltage)/(resistance*capacitance); }}
        }}");
        let (transaction, id, symbols) = eqiora_compiler::compile("transient-rc.eqi", &time_source)
            .unwrap()
            .remove(0)
            .into_parts();
        let voltage = symbols.get("voltage").unwrap().downcast().unwrap();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), id).unwrap();
        let model = ModelEnvelope::from_program(&program).unwrap();
        let controls = CommonOdePolicy::new(
            TimeMethod::ImplicitMidpoint,
            1e-6,
            1e-13,
            vec![
                CommonTimeTolerance::new(
                    eqiora_core::TimeStateCoordinate::new(voltage, 0, 0, false),
                    1e-14,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let transient = CommonOdePlan::resolve(
            &model,
            &program,
            controls,
            ImplicitMidpointTimeBackend::CAPABILITIES,
        )
        .unwrap();
        let initial = transient.initial_state(0.).unwrap();
        assert_eq!(initial.values(), &[initial_voltage]);
        let run = CommonOdeRunRequest::new(transient, initial, 0.02, times.clone()).unwrap();
        let solution = ImplicitMidpointTimeBackend
            .solve(&run.problem().unwrap(), run.time_plan())
            .unwrap();

        let harmonic = resolve(&source().replace(
            "initial_voltage: V = 0 [V]",
            &format!("initial_voltage: V = {initial_voltage} [V]"),
        ))
        .unwrap();
        identities.push(harmonic.identity().to_owned());
        let result = harmonic
            .run_result(
                &harmonic.initial_state(&[]).unwrap(),
                &REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
        let original_voltage = harmonic
            .harmonic_amplitudes()
            .find(|(name, _, _)| *name == "voltage_hat")
            .unwrap()
            .1;
        let reconstruct = |time| {
            harmonic
                .reconstruct_harmonic_fields(&result, eqiora_core::DynQuantity::new(time, seconds))
                .unwrap()
                .into_iter()
                .find(|(field, _)| *field == original_voltage)
                .unwrap()
                .1
                .real_scalar_value()
                .unwrap()
                .value()
        };
        assert!((reconstruct(0.) - initial_voltage).abs() > 0.49);
        // Let s=t/(R*C), delta=h/(R*C)=0.001. Exact V=.5(cos(s)+sin(s))+(V0-.5)e^-s.
        // For V0 in {0,1}, |V''|,|V'''| <= sqrt(.5)+.5 < 1.21 in s coordinates.
        // Midpoint local defect <= (1.21/24+1.21/8)*delta^3/(1+delta/2).
        // Summing the contractive recurrence q=(1-delta/2)/(1+delta/2) gives <=.202*delta^2.
        // Linear sample interpolation adds <=1.21*delta^2/8. The 5e-7 V bound leaves
        // room above their <3.54e-7 V sum for binary64 arithmetic and Newton residuals.
        for (index, &time) in solution.times().iter().enumerate() {
            let s = 1000. * time;
            let settled = 0.5 * (s.cos() + s.sin());
            let decaying = (initial_voltage - 0.5) * (-s).exp();
            let numerical = solution.state(index).unwrap()[0];
            assert!(
                (numerical - settled - decaying).abs() < 5e-7,
                "t={time}, V0={initial_voltage}, V={numerical}"
            );
            assert!((reconstruct(time) - settled).abs() < 1e-10);
            assert!((numerical - reconstruct(time) - decaying).abs() < 5.001e-7);
        }
        // Positive R and C ensure decay; after 20 time constants its amplitude is <1.04e-9 V.
        assert!((solution.state(times.len() - 1).unwrap()[0] - reconstruct(0.02)).abs() < 5.02e-7);
    }
    assert_ne!(
        identities[0], identities[1],
        "retained initial conditions must change lineage"
    );
}
