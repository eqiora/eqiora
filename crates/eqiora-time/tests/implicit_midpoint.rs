use eqiora_core::Diagnostic;
use eqiora_time::*;

struct Oscillators {
    mass: [f64; 6],
    fail: bool,
}
impl TimeSystem for Oscillators {
    fn dimension(&self) -> usize {
        6
    }
    fn rhs(&self, time: f64, state: &[f64], out: &mut [f64]) -> Result<(), Diagnostic> {
        for (index, frequency) in [1., 2., 3.].into_iter().enumerate() {
            out[2 * index] = -frequency * state[2 * index + 1] * self.mass[2 * index];
            out[2 * index + 1] = frequency * state[2 * index] * self.mass[2 * index + 1];
        }
        if self.fail && time > 0.3 {
            out[1] = f64::NAN;
        }
        Ok(())
    }
    fn rhs_jvp(
        &self,
        time: f64,
        _: &[f64],
        direction: &[f64],
        out: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.rhs(time, direction, out)
    }
    fn mass_action(&self, _: f64, direction: &[f64], out: &mut [f64]) -> Result<(), Diagnostic> {
        for ((out, direction), mass) in out.iter_mut().zip(direction).zip(self.mass) {
            *out = direction * mass;
        }
        Ok(())
    }
}
fn solve(
    system: &Oscillators,
    start: f64,
    initial: Vec<f64>,
    h: f64,
    outputs: Vec<f64>,
) -> Result<TimeSolution, Diagnostic> {
    let problem = TimeProblem::new(
        system,
        TimeEquationClass::MassMatrix {
            rank: MassMatrixRank::Full,
        },
        InitialConditionPolicy::Provided,
        initial,
    )?;
    let plan = TimePlan::new(
        TimeMethod::ImplicitMidpoint,
        start,
        h,
        1e-12,
        vec![1e-14; 6],
        outputs,
    )?;
    ImplicitMidpointTimeBackend::new().solve(&problem, &plan)
}
fn seed() -> Vec<f64> {
    vec![1., 0., 2., 0., 3., 0.]
}
#[test]
fn cayley_phase_norm_convergence_and_sampling_share_one_history() {
    let system = Oscillators {
        mass: [1.; 6],
        fail: false,
    };
    let coarse = solve(&system, 0., seed(), 0.1, vec![1.]).unwrap();
    let dense = solve(&system, 0., seed(), 0.1, vec![0.035, 0.37, 0.72, 1.]).unwrap();
    assert_eq!(coarse.history(), dense.history());
    assert_eq!(coarse.state(0), dense.state(3));
    for (i, frequency) in [1_f64, 2., 3.].into_iter().enumerate() {
        let amplitude = (i + 1) as f64;
        let angle = 20. * (0.1 * frequency / 2.).atan();
        let values = coarse.state(0).unwrap();
        assert!((values[2 * i] - amplitude * angle.cos()).abs() < 1e-12);
        assert!((values[2 * i + 1] - amplitude * angle.sin()).abs() < 1e-12);
        assert!((values[2 * i].hypot(values[2 * i + 1]) - amplitude).abs() < 1e-12);
    }
    // Collocation interpolation is intentionally not projected onto the unit circle.
    let sample = dense.state(0).unwrap();
    assert!(sample[0].hypot(sample[1]) < 0.9995);
    let fine = solve(&system, 0., seed(), 0.05, vec![1.]).unwrap();
    let error = |s: &TimeSolution| {
        let value = s.state(0).unwrap();
        (value[0] - 1_f64.cos()).hypot(value[1] - 1_f64.sin())
    };
    assert!((3.9..4.1).contains(&(error(&coarse) / error(&fine))));
}
#[test]
fn restart_equation_rescaling_and_nonfinite_trial_preserve_accepted_state() {
    let system = Oscillators {
        mass: [1.; 6],
        fail: false,
    };
    let prefix = solve(&system, 0., seed(), 0.1, vec![0.5]).unwrap();
    let accepted = prefix.state(0).unwrap().to_vec();
    let resumed = solve(&system, 0.5, accepted.clone(), 0.1, vec![1.]).unwrap();
    let full = solve(&system, 0., seed(), 0.1, vec![1.]).unwrap();
    for (a, b) in resumed.state(0).unwrap().iter().zip(full.state(0).unwrap()) {
        assert!((a - b).abs() < 1e-12);
    }
    let scaled = Oscillators {
        mass: [2e200, 3e-200, 7e150, 4e-150, 5e100, 6e-100],
        fail: false,
    };
    let other = solve(&scaled, 0., seed(), 0.1, vec![1.]).unwrap();
    for (a, b) in other.state(0).unwrap().iter().zip(full.state(0).unwrap()) {
        assert!((a - b).abs() < 1e-12);
    }
    let failing = Oscillators {
        mass: [1.; 6],
        fail: true,
    };
    assert!(solve(&failing, 0.5, accepted.clone(), 0.1, vec![1.]).is_err());
    assert_eq!(accepted, prefix.state(0).unwrap());
    assert_eq!(
        solve(&system, 0.5, accepted, 0.1, vec![1.]).unwrap(),
        resumed
    );
}
#[test]
fn rounded_step_count_does_not_create_a_zero_length_step() {
    let system = Oscillators {
        mass: [1.; 6],
        fail: false,
    };
    let result = solve(&system, 0., seed(), 0.01, vec![0.07]).unwrap();
    assert_eq!(result.history().unwrap().steps().len(), 7);
}

#[test]
fn physical_rates_respect_equation_scaling_and_reject_invalid_mass_actions() {
    for mass in [
        [1.; 6],
        [2e200, 3e-200, 7e150, 4e-150, 5e100, 6e-100],
        [0.; 6],
        [f64::INFINITY; 6],
    ] {
        let system = Oscillators { mass, fail: false };
        let problem = TimeProblem::new(
            &system,
            TimeEquationClass::MassMatrix {
                rank: MassMatrixRank::Full,
            },
            InitialConditionPolicy::Provided,
            seed(),
        )
        .unwrap();
        let actual = problem.rate(0., &seed());
        if mass[0] == 0. || !mass[0].is_finite() {
            assert!(actual.is_err());
        } else {
            for (actual, expected) in actual.unwrap().iter().zip([0., 1., 0., 4., 0., 9.]) {
                assert!((actual - expected).abs() < 1e-13);
            }
        }
        assert!(problem.rate(f64::NAN, &seed()).is_err());
        assert!(problem.rate(0., &[1.]).is_err());
    }
}
