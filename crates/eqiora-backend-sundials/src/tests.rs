use crate::IdaTimeBackend;
use eqiora_core::Diagnostic;
use eqiora_time::*;
struct Branch(bool);
impl ImplicitTimeSystem for Branch {
    fn dimension(&self) -> usize {
        4
    }
    fn residual(&self, _: f64, y: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Diagnostic> {
        out.copy_from_slice(&[
            v[0] + y[1],
            v[1] - y[0],
            y[2] * y[2] - y[3] * y[3] - y[0],
            2. * y[2] * y[3] - y[1],
        ]);
        if self.0 {
            let (x, z) = ((1. + y[2] * y[2]) * out[0], out[2]);
            out[0] = x + z;
            out[2] = x - z;
        }
        Ok(())
    }
    fn residual_jvp(
        &self,
        _: f64,
        y: &[f64],
        v: &[f64],
        d: &[f64],
        dv: &[f64],
        out: &mut [f64],
    ) -> Result<(), Diagnostic> {
        out.copy_from_slice(&[
            dv[0] + d[1],
            dv[1] - d[0],
            2. * y[2] * d[2] - 2. * y[3] * d[3] - d[0],
            2. * y[3] * d[2] + 2. * y[2] * d[3] - d[1],
        ]);
        if self.0 {
            let (x, z) = (
                (1. + y[2] * y[2]) * out[0] + 2. * y[2] * d[2] * (v[0] + y[1]),
                out[2],
            );
            out[0] = x + z;
            out[2] = x - z;
        }
        Ok(())
    }
}
fn problem(
    system: &dyn ImplicitTimeSystem,
    policy: InitialConditionPolicy,
) -> ImplicitDaeProblem<'_> {
    ImplicitDaeProblem::new(
        system,
        vec![
            DaeVariableKind::Differential,
            DaeVariableKind::Differential,
            DaeVariableKind::Algebraic,
            DaeVariableKind::Algebraic,
        ],
        policy,
        vec![1., 0., 1., 0.],
        vec![0., 1., 0., 0.5],
    )
    .unwrap()
}
fn plan() -> TimePlan {
    TimePlan::new(
        TimeMethod::Bdf,
        0.,
        1e-4,
        1e-9,
        vec![1e-11; 4],
        (1..=128)
            .map(|i| std::f64::consts::TAU * f64::from(i) / 128.)
            .collect(),
    )
    .unwrap()
}
#[test]
fn original_and_state_dependent_row_bases_follow_the_same_continuous_branch() {
    for transformed in [false, true] {
        let system = Branch(transformed);
        let problem = problem(&system, InitialConditionPolicy::Provided);
        let solution = IdaTimeBackend::new().solve(&problem, &plan()).unwrap();
        for (i, &t) in solution.times().iter().enumerate() {
            let exact = [t.cos(), t.sin(), (t / 2.).cos(), (t / 2.).sin()];
            for (&actual, expected) in solution.state(i).unwrap().iter().zip(exact) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
        assert_eq!(solution.report().backend_version(), "7.8.0");
        assert_eq!(
            solution.report().equation_class(),
            TimeEquationClass::GeneralImplicitDae
        );
    }
}
#[test]
fn provided_pair_is_immutable_and_consistency_solve_holds_differential_state() {
    let system = Branch(true);
    let backend = IdaTimeBackend::new();
    let provided = problem(&system, InitialConditionPolicy::Provided);
    let accepted = backend.initialize(&provided, &plan()).unwrap();
    assert_eq!(accepted.state(), provided.initial_state());
    assert_eq!(accepted.derivative(), provided.initial_derivative());
    let guess = ImplicitDaeProblem::new(
        &system,
        provided.variable_kinds().to_vec(),
        InitialConditionPolicy::SolveConsistent,
        vec![1., 0., 1.1, 0.],
        vec![0., 0.9, 0., 0.5],
    )
    .unwrap();
    let accepted = backend.initialize(&guess, &plan()).unwrap();
    assert_eq!(&accepted.state()[..2], &[1., 0.]);
    assert_eq!(&accepted.derivative()[2..], &[0., 0.5]);
    let mut residual = vec![0.; 4];
    system
        .residual(0., accepted.state(), accepted.derivative(), &mut residual)
        .unwrap();
    assert!(residual.iter().all(|r| r.abs() < 1e-7));
    assert_eq!(guess.initial_state()[2], 1.1);
}
struct Failing {
    mode: u8,
}
impl ImplicitTimeSystem for Failing {
    fn dimension(&self) -> usize {
        4
    }
    fn residual(&self, t: f64, y: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Diagnostic> {
        match self.mode {
            0 => return Err(crate::runtime::failure("original callback diagnostic")),
            1 => panic!("residual panic"),
            4 => std::panic::panic_any(PanickingPayload),
            2 => {
                out.fill(f64::NAN);
                return Ok(());
            }
            _ => {}
        }
        Branch(false).residual(t, y, v, out)
    }
    fn residual_jvp(
        &self,
        t: f64,
        y: &[f64],
        v: &[f64],
        d: &[f64],
        dv: &[f64],
        out: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if self.mode == 3 {
            panic!("JVP panic");
        }
        Branch(false).residual_jvp(t, y, v, d, dv, out)
    }
}
#[test]
fn callback_error_nonfinite_and_panics_are_contained() {
    for mode in 0..5 {
        let system = Failing { mode };
        let problem = problem(&system, InitialConditionPolicy::Provided);
        let error = IdaTimeBackend::new().solve(&problem, &plan()).unwrap_err();
        let text = format!("{error:?}");
        assert!(
            text.contains(match mode {
                0 => "original callback diagnostic",
                2 => "non-finite",
                _ => "callback panicked",
            }),
            "{text}"
        );
    }
}
#[test]
fn method_mismatch_is_rejected() {
    let system = Branch(false);
    let problem = problem(&system, InitialConditionPolicy::Provided);
    let plan = TimePlan::new(
        TimeMethod::Tsitouras45,
        0.,
        1e-4,
        1e-9,
        vec![1e-11; 4],
        vec![0.01],
    )
    .unwrap();
    assert!(IdaTimeBackend::new().solve(&problem, &plan).is_err());
}

// A callback may panic with a payload whose destructor also panics. It must
// be disposed of after returning from C, preserving the original diagnostic.
struct PanickingPayload;
impl Drop for PanickingPayload {
    fn drop(&mut self) {
        panic!("panic payload destructor");
    }
}
struct BoundedTime {
    end: f64,
}
impl ImplicitTimeSystem for BoundedTime {
    fn dimension(&self) -> usize {
        4
    }
    fn residual(&self, t: f64, y: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Diagnostic> {
        if t > self.end {
            return Err(crate::runtime::failure("evaluated beyond terminal time"));
        }
        Branch(false).residual(t, y, v, out)
    }
    fn residual_jvp(
        &self,
        t: f64,
        y: &[f64],
        v: &[f64],
        d: &[f64],
        dv: &[f64],
        out: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if t > self.end {
            return Err(crate::runtime::failure("JVP beyond terminal time"));
        }
        Branch(false).residual_jvp(t, y, v, d, dv, out)
    }
}
#[test]
fn terminal_domain_bound_is_respected_with_independent_output_cadence() {
    let system = BoundedTime { end: 0.1 };
    let problem = problem(&system, InitialConditionPolicy::Provided);
    let mut histories = Vec::new();
    for times in [vec![0.1], vec![0.025, 0.05, 0.075, 0.1]] {
        let plan = TimePlan::new(TimeMethod::Bdf, 0., 1e-4, 1e-9, vec![1e-11; 4], times).unwrap();
        let solution = IdaTimeBackend::new().solve(&problem, &plan).unwrap();
        histories.push(solution.history().expect("native accepted history").clone());
        for (i, &t) in solution.times().iter().enumerate() {
            for (&actual, expected) in solution.state(i).unwrap().iter().zip([
                t.cos(),
                t.sin(),
                (t / 2.).cos(),
                (t / 2.).sin(),
            ]) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }
    assert_eq!(
        histories[0], histories[1],
        "output cadence must not select native steps"
    );
}

#[test]
fn reference_owner_matches_independent_discrete_branch_before_native_comparison() {
    for transformed in [false, true] {
        let system = Branch(transformed);
        let problem = problem(&system, InitialConditionPolicy::Provided);
        let steps = 256;
        let end = 0.1;
        let h: f64 = end / f64::from(steps);
        let reference_plan = TimePlan::new(
            TimeMethod::ImplicitEuler,
            0.,
            h,
            1e-9,
            vec![1e-11; 4],
            vec![end],
        )
        .unwrap();
        let reference = ReferenceImplicitTimeBackend::new()
            .solve(&problem, &reference_plan)
            .unwrap();
        // Independently solve the BE recurrence: u_n=(1-i*h)^(-n),
        // and retain the continuously selected square-root branch for z_n.
        let angle = f64::from(steps) * h.atan();
        let radius = (-0.5 * f64::from(steps) * (h * h).ln_1p()).exp();
        let discrete = [
            radius * angle.cos(),
            radius * angle.sin(),
            radius.sqrt() * (angle / 2.).cos(),
            radius.sqrt() * (angle / 2.).sin(),
        ];
        for (&actual, expected) in reference.state(0).unwrap().iter().zip(discrete) {
            assert!((actual - expected).abs() < 1e-7);
        }
        let native_plan =
            TimePlan::new(TimeMethod::Bdf, 0., 1e-4, 1e-9, vec![1e-11; 4], vec![end]).unwrap();
        let native = IdaTimeBackend::new().solve(&problem, &native_plan).unwrap();
        for (&actual, expected) in native.state(0).unwrap().iter().zip([
            end.cos(),
            end.sin(),
            (end / 2.).cos(),
            (end / 2.).sin(),
        ]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        // Both discretizations approximate the same branch. The reference's
        // O(h) error, not an observed provider difference, sets this bound.
        for (a, b) in native
            .state(0)
            .unwrap()
            .iter()
            .zip(reference.state(0).unwrap())
        {
            assert!((a - b).abs() < 4. * h + 1e-5);
        }
    }
}

#[test]
fn consistent_initialization_does_not_use_first_output_time_as_a_solver_control() {
    let system = RecordingScale(std::cell::Cell::new(None));
    let problem = ImplicitDaeProblem::new(
        &system,
        vec![
            DaeVariableKind::Differential,
            DaeVariableKind::Differential,
            DaeVariableKind::Algebraic,
            DaeVariableKind::Algebraic,
        ],
        InitialConditionPolicy::SolveConsistent,
        vec![1., 0., 1.1, 0.],
        vec![0., 0.9, 0., 0.5],
    )
    .unwrap();
    let mut histories = Vec::new();
    let mut scales = Vec::new();
    for times in [vec![0.1], vec![1e-9, 0.025, 0.05, 0.075, 0.1]] {
        let plan = TimePlan::new(TimeMethod::Bdf, 0., 1e-4, 1e-9, vec![1e-11; 4], times).unwrap();
        system.0.set(None);
        let solution = IdaTimeBackend::new().solve(&problem, &plan).unwrap();
        scales.push(system.0.get().expect("initial paired-JVP scale"));
        histories.push(solution.history().unwrap().clone());
    }
    assert_eq!(
        scales[0], scales[1],
        "initial Newton scale changed with output cadence"
    );
    assert!(
        histories[0] == histories[1],
        "consistent initialization changed with output cadence"
    );
}

struct RecordingScale(std::cell::Cell<Option<f64>>);
impl ImplicitTimeSystem for RecordingScale {
    fn dimension(&self) -> usize {
        4
    }
    fn residual(&self, t: f64, y: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Diagnostic> {
        Branch(true).residual(t, y, v, out)
    }
    fn residual_jvp(
        &self,
        t: f64,
        y: &[f64],
        v: &[f64],
        d: &[f64],
        dv: &[f64],
        out: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if self.0.get().is_none()
            && let Some(&scale) = dv.iter().find(|&&value| value != 0.)
        {
            self.0.set(Some(scale));
        }
        Branch(true).residual_jvp(t, y, v, d, dv, out)
    }
}

struct CubicRate;
impl ImplicitTimeSystem for CubicRate {
    fn dimension(&self) -> usize {
        1
    }
    fn residual(&self, _: f64, y: &[f64], v: &[f64], out: &mut [f64]) -> Result<(), Diagnostic> {
        out[0] = (v[0] + y[0]) * (1. + v[0] * v[0]);
        Ok(())
    }
    fn residual_jvp(
        &self,
        _: f64,
        y: &[f64],
        v: &[f64],
        dy: &[f64],
        dv: &[f64],
        out: &mut [f64],
    ) -> Result<(), Diagnostic> {
        out[0] = (1. + v[0] * v[0]) * (dy[0] + dv[0]) + 2. * v[0] * (v[0] + y[0]) * dv[0];
        Ok(())
    }
}

#[test]
fn nonlinear_rate_residual_uses_paired_actions_and_retains_the_analytic_solution() {
    // The strictly positive factor 1+v² makes F=(v+y)(1+v²)=0
    // equivalent to y'=-y for every real v. Thus y=exp(-t), and
    // F_v=1+y²>0 along the solution. This cannot be a TimeSystem M(t)v=f(t,y).
    let problem = ImplicitDaeProblem::new(
        &CubicRate,
        vec![DaeVariableKind::Differential],
        InitialConditionPolicy::Provided,
        vec![1.],
        vec![-1.],
    )
    .unwrap();
    let plan = TimePlan::new(
        TimeMethod::Bdf,
        0.,
        1e-4,
        1e-9,
        vec![1e-11],
        vec![0.1, 0.3, 1.],
    )
    .unwrap();
    let solution = IdaTimeBackend::new().solve(&problem, &plan).unwrap();
    for (i, &time) in solution.times().iter().enumerate() {
        assert!((solution.state(i).unwrap()[0] - (-time).exp()).abs() < 1e-6);
    }
    for step in solution.history().unwrap().steps() {
        let time = step.start_time() + 0.37 * (step.end_time() - step.start_time());
        assert!((step.sample(time).unwrap()[0] - (-time).exp()).abs() < 1e-6);
    }
}
