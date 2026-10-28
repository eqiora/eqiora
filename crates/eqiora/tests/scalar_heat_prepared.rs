//! Ordinary scalar-storage Runs retain only a private, Run-local provider session.
use eqiora::{
    Diagnostic,
    api::ModelDocument,
    artifact::{
        CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1,
        MeshProductionLineageEnvelopeV1, ModelEnvelope,
    },
    compiler::StaticBindingValue,
    diagnostic::codes,
    geometry::GeometryGraph,
    solver::{
        CanonicalCsrSystemView, CsrStorage, LinearProblem, LinearSolution, LinearSolver,
        LinearSolverBackend, PreparedLinearSolver, PreparedLinearStructureIdentity,
        ReductionPolicy, ReplicatedLinearExecution, SolverCapabilities, SolverPlan, SolverProvider,
    },
};
use eqiora_backend_faer::{FAER_SOLVER_PROVIDER, FaerLinearSolver};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, CommonState, CommonTemporalPolicy, CommonTrajectory,
    CommonTransientRunRequest, ResolvedCommonPlan,
};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    ops::ControlFlow,
    sync::{Arc, Mutex},
};

// Existing storage/vector.rs mode: grad(g) is [2,3] at the center, zero on
// boundary nodes. Two Q1 cells per axis give M_ii=1/9, K_ii=8/3, K/M=24.
const SOURCE: &str = r#"
public component Heat(support region:volume(ambient_dimension=2),
 support left:boundary(parent=region),support right:boundary(parent=region),
 support bottom:boundary(parent=region),support top:boundary(parent=region)) {
 variable g:m on region in smooth;
 relation potential on region {
   g=256[m]*(2*coordinate(0)/1[m]+3*coordinate(1)/1[m]-2.5)
      *(coordinate(0)/1[m])^2*(1-coordinate(0)/1[m])^2
      *(coordinate(1)/1[m])^2*(1-coordinate(1)/1[m])^2;
 }
 state u:vector<1,2> on region in smooth;
 initial {u=grad(g);}
 relation balance on region { CAPACITY*1[s/m^2]*derivative(u)=div(grad(u)); }
 relation fixed_left on left {trace(u)=0;}
 relation fixed_right on right {trace(u)=0;}
 relation fixed_bottom on bottom {trace(u)=0;}
 relation fixed_top on top {trace(u)=0;}
}
"#;

fn fixture(dt: f64, timed: bool, backend: &dyn LinearSolverBackend) -> ResolvedCommonPlan {
    fixture_source(dt, timed, backend, SOURCE)
}

fn fixture_source(
    dt: f64,
    timed: bool,
    backend: &dyn LinearSolverBackend,
    source: &str,
) -> ResolvedCommonPlan {
    fixture_source_options(dt, timed, backend, source, false)
}

fn fixture_source_options(
    dt: f64,
    timed: bool,
    backend: &dyn LinearSolverBackend,
    source: &str,
    nonlinear: bool,
) -> ResolvedCommonPlan {
    let graph = GeometryGraph::new();
    let square = graph.rectangle([0., 1.], [0., 1.]).unwrap();
    let edges = square.boundaries();
    let geometry = graph
        .build(
            &square,
            &BTreeMap::from([
                ("region".into(), vec![square.region().into()]),
                ("left".into(), vec![edges[0].into()]),
                ("right".into(), vec![edges[1].into()]),
                ("bottom".into(), vec![edges[2].into()]),
                ("top".into(), vec![edges[3].into()]),
            ]),
        )
        .unwrap();
    let region = geometry.entity_set("region").unwrap();
    let bindings = ["region", "left", "right", "bottom", "top"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "region").then_some(region),
            },
        )
    });
    let document = ModelDocument::compile_selected(
        "prepared-heat.eqi",
        &source.replace("CAPACITY", if timed { "(1+time()/1[s])" } else { "1" }),
        "Heat",
        &bindings,
    )
    .unwrap();
    let counts = if nonlinear { [3, 3] } else { [2, 2] };
    let cells = CartesianMeshCellsV2::new(counts).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(&geometry, counts)
            .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    let linear = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-12,
        1e-14,
        NonZeroUsize::new(if nonlinear { 1000 } else { 64 }).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let linear = CommonLinearRequest::exact(linear, FAER_SOLVER_PROVIDER).unwrap();
    let solve = if nonlinear {
        CommonSolvePolicy::Newton {
            linear,
            nonlinear: eqiora::realization::NonlinearSolvePlan::new(
                1e-10,
                1e-12,
                NonZeroUsize::new(12).unwrap(),
                12,
            )
            .unwrap(),
        }
    } else {
        CommonSolvePolicy::Linear(linear)
    };
    ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(document.program()).unwrap(),
        owner,
        CommonSpatialPolicy::Q1,
        solve,
        None,
        Some(CommonTemporalPolicy::backward_euler(dt).unwrap()),
        backend,
        None,
        None,
    )
    .unwrap()
}

#[derive(Debug, Default)]
struct Observed {
    prepares: usize,
    cold: usize,
    drops: usize,
    begins: usize,
    commits: usize,
    discards: usize,
    calls: Vec<(usize, PreparedLinearStructureIdentity, Vec<f64>, Vec<f64>)>,
}
#[derive(Debug)]
struct Probe {
    observed: Arc<Mutex<Observed>>,
    warm: bool,
    reject_call: Option<usize>,
    provider_error: bool,
    stale_provider: bool,
}
impl Probe {
    fn new(warm: bool, reject_call: Option<usize>) -> Self {
        Self {
            observed: Arc::default(),
            warm,
            reject_call,
            provider_error: false,
            stale_provider: false,
        }
    }
}
impl LinearSolverBackend for Probe {
    fn provider(&self) -> SolverProvider {
        FAER_SOLVER_PROVIDER
    }
    fn capabilities(&self) -> SolverCapabilities {
        FaerLinearSolver.capabilities()
    }
    fn prepare_linear(
        &self,
        plan: SolverPlan,
    ) -> Result<Option<Box<dyn PreparedLinearSolver>>, Diagnostic> {
        let mut o = self.observed.lock().unwrap();
        o.prepares += 1;
        let session = o.prepares;
        drop(o);
        if !self.warm {
            return Ok(None);
        }
        let inner = FaerLinearSolver
            .prepare_linear(plan)?
            .expect("ordinary sparse LU supports preparation");
        Ok(Some(Box::new(Session {
            inner,
            observed: self.observed.clone(),
            session,
            reject_call: self.reject_call,
            provider_error: self.provider_error,
            stale_provider: self.stale_provider,
        })))
    }
    fn solve_with_execution(
        &self,
        problem: &LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        self.observed.lock().unwrap().cold += 1;
        let solution = FaerLinearSolver.solve_with_execution(problem, plan, execution)?;
        if self.stale_provider {
            return stale_provider_candidate(problem, solution);
        }
        Ok(solution)
    }
}
#[derive(Debug)]
struct Session {
    inner: Box<dyn PreparedLinearSolver>,
    observed: Arc<Mutex<Observed>>,
    session: usize,
    reject_call: Option<usize>,
    provider_error: bool,
    stale_provider: bool,
}
impl Drop for Session {
    fn drop(&mut self) {
        self.observed.lock().unwrap().drops += 1;
    }
}
impl PreparedLinearSolver for Session {
    fn begin_action(&mut self) -> Result<(), Diagnostic> {
        self.inner.begin_action()?;
        self.observed.lock().unwrap().begins += 1;
        Ok(())
    }
    fn commit_action(&mut self) -> Result<(), Diagnostic> {
        self.inner.commit_action()?;
        self.observed.lock().unwrap().commits += 1;
        Ok(())
    }
    fn discard_action(&mut self) -> Result<(), Diagnostic> {
        self.inner.discard_action()?;
        self.observed.lock().unwrap().discards += 1;
        Ok(())
    }
    fn solve(
        &mut self,
        structure: &PreparedLinearStructureIdentity,
        problem: &LinearProblem<'_>,
    ) -> Result<LinearSolution, Diagnostic> {
        let system = problem
            .canonical_csr_system()
            .expect("ordinary canonical operator");
        let mut o = self.observed.lock().unwrap();
        o.calls.push((
            self.session,
            structure.clone(),
            system.values().to_vec(),
            system.right_hand_side().to_vec(),
        ));
        let occurrence = o.calls.len();
        drop(o);
        if self.reject_call == Some(occurrence) {
            if self.provider_error {
                return Err(Diagnostic::error(
                    codes::NUMERICAL_SOLVE_FAILED,
                    "injected prepared provider failure",
                ));
            }
            // Real Faer accepts and retains factors for 2A. The outer spatial owner
            // must reject its otherwise successful candidate against the original A.
            let storage = Doubled {
                system,
                values: system.values().iter().map(|v| 2. * v).collect(),
            };
            let wrong = CanonicalCsrSystemView::new(
                &storage,
                system.right_hand_side(),
                system.properties(),
            )?;
            let solution = self.inner.solve(structure, &wrong.linear_problem()?)?;
            assert!(solution.report().true_residual_norm() <= solution.report().residual_target());
            return Ok(solution);
        }
        let solution = self.inner.solve(structure, problem)?;
        if self.stale_provider {
            return stale_provider_candidate(problem, solution);
        }
        Ok(solution)
    }
}

fn stale_provider_candidate(
    problem: &LinearProblem<'_>,
    solution: LinearSolution,
) -> Result<LinearSolution, Diagnostic> {
    let (values, report) = solution.into_parts();
    eqiora::solver::LinearSolution::accept(
        problem,
        report.solver_plan(),
        SolverProvider::new(
            FAER_SOLVER_PROVIDER.id(),
            "stale",
            FAER_SOLVER_PROVIDER.libraries(),
        ),
        report.reason(),
        report.completed_iterations(),
        report.reported_residual_norm(),
        values,
    )
}
struct Doubled<'a> {
    system: &'a CanonicalCsrSystemView,
    values: Vec<f64>,
}
impl CsrStorage for Doubled<'_> {
    fn rows(&self) -> usize {
        self.system.rows()
    }
    fn columns(&self) -> usize {
        self.system.columns()
    }
    fn row_offsets(&self) -> &[usize] {
        self.system.row_offsets()
    }
    fn column_indices(&self) -> &[usize] {
        self.system.column_indices()
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
}

fn initial(plan: &ResolvedCommonPlan) -> CommonState {
    plan.as_spatial()
        .unwrap()
        .initial_state(Some(0.), Vec::new())
        .unwrap()
}
fn run(
    plan: &ResolvedCommonPlan,
    state: CommonState,
    steps: usize,
    backend: &dyn LinearSolverBackend,
) -> CommonTrajectory {
    let request =
        CommonTransientRunRequest::from_steps(plan.clone(), state, steps, (1..=steps).collect())
            .unwrap();
    let ControlFlow::Continue(outputs) = request
        .advance_accepted_actions(backend, |_, _| false)
        .unwrap()
    else {
        panic!("complete heat Run")
    };
    CommonTrajectory::accept_spatial_transient(request, outputs).unwrap()
}
fn assert_state(plan: &ResolvedCommonPlan, state: &CommonState, factor: f64) {
    let (_, _, _, values) = plan
        .as_spatial()
        .unwrap()
        .state_fields(state)
        .unwrap()
        .next()
        .unwrap();
    let mut expected = [0.; 18];
    expected[8] = 2. * factor;
    expected[9] = 3. * factor;
    assert_eq!(values.len(), expected.len());
    for (actual, want) in values.iter().zip(expected) {
        assert!((actual - want).abs() < 1e-12, "{actual} != {want}");
    }
}
fn assert_trajectory(
    plan: &ResolvedCommonPlan,
    trajectory: &CommonTrajectory,
    dt: f64,
    timed: bool,
    start: usize,
    mut factor: f64,
) {
    let result = CommonResult::accept_trajectory(0., trajectory.clone()).unwrap();
    for (index, (_, step)) in trajectory.spatial_steps().unwrap().iter().enumerate() {
        let c = if timed {
            1. + (start + index + 1) as f64 * dt
        } else {
            1.
        };
        factor *= c / (c + 24. * dt);
        assert_state(plan, step.state(), factor);
        let evidence = result.solve_evidence(Some(index), None).unwrap();
        assert_eq!(evidence.solver().id(), FAER_SOLVER_PROVIDER.id().as_str());
        assert!(evidence.true_residual_norm() <= evidence.residual_target());
    }
}

#[test]
fn ordinary_heat_warm_and_cold_match_with_rhs_only_reuse() {
    let warm = Probe::new(true, None);
    let cold = Probe::new(false, None);
    let plan = fixture(1. / 24., false, &warm);
    let state = initial(&plan);
    let warm_run = run(&plan, state.clone(), 3, &warm);
    let cold_run = run(&plan, state, 3, &cold);
    for ((_, a), (_, b)) in warm_run
        .spatial_steps()
        .unwrap()
        .iter()
        .zip(cold_run.spatial_steps().unwrap())
    {
        assert_eq!(a.state(), b.state());
    }
    assert_trajectory(&plan, &warm_run, 1. / 24., false, 0, 1.);
    let o = warm.observed.lock().unwrap();
    assert_eq!((o.prepares, o.cold, o.drops, o.calls.len()), (1, 0, 1, 3));
    for pair in o.calls.windows(2) {
        assert_eq!(pair[0].1, pair[1].1);
        assert_eq!(pair[0].2, pair[1].2);
        assert_ne!(pair[0].3, pair[1].3);
    }
    let o = cold.observed.lock().unwrap();
    assert_eq!((o.prepares, o.cold, o.drops, o.calls.len()), (1, 3, 0, 0));
}

#[test]
fn time_capacity_changes_coefficients_and_distinct_dt_starts_cold() {
    let warm = Probe::new(true, None);
    let cold = Probe::new(false, None);
    for dt in [1. / 24., 1. / 48.] {
        let plan = fixture(dt, true, &warm);
        let state = initial(&plan);
        let a = run(&plan, state.clone(), 3, &warm);
        let b = run(&plan, state, 3, &cold);
        assert_trajectory(&plan, &a, dt, true, 0, 1.);
        for ((_, x), (_, y)) in a
            .spatial_steps()
            .unwrap()
            .iter()
            .zip(b.spatial_steps().unwrap())
        {
            assert_eq!(x.state(), y.state());
        }
    }
    let o = warm.observed.lock().unwrap();
    assert_eq!((o.prepares, o.cold, o.drops, o.calls.len()), (2, 0, 2, 6));
    for session in o.calls.chunks(3) {
        for pair in session.windows(2) {
            assert_eq!(pair[0].1, pair[1].1);
            assert_ne!(pair[0].2, pair[1].2);
        }
    }
    assert_ne!(o.calls[0].0, o.calls[3].0);
    assert_ne!(o.calls[0].2, o.calls[3].2);
}

#[test]
fn outer_acceptance_error_drops_session_before_another_boundary_or_retry() {
    let backend = Probe::new(true, Some(2));
    let plan = fixture(1. / 24., false, &backend);
    let state = initial(&plan);
    let before = state.to_bytes().unwrap();
    let request =
        CommonTransientRunRequest::from_steps(plan.clone(), state.clone(), 4, vec![1, 2, 3, 4])
            .unwrap();
    let mut boundaries = Vec::new();
    let mut last = None;
    let error = request
        .advance_accepted_actions(&backend, |step, state| {
            boundaries.push(step);
            last = Some(state.clone());
            false
        })
        .unwrap_err();
    assert_eq!(error.code(), codes::NUMERICAL_SOLVE_FAILED);
    assert!(error.message().contains("residual"));
    assert_eq!(boundaries, vec![0, 1]);
    assert_eq!(state.to_bytes().unwrap(), before);
    let o = backend.observed.lock().unwrap();
    assert_eq!((o.prepares, o.drops, o.calls.len()), (1, 1, 2));
    assert_eq!((o.begins, o.commits, o.discards), (2, 1, 1));
    drop(o);
    let accepted = last.unwrap();
    assert_state(&plan, &accepted, 0.5);
    let accepted = CommonState::from_bytes(&accepted.to_bytes().unwrap(), &plan).unwrap();
    let resumed = run(&plan, accepted, 2, &backend);
    assert_trajectory(&plan, &resumed, 1. / 24., false, 1, 0.5);
    let o = backend.observed.lock().unwrap();
    assert_eq!((o.prepares, o.drops, o.calls.len()), (2, 2, 4));
    assert_ne!(o.calls[1].0, o.calls[2].0);
}

#[test]
fn cancellation_resume_uses_a_new_session_and_only_accepted_state() {
    for boundary in [0, 2] {
        let backend = Probe::new(true, None);
        let plan = fixture(1. / 24., false, &backend);
        let state = initial(&plan);
        let request =
            CommonTransientRunRequest::from_steps(plan.clone(), state, 4, vec![1, 2, 3, 4])
                .unwrap();
        let ControlFlow::Break((step, state)) = request
            .advance_accepted_actions(&backend, |step, _| step == boundary)
            .unwrap()
        else {
            panic!("cancel at accepted boundary")
        };
        assert_eq!(step, boundary);
        let factor = 0.5_f64.powi(boundary as i32);
        assert_state(&plan, &state, factor);
        let o = backend.observed.lock().unwrap();
        assert_eq!((o.prepares, o.drops, o.calls.len()), (1, 1, boundary));
        drop(o);
        let state = CommonState::from_bytes(&state.to_bytes().unwrap(), &plan).unwrap();
        let resumed = run(&plan, state, 2, &backend);
        assert_trajectory(&plan, &resumed, 1. / 24., false, boundary, factor);
        let o = backend.observed.lock().unwrap();
        assert_eq!((o.prepares, o.drops, o.calls.len()), (2, 2, boundary + 2));
        assert_eq!(
            o.calls.iter().map(|call| call.0).collect::<Vec<_>>(),
            std::iter::repeat_n(1, boundary)
                .chain([2, 2])
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn prepared_provider_failure_is_not_a_cold_fallback() {
    let mut backend = Probe::new(true, Some(2));
    backend.provider_error = true;
    let plan = fixture(1. / 24., false, &backend);
    let request =
        CommonTransientRunRequest::from_steps(plan.clone(), initial(&plan), 4, vec![1, 2, 3, 4])
            .unwrap();
    let mut boundaries = Vec::new();
    let error = request
        .advance_accepted_actions(&backend, |step, _| {
            boundaries.push(step);
            false
        })
        .unwrap_err();
    assert_eq!(error.message(), "injected prepared provider failure");
    assert_eq!(boundaries, vec![0, 1]);
    let observed = backend.observed.lock().unwrap();
    assert_eq!(
        (
            observed.prepares,
            observed.cold,
            observed.drops,
            observed.calls.len()
        ),
        (1, 0, 1, 2)
    );
}

#[test]
fn later_state_certificate_failure_drops_the_successful_linear_session() {
    // Existing time_data.rs falsifier: at t=1 the BE stiffness is regular and
    // the zero solution solves it, but the original storage Jacobian is zero.
    let backend = Probe::new(true, None);
    let source = SOURCE.replace("CAPACITY", "(1-time()/1[s])");
    let plan = fixture_source(1., false, &backend, &source);
    let state = initial(&plan);
    let before = state.to_bytes().unwrap();
    let request =
        CommonTransientRunRequest::from_steps(plan, state.clone(), 3, vec![1, 2, 3]).unwrap();
    let mut boundaries = Vec::new();
    let error = request
        .advance_accepted_actions(&backend, |step, _| {
            boundaries.push(step);
            false
        })
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("Jacobian is singular at its consistent rate"),
        "{error:?}"
    );
    assert_eq!(boundaries, vec![0]);
    assert_eq!(state.to_bytes().unwrap(), before);
    let observed = backend.observed.lock().unwrap();
    assert_eq!(
        (
            observed.prepares,
            observed.cold,
            observed.drops,
            observed.calls.len()
        ),
        (1, 0, 1, 1)
    );
    assert_eq!(
        (observed.begins, observed.commits, observed.discards),
        (1, 0, 1)
    );
}

#[test]
fn false_provider_release_is_rejected_before_callback_or_reuse() {
    for (warm, expected) in [(true, (1, 0, 1, 1)), (false, (1, 1, 0, 0))] {
        let mut backend = Probe::new(warm, None);
        backend.stale_provider = true;
        let plan = fixture(1. / 24., false, &backend);
        let request =
            CommonTransientRunRequest::from_steps(plan.clone(), initial(&plan), 3, vec![1, 2, 3])
                .unwrap();
        let mut boundaries = Vec::new();
        let error = request
            .advance_accepted_actions(&backend, |step, _| {
                boundaries.push(step);
                false
            })
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert_eq!(boundaries, vec![0]);
        let observed = backend.observed.lock().unwrap();
        assert_eq!(
            (
                observed.prepares,
                observed.cold,
                observed.drops,
                observed.calls.len()
            ),
            expected
        );
    }
}

#[test]
fn complex_physical_fields_keep_the_existing_real_coordinate_prepared_route() {
    let warm = Probe::new(true, None);
    let cold = Probe::new(false, None);
    let source = SOURCE
        .replace("u:vector<1,2>", "u:vector<complex<1>,2>")
        .replace("u=grad(g)", "u=math.complex(1,2)*grad(g)");
    let plan = fixture_source(1. / 24., false, &warm, &source);
    let state = initial(&plan);
    let a = run(&plan, state.clone(), 2, &warm);
    let b = run(&plan, state, 2, &cold);
    for (index, ((_, x), (_, y))) in a
        .spatial_steps()
        .unwrap()
        .iter()
        .zip(b.spatial_steps().unwrap())
        .enumerate()
    {
        assert_eq!(x.state(), y.state());
        let (_, _, _, values) = plan
            .as_spatial()
            .unwrap()
            .state_fields(x.state())
            .unwrap()
            .next()
            .unwrap();
        let mut expected = [0.; 36];
        for (offset, value) in [2., 4., 3., 6.].into_iter().enumerate() {
            expected[16 + offset] = value * 0.5_f64.powi(index as i32 + 1);
        }
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
    let observed = warm.observed.lock().unwrap();
    assert_eq!(
        (
            observed.prepares,
            observed.cold,
            observed.drops,
            observed.calls.len()
        ),
        (1, 0, 1, 2)
    );
}

// Same two-field fixture and physical coefficients as the numerical-admission
// coupled_newton test. The zero-force control is exactly zero: every derivative,
// transport, diffusion and pairwise interaction term vanishes independently.
fn coupled_source(force: f64) -> String {
    let mut source = String::from(
        r#"
public operator dyad(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);
public component Heat(support region:volume(ambient_dimension=2),
 support left:boundary(parent=region),support right:boundary(parent=region),
 support bottom:boundary(parent=region),support top:boundary(parent=region)) {
"#,
    );
    for name in ["a", "b"] {
        source +=
            &format!("state {name}:vector<m/s,2> on region in smooth; initial {{ {name}=0; }}\n");
    }
    for (name, other) in [("a", "b"), ("b", "a")] {
        let forcing = if name == "a" { force } else { 0. };
        source += &format!(
            "relation balance_{name} on region {{ 4[kg/m^3]*derivative({name}) + div(4[kg/m^3]*dyad(left={name},right={name})) - div(0.1[kg/(m*s)]*grad({name})) + 2[kg/(m^3*s)]*({name}-{other}) - {forcing}[kg/(m^2*s^2)]*grad(coordinate(0)) = 0; }}\n"
        );
        for boundary in ["left", "right", "bottom", "top"] {
            source += &format!("relation {boundary}_{name} on {boundary} {{ trace({name})=0; }}\n");
        }
    }
    source += "}";
    source
}

#[test]
fn nonlinear_run_prepares_once_and_refreshes_each_changed_jacobian() {
    for force in [0., 3.] {
        let backend = Probe::new(true, None);
        let plan = fixture_source_options(0.125, false, &backend, &coupled_source(force), true);
        // This exact admitted provider/policy supports preparation.
        let linear = SolverPlan::new(
            LinearSolver::SparseLu,
            1e-12,
            1e-14,
            NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        assert!(
            <FaerLinearSolver as LinearSolverBackend<f64>>::prepare_linear(
                &FaerLinearSolver,
                linear
            )
            .unwrap()
            .is_some()
        );
        let state = initial(&plan);
        let before = state.to_bytes().unwrap();
        let trajectory = run(&plan, state.clone(), 2, &backend);
        let cold = Probe::new(false, None);
        let cold_trajectory = run(&plan, state.clone(), 2, &cold);
        for ((_, warm), (_, cold)) in trajectory
            .spatial_steps()
            .unwrap()
            .iter()
            .zip(cold_trajectory.spatial_steps().unwrap())
        {
            assert_eq!(warm.state(), cold.state());
            if force != 0. {
                for (_, _, _, values) in plan
                    .as_spatial()
                    .unwrap()
                    .state_fields(warm.state())
                    .unwrap()
                {
                    assert!(
                        values.iter().fold(0_f64, |norm, value| norm.hypot(*value)) > 1e-6,
                        "unforced Field responds through coupling"
                    );
                }
            }
        }
        assert_eq!(state.to_bytes().unwrap(), before);
        let result = CommonResult::accept_trajectory(0., trajectory).unwrap();
        let bytes = result.to_bytes().unwrap();
        let replayed = CommonResult::from_bytes(&bytes, &plan).unwrap();
        assert_eq!(replayed.to_bytes().unwrap(), bytes);
        let mut corrections = 0;
        for output in [0, 1] {
            let iterations = replayed.nonlinear_iterations(Some(output)).unwrap();
            if force == 0. {
                assert_eq!(iterations, 0);
            } else {
                assert!(iterations > 1);
            }
            corrections += iterations;
            assert!(
                replayed.nonlinear_true_residual_norm(Some(output)).unwrap()
                    <= replayed.nonlinear_residual_target(Some(output)).unwrap()
            );
            for iteration in 0..iterations {
                let solve = replayed
                    .solve_evidence(Some(output), Some(iteration))
                    .unwrap();
                assert_eq!(solve.solver().id(), FAER_SOLVER_PROVIDER.id().as_str());
                assert!(solve.true_residual_norm() <= solve.residual_target());
            }
        }
        if force == 0. {
            for (_, step) in result.trajectory().unwrap().spatial_steps().unwrap() {
                for (_, _, _, values) in plan
                    .as_spatial()
                    .unwrap()
                    .state_fields(step.state())
                    .unwrap()
                {
                    assert!(values.iter().all(|value| value.abs() < 1e-12));
                }
            }
        }
        let observed = backend.observed.lock().unwrap();
        assert_eq!(
            (
                observed.prepares,
                observed.cold,
                observed.drops,
                observed.calls.len()
            ),
            (1, 0, 1, corrections)
        );
        assert_eq!(
            (observed.begins, observed.commits, observed.discards),
            (2, 2, 0)
        );
        for pair in observed.calls.windows(2) {
            assert_eq!(pair[0].1, pair[1].1);
        }
        if force != 0. {
            assert!(
                observed.calls.windows(2).any(|pair| pair[0].2 != pair[1].2),
                "changed Newton Jacobians refresh coefficients"
            );
        }
    }
}

#[test]
fn rejected_nonlinear_correction_discards_action_after_an_accepted_step() {
    let backend = Probe::new(true, Some(4));
    let plan = fixture_source_options(0.125, false, &backend, &coupled_source(3.), true);
    let state = initial(&plan);
    let before = state.to_bytes().unwrap();
    let request =
        CommonTransientRunRequest::from_steps(plan.clone(), state.clone(), 3, vec![1, 2, 3])
            .unwrap();
    let mut boundaries = Vec::new();
    let mut accepted = None;
    let error = request
        .advance_accepted_actions(&backend, |step, state| {
            boundaries.push(step);
            accepted = Some(state.clone());
            false
        })
        .unwrap_err();
    assert!(error.message().contains("residual"), "{error:?}");
    assert_eq!(boundaries, vec![0, 1]);
    assert_eq!(state.to_bytes().unwrap(), before);
    let observed = backend.observed.lock().unwrap();
    assert_eq!(
        (
            observed.prepares,
            observed.begins,
            observed.commits,
            observed.discards,
            observed.drops
        ),
        (1, 2, 1, 1, 1)
    );
    drop(observed);
    let accepted = CommonState::from_bytes(&accepted.unwrap().to_bytes().unwrap(), &plan).unwrap();
    let resumed = run(&plan, accepted.clone(), 2, &backend);
    let cold = Probe::new(false, None);
    let expected = run(&plan, accepted, 2, &cold);
    for ((_, actual), (_, expected)) in resumed
        .spatial_steps()
        .unwrap()
        .iter()
        .zip(expected.spatial_steps().unwrap())
    {
        assert_eq!(actual.state(), expected.state());
    }
    let observed = backend.observed.lock().unwrap();
    assert_eq!(
        (
            observed.prepares,
            observed.commits,
            observed.discards,
            observed.drops
        ),
        (2, 3, 1, 2)
    );
}
