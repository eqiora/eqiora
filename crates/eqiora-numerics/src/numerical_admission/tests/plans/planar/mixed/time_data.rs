//! Continuous initialization and discrete history have distinct rate data.
use super::*;

fn resolve() -> (ResolvedCommonPlan, Vec<Vec<f64>>) {
    let source = FLOW
        .replace("variable u:", "state u:")
        .replace("variable p:", "state p:")
        .replace(
            "{g=1[m/s]*coordinate(0);}",
            "{g=(time()/1[s])^2*1[m/s]*coordinate(0);}",
        )
        .replace(
            "{-div(2[kg/(m*s)]",
            "{1[kg/m^3]*derivative(u)-div(2[kg/(m*s)]",
        )
        .replace(
            "relation constraint on body",
            "initial {u=grad(g); p=0[kg/(m*s^2)];} relation constraint on body",
        );
    let geometry = geometry(true);
    let body = geometry.entity_set("body").unwrap();
    let mut supports = vec![("body", body, None)];
    supports.extend(["left", "right", "bottom", "top"].map(|name| {
        (
            name,
            geometry.entity_set(name).unwrap(),
            Some(("body", body)),
        )
    }));
    let model = compile_model(
        "timed-mini.eqi",
        &source,
        &geometry,
        "Throughflow",
        &supports,
        &[],
    );
    let owner = open_resources(&geometry, false);
    let vertices = owner.simplicial_mesh().unwrap().mesh().vertices().to_vec();
    let plan = ResolvedCommonPlan::resolve(
        &model,
        owner,
        CommonMethodRequest::Exact {
            spatial: CommonSpatialPolicy::MiniP1,
            formulation: FormulationKind::MixedGalerkin,
        },
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            0.,
            1e-13,
            NonZeroUsize::new(1000).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(0.25).unwrap().into()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    (plan, vertices)
}

fn values(plan: &CommonSpatialPlan, vertices: &[Vec<f64>], t: f64, acceleration: f64) -> Vec<f64> {
    plan.fields()
        .flat_map(|(_, ty)| {
            if ty.shape().is_scalar() {
                vertices
                    .iter()
                    .map(|point| acceleration * (1. - point[0]))
                    .collect::<Vec<_>>()
            } else {
                let mut values = [t * t, 0.].repeat(vertices.len());
                values.extend([0.; 10]);
                values
            }
        })
        .collect()
}

#[test]
fn timed_mini_initialization_requires_the_actual_prescribed_trace_rate() {
    let (resolved, vertices) = resolve();
    let plan = resolved.as_spatial().unwrap();
    plan.initial_state(Some(0.), Vec::new()).unwrap();
    // Uniform u=t² has u_dot=2t, zero strain and divergence. Momentum and
    // zero outlet traction therefore require p=2t(1-x), exactly in MINI/P1.
    let consistent = plan.scalar_state(Some(1.), values(plan, &vertices, 1., 2.));
    let inconsistent = plan.scalar_state(Some(1.), values(plan, &vertices, 1., 0.));
    assert!(
        consistent.is_ok() && inconsistent.is_err(),
        "continuous pair error: {:?}; omitted boundary rate accepted: {}",
        consistent.as_ref().err().map(Diagnostic::message),
        inconsistent.is_ok()
    );
}

#[test]
fn timed_mini_accepted_step_preserves_its_discrete_rate_and_replays() {
    let (resolved, vertices) = resolve();
    let plan = resolved.as_spatial().unwrap();
    let initial = plan.initial_state(Some(0.), Vec::new()).unwrap();
    // The first BE step has endpoint secant (h²-0)/h=h, so its exact
    // discrete solution is u=h², p=h(1-x), not continuous p=2h(1-x).
    // Acceptance/replay must check the step it represents; imposing the
    // continuous initialization tangent here would reject a correct BE step.
    let mut state = initial;
    for index in 1..=3 {
        let t = index as f64 * 0.25;
        let mut step = plan
            .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let expected = values(plan, &vertices, t, (2 * index - 1) as f64 * 0.25);
        for (&actual, expected) in step.state().spatial_values().unwrap().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
        step.reaccept_spatial(plan).unwrap();
        let bytes = step.state().to_bytes().unwrap();
        let replayed = CommonState::from_bytes(&bytes, &resolved).unwrap();
        assert_eq!(replayed, *step.state());
        let CommonStateKind::Spatial {
            previous: Some(previous),
            ..
        } = &replayed.kind
        else {
            panic!("accepted State omitted its predecessor");
        };
        // Continuous pressure belongs to a different equation at this endpoint.
        assert!(
            plan.spatial_state(
                Some(t),
                values(plan, &vertices, t, 2. * t),
                Some(previous.clone())
            )
            .is_err()
        );
        let mut changed = previous.clone();
        changed.grid.origin_s += 0.125;
        assert!(
            plan.spatial_state(
                Some(t),
                replayed.spatial_values().unwrap().to_vec(),
                Some(changed)
            )
            .is_err()
        );
        assert!(
            plan.spatial_state(
                Some(t + 0.125),
                replayed.spatial_values().unwrap().to_vec(),
                Some(previous.clone())
            )
            .is_err()
        );
        // Each following solve starts from the deserialized checkpoint.
        state = replayed;
    }
}
