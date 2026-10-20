//! Stationary Newton acceptance has a physical predictor without time history.
use super::*;
use crate::form_compiler::region::BoundRegionForm;
use eqiora_realization::NonlinearSolvePlan;
use eqiora_solver::LinearSolveRequest;

pub(super) fn check_stationary(
    mesh: &CartesianMesh,
    mapping: &RegionDofMap<f64>,
    forms: &[(BoundRegionForm<f64>, QuadratureRule)],
    length: usize,
) {
    let input = || RegionSolveInput {
        operator_properties: LinearOperatorProperties::General,
        forms: forms.to_vec(),
        natural: vec![],
        previous: None,
        prescribed_states: BTreeMap::new(),
        geometry_action: None,
    };
    let inventory = mapping.fields.keys().copied().collect::<Vec<_>>();
    let initial = mapping
        .recover(&vec![0.; mapping.free_count()], &inventory)
        .unwrap();
    let nonlinear = NonlinearSolvePlan::new(0., 1e-11, NonZeroUsize::new(4).unwrap(), 4).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-14,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let execute = |seed| {
        mapping
            .solve_nonlinear(
                mesh,
                &input(),
                seed,
                nonlinear,
                NonZeroUsize::MIN,
                LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver),
            )
            .unwrap()
    };
    let output = execute(&initial);
    assert_eq!(output.solve.evidence.iterations, 1);
    // Independently: -2 u'' = 0, u(0)=0, u(L)=1 gives u=x/L.
    // The fixture uses Field scale 2, so confusing physical and normalized
    // coefficients changes both this solution and the replay residual.
    for field in output.fields.values() {
        for (dof, value) in &field.coefficients {
            let exact = dof.entity.index() as f64 / (2 * length) as f64;
            assert!((value - exact).abs() < 1e-11);
        }
    }
    let validate = |fields, seed, iterations| {
        mapping.validate_nonlinear_coefficients(mesh, &input(), fields, seed, nonlinear, iterations)
    };
    let (norm, target, packets, targets, initial_norm) =
        validate(&output.fields, &initial, 1).unwrap();
    assert!(norm <= target);
    assert_eq!(
        initial_norm,
        Some(output.solve.evidence.initial_residual_norm)
    );
    assert_eq!(packets, output.solve.assembly.report.packet_count());
    assert_eq!(targets, output.solve.assembly.report.target_count());
    assert!(validate(&initial, &initial, 1).is_err());
    assert!(validate(&output.fields, &initial, 0).is_err());
    assert!(validate(&output.fields, &initial, 5).is_err());

    let accepted_seed = execute(&output.fields);
    assert_eq!(accepted_seed.solve.evidence.iterations, 0);
    assert!(accepted_seed.solve.evidence.linear_solves.is_empty());
    assert_eq!(accepted_seed.fields, output.fields);
    validate(&output.fields, &output.fields, 0).unwrap();
    assert!(validate(&output.fields, &output.fields, 1).is_err());

    let mut corrupt = output.fields.clone();
    let prescribed = mapping
        .keys()
        .find(|key| mapping.free_dof(*key).is_none())
        .unwrap();
    *corrupt
        .get_mut(&prescribed.field)
        .unwrap()
        .coefficients
        .get_mut(&prescribed)
        .unwrap() += 1.;
    assert!(validate(&corrupt, &initial, 1).is_err());
}
