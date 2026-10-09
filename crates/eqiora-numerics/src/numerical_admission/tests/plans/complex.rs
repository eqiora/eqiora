use super::*;
use num_complex::Complex64 as C;

#[test]
fn complex_spatial_plan_retains_types_exact_solver_and_replay() {
    let geometry = cartesian_interval();
    let source = POISSON_INTERVAL
        .replace("variable potential: 1", "variable potential: complex<1>")
        .replace(
            "-div(grad(potential)) - source_scale = 0",
            "-div(grad(potential)) - math.complex(source_scale, 2*source_scale) = 0",
        );
    let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
    let exact = exact_reference_linear(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(64).unwrap(),
    );
    let resolve = |method, request| {
        ResolvedCommonPlan::resolve(
            &model,
            cartesian_box_resources(&geometry, &[4]),
            method,
            CommonSolvePolicy::Linear(request),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
    };
    let plan = resolve(CommonSpatialPolicy::Q1, exact).unwrap();
    let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
    let scalar = plan.as_linear().unwrap();
    assert_eq!(
        scalar.fields().next().unwrap().1.scalar_domain(),
        eqiora_core::ScalarDomain::Complex
    );
    assert_eq!(scalar.solver_provider(), REFERENCE_LINEAR_SOLVER.provider());
    scalar.reauthenticate_portable_realization().unwrap();
    let result = scalar.run_result(&REFERENCE_LINEAR_SOLVER).unwrap();
    assert_eq!(
        result.field_scalar_domain(0),
        Some(eqiora_core::ScalarDomain::Complex)
    );
    let (_, coordinates, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(shape, &[5]);
    assert_eq!(coordinates.len(), 10);
    // -u''=1+2i on [0,1], zero endpoints: nodal Q1 values equal
    // (1+2i)x(1-x)/2, independently of the implementation's matrix.
    for (pair, x) in coordinates
        .as_chunks::<2>()
        .0
        .iter()
        .zip([0., 0.25, 0.5, 0.75, 1.])
    {
        assert!((C::new(pair[0], pair[1]) - C::new(1., 2.) * (x * (1. - x) / 2.)).norm() < 1e-11);
    }
    let bytes = result.to_bytes().unwrap();
    let replayed = crate::CommonResult::from_bytes(&bytes, &plan).unwrap();
    assert_eq!(replayed.to_bytes().unwrap(), bytes);
    assert_eq!(
        replayed.field_block(0, 0).unwrap(),
        result.field_block(0, 0).unwrap()
    );
    let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        wire["content"]["payload"]["fields"][0]["scalar_domain"],
        "complex"
    );
    let mut wrong_domain = wire.clone();
    let field = &mut wrong_domain["content"]["payload"]["fields"][0];
    field["scalar_domain"] = serde_json::json!("real");
    let values = field["blocks"][0]["values"].as_array_mut().unwrap();
    *values = values.iter().step_by(2).cloned().collect();
    assert!(
        crate::CommonResult::from_bytes(&serde_json::to_vec(&wrong_domain).unwrap(), &plan)
            .unwrap_err()
            .message()
            .contains("Result Fields differ from the exact Plan")
    );
    let mut missing_part = wire.clone();
    missing_part["content"]["payload"]["fields"][0]["blocks"][0]["values"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(
        crate::CommonResult::from_bytes(&serde_json::to_vec(&missing_part).unwrap(), &plan)
            .unwrap_err()
            .message()
            .contains("coefficient count differs from its scalar domain")
    );
    let mut retired = wire;
    retired["schema"] = serde_json::json!("eqiora.common-result/v11");
    assert!(
        crate::CommonResult::from_bytes(&serde_json::to_vec(&retired).unwrap(), &plan)
            .unwrap_err()
            .message()
            .contains("unknown schema")
    );
    let mut nonfinite = scalar.run(&REFERENCE_LINEAR_SOLVER).unwrap();
    nonfinite.fields[0].2[1] = f64::NAN;
    assert!(
        crate::CommonResult::accept_linear(scalar.clone(), 0., nonfinite)
            .unwrap_err()
            .message()
            .contains("finite coefficients")
    );
    let mut forged = scalar.clone();
    forged.fields[0].1 = eqiora_core::ValueType::scalar(
        eqiora_core::ScalarDomain::Real,
        forged.fields[0].1.dimension(),
    )
    .unwrap();
    assert!(forged.reauthenticate_portable_realization().is_err());
    assert!(resolve(CommonSpatialPolicy::CellCenteredTpfa, exact).is_err());
    let automatic = CommonLinearRequest::program_controlled(
        1e-12,
        1e-14,
        NonZeroUsize::new(64).unwrap(),
        SolverPlanningObjective::Robust,
    )
    .unwrap();
    assert!(
        resolve(CommonSpatialPolicy::Q1, automatic)
            .unwrap_err()
            .message()
            .contains("exact linear controls")
    );
    let cg = exact_reference_linear(
        LinearSolver::ConjugateGradient,
        1e-12,
        1e-14,
        NonZeroUsize::new(64).unwrap(),
    );
    assert!(resolve(CommonSpatialPolicy::Q1, cg).is_err());
}
