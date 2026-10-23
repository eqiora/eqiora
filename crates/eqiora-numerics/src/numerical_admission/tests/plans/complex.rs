use super::*;
use num_complex::Complex64 as C;

mod cartesian;
mod convergence;

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
            None,
        )
    };
    let plan = resolve(CommonSpatialPolicy::Q1, exact).unwrap();
    let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
    let scalar = plan.as_spatial().unwrap();
    assert_eq!(
        scalar.fields().next().unwrap().1.scalar_domain(),
        eqiora_core::ScalarDomain::Complex
    );
    assert_eq!(scalar.solver_provider(), REFERENCE_LINEAR_SOLVER.provider());
    scalar.reauthenticate_portable_realization().unwrap();
    crate::region_assembly::InterfaceReactions::<f64>::reset_evaluations();
    let result = scalar.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
    assert_eq!(
        crate::region_assembly::InterfaceReactions::<f64>::evaluations(),
        0
    );
    assert_eq!(
        result.field(0).map(|(_, ty, _)| ty.scalar_domain()),
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
    // Type metadata belongs only to the Plan-bound Model. A Result cannot
    // override its domain, units, or shape, even with plausible old fields.
    for (key, value) in [
        ("scalar_domain", serde_json::json!("real")),
        ("dimension", serde_json::json!(vec![[0, 1]; 7])),
        ("value_shape", serde_json::json!([])),
    ] {
        assert!(wire["content"]["payload"]["fields"][0].get(key).is_none());
        let mut overridden = wire.clone();
        overridden["content"]["payload"]["fields"][0][key] = value;
        assert!(
            crate::CommonResult::from_bytes(&serde_json::to_vec(&overridden).unwrap(), &plan)
                .unwrap_err()
                .message()
                .contains("unknown field")
        );
    }
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
    retired["schema"] = serde_json::json!("eqiora.common-result/v23");
    assert!(
        crate::CommonResult::from_bytes(&serde_json::to_vec(&retired).unwrap(), &plan)
            .unwrap_err()
            .message()
            .contains("unknown schema")
    );
    let mut nonfinite = scalar.run(None, &REFERENCE_LINEAR_SOLVER).unwrap();
    nonfinite.fields[0].2[1] = f64::NAN;
    assert!(
        crate::CommonResult::accept_linear(scalar.clone(), None, 0., nonfinite)
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
    let tpfa = resolve(CommonSpatialPolicy::CellCenteredTpfa, exact).unwrap();
    let tpfa = replay_plan(tpfa, &REFERENCE_LINEAR_SOLVER);
    let tpfa_result = tpfa
        .as_spatial()
        .unwrap()
        .run_result(None, &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (association, coefficients, shape) = tpfa_result.field_block(0, 0).unwrap();
    assert_eq!(association, "cell");
    assert_eq!(shape, &[4]);
    for (pair, expected) in coefficients.as_chunks::<2>().0.iter().zip([1., 2., 2., 1.]) {
        assert!((C::new(pair[0], pair[1]) - C::new(1., 2.) * (expected / 16.)).norm() < 1e-11);
    }
    assert_eq!(
        crate::CommonResult::from_bytes(&tpfa_result.to_bytes().unwrap(), &tpfa).unwrap(),
        tpfa_result
    );
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

#[test]
fn real_and_complex_reaction_diffusion_share_spatial_execution_and_replay() {
    let geometry = cartesian_interval();
    for complex in [false, true] {
        for natural in [false, true] {
            let (balance, left, right) = if complex {
                (
                    "-div(math.complex(2, 3)*grad(potential)) + math.complex(3, -2)*source_scale*potential - math.complex(5 + 4*coordinate(0)/1[m], 1 - 7*coordinate(0)/1[m])*source_scale = 0",
                    "trace(potential) = math.complex(1, 1)",
                    if natural {
                        "normal(math.complex(2, 3)*grad(potential)) = math.complex(7[1/m], 4[1/m])"
                    } else {
                        "trace(potential) = math.complex(3, 0)"
                    },
                )
            } else {
                (
                    "-div(2*grad(potential)) + 3*source_scale*potential - (3 + 6*coordinate(0)/1[m])*source_scale = 0",
                    "trace(potential) = 1",
                    if natural {
                        "normal(2*grad(potential)) = 4[1/m]"
                    } else {
                        "trace(potential) = 3"
                    },
                )
            };
            let mut source = POISSON_INTERVAL
                .replace(" on body in h1;", " on body in smooth;")
                .replace("-div(grad(potential)) - source_scale = 0", balance)
                .replace(
                    "relation left_value on left { trace(potential) = 0; }",
                    &format!("relation left_value on left {{ {left}; }}"),
                )
                .replace(
                    "relation right_value on right { trace(potential) = 0; }",
                    &format!("relation right_value on right {{ {right}; }}"),
                );
            if complex {
                source = source.replace("variable potential: 1", "variable potential: complex<1>");
            }
            let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
            for method in [
                CommonSpatialPolicy::Q1,
                CommonSpatialPolicy::CellCenteredTpfa,
            ] {
                let plan = ResolvedCommonPlan::resolve(
                    &model,
                    cartesian_box_resources(&geometry, &[8]),
                    method,
                    CommonSolvePolicy::Linear(exact_reference_linear(
                        LinearSolver::BiConjugateGradientStabilized,
                        1e-12,
                        1e-14,
                        NonZeroUsize::new(128).unwrap(),
                    )),
                    None,
                    None,
                    &REFERENCE_LINEAR_SOLVER,
                    None,
                    None,
                )
                .unwrap();
                let plan = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
                assert_eq!(
                    plan.operator_properties(),
                    Some(LinearOperatorProperties::General)
                );
                let result = plan
                    .as_spatial()
                    .unwrap()
                    .run_result(None, &REFERENCE_LINEAR_SOLVER)
                    .unwrap();
                let domain = if complex {
                    eqiora_core::ScalarDomain::Complex
                } else {
                    eqiora_core::ScalarDomain::Real
                };
                assert_eq!(
                    result.field(0).map(|(_, ty, _)| ty.scalar_domain()),
                    Some(domain)
                );
                let (association, values, shape) = result.field_block(0, 0).unwrap();
                let nodal = method == CommonSpatialPolicy::Q1;
                assert_eq!(association, if nodal { "vertex" } else { "cell" });
                assert_eq!(shape, &[if nodal { 9 } else { 8 }]);
                let width = if complex { 2 } else { 1 };
                // Independent affine solutions u=1+2x and u=1+2x+i(1-x).
                // For a=2+3i, q=3-2i: q*u=(5+4x)+i(1-7x), a*u'=7+4i.
                for (index, value) in values.chunks_exact(width).enumerate() {
                    let x = (index as f64 + if nodal { 0. } else { 0.5 }) / 8.;
                    assert!((value[0] - (1. + 2. * x)).abs() < 1e-10);
                    if complex {
                        assert!((value[1] - (1. - x)).abs() < 1e-10);
                    }
                }
                let bytes = result.to_bytes().unwrap();
                assert_eq!(
                    crate::CommonResult::from_bytes(&bytes, &plan).unwrap(),
                    result
                );
                if complex {
                    for transform in [0, 1, 2] {
                        let mut changed = values.to_vec();
                        for pair in changed.as_chunks_mut::<2>().0 {
                            let (re, im) = (pair[0], pair[1]);
                            match transform {
                                0 => pair[1] = -im,
                                1 => {
                                    pair[0] = -im;
                                    pair[1] = re;
                                }
                                _ => {
                                    pair[0] = -re;
                                    pair[1] = -im;
                                }
                            }
                        }
                        let resealed = super::result_replay::reseal_values(&result, 0, &changed);
                        let error = crate::CommonResult::from_bytes(&resealed, &plan).unwrap_err();
                        assert!(
                            error.message().contains("residual")
                                || error.message().contains("prescribed"),
                            "{error:?}"
                        );
                    }
                }
            }
        }
    }
}
