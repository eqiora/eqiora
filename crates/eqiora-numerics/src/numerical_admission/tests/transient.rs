use super::*;

fn newton_policy(linear: CommonLinearRequest, nonlinear: NonlinearSolvePlan) -> CommonSolvePolicy {
    CommonSolvePolicy::Newton { nonlinear, linear }
}

fn assert_solver_structure(
    plan: &CommonTransientFlowPlan,
    backend: &dyn LinearSolverBackend,
    with_gauge: bool,
) {
    use eqiora_solver::{AlgebraicConstraint, AlgebraicStructure, HostSerialSolverProfile};
    let (velocity, pressure) = match plan.admission.recognized_model() {
        RecognizedNativeModel::Transient(model) => (model.velocity(), model.pressure()),
        RecognizedNativeModel::TransientGeometry(binding) => {
            (binding.velocity(), binding.pressure())
        }
        _ => panic!("transient fixture"),
    };
    let velocity = velocity.downcast().unwrap();
    let pressure = pressure.downcast().unwrap();
    let gauge = AlgebraicConstraint::ZeroIntegral { field: pressure };
    let constraints: Vec<_> = with_gauge.then_some(gauge).into_iter().collect();
    let expected = AlgebraicStructure::new([velocity, pressure], constraints.clone()).unwrap();
    plan.admission
        .linear
        .planning_profile
        .as_ref()
        .unwrap()
        .require_structure(Some(&expected))
        .unwrap();
    let foreign = eqiora_core::Id::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap());
    let state = if with_gauge {
        plan.zero_state(0.0).unwrap()
    } else {
        let NativeMeshResources::AffineTriangleSimplicial { mesh, .. } = plan.admission.resources()
        else {
            panic!("traction fixture owns affine triangles");
        };
        let vertices = mesh.mesh().vertices().len();
        let cells = mesh.mesh().entity_count(2).unwrap();
        let digest = plan.admission.model().digest().unwrap();
        plan.initial_state(
            0.0,
            vec![
                CommonInitialField::new(
                    digest.clone(),
                    velocity,
                    Some(CommonInitialValues::Vector2(
                        vec![[0.0; 2]; vertices].into_boxed_slice(),
                    )),
                    Some(CommonInitialValues::Vector2(
                        vec![[0.0; 2]; cells].into_boxed_slice(),
                    )),
                )
                .unwrap(),
                CommonInitialField::new(
                    digest,
                    pressure,
                    Some(CommonInitialValues::Scalar(
                        vec![0.0; vertices].into_boxed_slice(),
                    )),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    };
    for structure in [
        None,
        Some(AlgebraicStructure::new([velocity], []).unwrap()),
        Some(AlgebraicStructure::new([pressure], constraints.clone()).unwrap()),
        Some(AlgebraicStructure::new([foreign, pressure], constraints.clone()).unwrap()),
        Some(AlgebraicStructure::new([velocity, pressure, foreign], constraints).unwrap()),
        Some(
            AlgebraicStructure::new(
                [velocity, pressure],
                [AlgebraicConstraint::ZeroIntegral { field: velocity }],
            )
            .unwrap(),
        ),
        Some(
            AlgebraicStructure::new([velocity, pressure], (!with_gauge).then_some(gauge)).unwrap(),
        ),
    ] {
        let mut changed = plan.clone();
        let profile =
            HostSerialSolverProfile::canonical_csr(LinearOperatorProperties::General, None, None);
        changed.admission.linear.planning_profile = Some(match structure {
            Some(structure) => profile.with_structure(structure).unwrap(),
            None => profile,
        });
        let error = changed
            .prepare_execution(&state, backend)
            .err()
            .expect("stale profile reached preparation");
        assert!(error.message().contains("structure"), "{error:?}");
    }
    let mut omitted = plan.clone();
    omitted.admission.linear.planning_profile = None;
    assert!(omitted.prepare_execution(&state, backend).is_err());
}

#[test]
fn prepared_transient_methods_keep_authoritative_common_grid_time_bits() {
    let geometry = rectangle();
    let model = transient_model();
    let step_s = 0.0001_f64;
    let temporal = CommonBackwardEuler::from_seconds(step_s).unwrap();
    let scaling =
        IncompressibleScalingRequest2d::from_si(Some(0.41), Some(0.3), Some(0.09)).unwrap();
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(16).unwrap(), 12).unwrap();
    let resolve = |owner, spatial| {
        let linear = if spatial == CommonSpatialPolicy::MiniP1 {
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::SparseLu,
                    1e-10,
                    1e-12,
                    NonZeroUsize::new(2_000).unwrap(),
                )
                .unwrap()
                .with_preconditioner(PreconditionerPolicy::Identity)
                .with_reduction(ReductionPolicy::Fast),
                ResolveOnlyBackend.provider(),
            )
            .unwrap()
        } else {
            exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-10,
                1e-12,
                NonZeroUsize::new(2_000).unwrap(),
            )
        };
        resolve_common_plan(
            &model,
            owner,
            spatial,
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .unwrap()
        .as_transient_flow()
        .cloned()
        .expect("fixture retains its admitted transient_flow Plan")
    };

    let mini = resolve(affine_resources(&geometry), CommonSpatialPolicy::MiniP1);
    let cell_resources = {
        let cells = CartesianMeshCellsV2::new([3, 4]).unwrap();
        let (mesh, correspondence) =
            GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
                &geometry,
                cells.cells().try_into().unwrap(),
            )
            .unwrap();
        let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
            &cells,
            &geometry,
            &mesh,
            &correspondence,
        )
        .unwrap();
        AuthenticatedCommonMesh::structured_cartesian(
            geometry.clone(),
            mesh,
            correspondence,
            production,
        )
        .unwrap()
    };
    let cell_centered = resolve(cell_resources, CommonSpatialPolicy::CellCentered);
    for (plan, backend) in [
        (&mini, &ResolveOnlyBackend as &dyn LinearSolverBackend),
        (&cell_centered, &REFERENCE_LINEAR_SOLVER),
    ] {
        let previous = plan.zero_state(0.0).unwrap();
        let accepted = plan.advance_one(&previous, backend).unwrap();
        assert_eq!(accepted.time_s().to_bits(), step_s.to_bits());
        assert!(Arc::ptr_eq(&accepted.model, &previous.model));
        assert!(Arc::ptr_eq(&accepted.resources, &previous.resources));
    }
}

#[test]
pub(super) fn transient_common_plan_resolves_exact_mini_and_supplied_cartesian_resources() {
    let geometry = rectangle();
    let model = transient_model();
    let replayed = ModelEnvelope::from_json(
        &model.canonical_json().unwrap(),
        ModelDecoderLimits::default(),
    )
    .unwrap();
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::SparseLu,
            1e-10,
            1e-12,
            NonZeroUsize::new(2_000).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast),
        ResolveOnlyBackend.provider(),
    )
    .unwrap();
    let temporal = CommonBackwardEuler::from_seconds(0.01).unwrap();
    let nonlinear =
        NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(16).unwrap(), 12).unwrap();
    let scaling = IncompressibleScalingRequest2d::from_si(Some(1.0), Some(2.0), Some(3.0)).unwrap();
    let resolve = |model: &ModelEnvelope, owner, spatial, formulation| {
        let linear = if spatial == CommonSpatialPolicy::MiniP1 {
            CommonLinearRequest::exact(
                SolverPlan::new(
                    LinearSolver::SparseLu,
                    1e-10,
                    1e-12,
                    NonZeroUsize::new(2_000).unwrap(),
                )
                .unwrap()
                .with_preconditioner(PreconditionerPolicy::Identity)
                .with_reduction(ReductionPolicy::Fast),
                ResolveOnlyBackend.provider(),
            )
            .unwrap()
        } else {
            exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-10,
                1e-12,
                NonZeroUsize::new(2_000).unwrap(),
            )
        };
        let method = match formulation {
            None => CommonMethodRequest::Uniform(spatial),
            Some(formulation) => CommonMethodRequest::Exact {
                spatial,
                formulation,
            },
        };
        let resolved = resolve_common_plan(
            model,
            owner,
            method,
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .unwrap();
        replay_plan(resolved, &ResolveOnlyBackend)
            .as_transient_flow()
            .cloned()
            .expect("fixture retains its admitted transient_flow Plan")
    };
    let mini = resolve(
        &model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        None,
    );
    let mini_replay = resolve(
        &replayed,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        None,
    );
    let fvm = resolve(
        &model,
        resources(&geometry),
        CommonSpatialPolicy::CellCentered,
        None,
    );
    let resolve_program_controlled = |objective| {
        let linear = CommonLinearRequest::program_controlled(
            1.0e-10,
            1.0e-12,
            NonZeroUsize::new(2_000).unwrap(),
            objective,
        )
        .unwrap();
        resolve_common_plan(
            &model,
            resources(&geometry),
            CommonSpatialPolicy::CellCentered,
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &PlanningFaerBackend,
            None,
        )
        .unwrap()
        .as_transient_flow()
        .cloned()
        .expect("fixture retains its admitted transient_flow Plan")
    };
    let robust = resolve_program_controlled(SolverPlanningObjective::Robust);
    let fast = resolve_program_controlled(SolverPlanningObjective::Fast);
    let low_memory = resolve_program_controlled(SolverPlanningObjective::LowMemory);
    assert_solver_structure(&mini, &ResolveOnlyBackend, true);
    assert_solver_structure(&fvm, &REFERENCE_LINEAR_SOLVER, true);
    for planned in [&robust, &fast, &low_memory] {
        assert_solver_structure(planned, &PlanningFaerBackend, true);
    }
    // Replacing one essential boundary with zero traction removes the pressure
    // nullspace mathematically; the solver profile must lose exactly that gauge.
    let source = TRANSIENT_SOURCE.replace(
        "relation y_upper_value on y_upper { trace(velocity) = 0; }",
        "relation y_upper_value on y_upper { normal(2 * dynamic_viscosity * symmetric_part(grad(velocity)) - isotropic_lift(pressure)) = 0; }",
    );
    let compiled = eqiora_compiler::compile("mixed-transient.eqi", &source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model_id, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
    let traction_model = ModelEnvelope::from_program(&program).unwrap();
    let traction = resolve(
        &traction_model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        None,
    );
    assert_solver_structure(&traction, &ResolveOnlyBackend, false);

    let mini_exact = resolve(
        &model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        Some(FormulationKind::MixedGalerkin),
    );
    let fvm_exact = resolve(
        &model,
        resources(&geometry),
        CommonSpatialPolicy::CellCentered,
        Some(FormulationKind::IntegralConservative),
    );
    let custom_nonlinear =
        NonlinearSolvePlan::new(2.0e-9, 3.0e-11, NonZeroUsize::new(19).unwrap(), 7).unwrap();
    let custom = resolve_common_plan(
        &model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        newton_policy(linear, custom_nonlinear),
        Some(scaling),
        Some(temporal),
        &ResolveOnlyBackend,
        None,
    )
    .unwrap()
    .as_transient_flow()
    .cloned()
    .expect("fixture retains its admitted transient_flow Plan");
    let alternate_scaling =
        IncompressibleScalingRequest2d::from_si(Some(4.0), Some(5.0), Some(6.0)).unwrap();
    let fvm_alternate_scaling = resolve_common_plan(
        &model,
        resources(&geometry),
        CommonSpatialPolicy::CellCentered,
        newton_policy(
            exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-10,
                1e-12,
                NonZeroUsize::new(2_000).unwrap(),
            ),
            nonlinear,
        ),
        Some(alternate_scaling),
        Some(temporal),
        &ResolveOnlyBackend,
        None,
    )
    .unwrap()
    .as_transient_flow()
    .cloned()
    .expect("fixture retains its admitted transient_flow Plan");

    assert_eq!(mini.identity(), mini_replay.identity());
    assert_eq!(mini.realization_digest(), mini_replay.realization_digest());
    assert_ne!(mini.realization_digest(), fvm.realization_digest());
    assert_eq!(
        mini.realization_digest(),
        hex_bytes(&mini.portable_realization().digest().unwrap())
    );
    assert_ne!(mini.identity(), mini_exact.identity());
    assert_ne!(fvm.identity(), fvm_exact.identity());
    assert_ne!(robust.identity(), fast.identity());
    assert_ne!(fast.identity(), low_memory.identity());
    assert_eq!(
        robust.selected_solver_candidate_id(),
        Some("eqiora.faer.sparse-lu-general-identity-fast-f64")
    );
    assert_eq!(
        fast.selected_solver_candidate_id(),
        Some("eqiora.faer.sparse-lu-general-identity-fast-f64")
    );
    assert_eq!(
        low_memory.selected_solver_candidate_id(),
        Some("eqiora.faer.sparse-lu-general-identity-fast-f64")
    );
    for plan in [&robust, &fast, &low_memory] {
        assert_eq!(
            plan.solver_planning_policy_id(),
            Some("eqiora.host-serial-solver-planning/v2")
        );
        assert_eq!(plan.solver_planning_reasons().len(), 4);
        assert!(plan.selected_solver_evidence_case().is_some());
    }
    let planned_mini = resolve_common_plan(
        &model,
        affine_resources(&geometry),
        CommonSpatialPolicy::MiniP1,
        newton_policy(
            CommonLinearRequest::program_controlled(
                1.0e-10,
                1.0e-12,
                NonZeroUsize::new(2_000).unwrap(),
                SolverPlanningObjective::Robust,
            )
            .unwrap(),
            nonlinear,
        ),
        Some(scaling),
        Some(temporal),
        &PlanningFaerBackend,
        None,
    )
    .unwrap();
    assert_eq!(
        planned_mini.selected_solver_candidate_id(),
        Some("eqiora.faer.sparse-lu-general-identity-fast-f64")
    );
    assert_eq!(
        mini.formulation().effective(),
        mini_exact.formulation().effective()
    );
    assert_eq!(
        fvm.formulation().effective(),
        fvm_exact.formulation().effective()
    );
    assert_eq!(
        mini_exact.formulation().requested(),
        FormulationSelectionMode::Exact
    );
    assert_eq!(
        fvm_exact.formulation().requested(),
        FormulationSelectionMode::Exact
    );
    assert_eq!(
        mini.state_space_identity(),
        mini_exact.state_space_identity()
    );
    assert_eq!(fvm.state_space_identity(), fvm_exact.state_space_identity());
    assert_eq!(mini.realization_digest(), mini_exact.realization_digest());
    assert_eq!(fvm.realization_digest(), fvm_exact.realization_digest());
    assert_ne!(
        fvm.realization_digest(),
        fvm_alternate_scaling.realization_digest()
    );
    assert_eq!(
        mini_exact.formulation().selection_reason_codes(),
        &["eqiora.formulation.exact.mixed-galerkin-admitted/v1"]
    );
    assert_eq!(
        fvm_exact.formulation().selection_reason_codes(),
        &["eqiora.formulation.exact.integral-conservative-admitted/v1"]
    );
    assert_ne!(mini.identity(), fvm.identity());
    assert_ne!(mini.identity(), custom.identity());
    assert_eq!(custom.nonlinear(), custom_nonlinear);
    assert_eq!(mini.model_digest(), model.digest().unwrap().to_string());
    assert_eq!(mini.velocity_field_id(), fvm.velocity_field_id());
    assert_eq!(mini.pressure_field_id(), fvm.pressure_field_id());
    assert_eq!(mini.velocity_space().family(), SpaceFamily::SimplexP1Bubble);
    assert!(
        matches!(mini.pressure_space().family(), SpaceFamily::ContinuousLagrange { order } if order.get() == 1)
    );
    assert_eq!(fvm.velocity_space().family(), SpaceFamily::CellConstant);
    assert_eq!(fvm.pressure_space().family(), SpaceFamily::CellConstant);
    assert_eq!(mini.temporal().step().value().to_bits(), 0.01_f64.to_bits());
    assert_eq!(mini.scales().length().value().to_bits(), 1.0_f64.to_bits());
    assert_eq!(
        mini.scales().velocity().value().to_bits(),
        2.0_f64.to_bits()
    );
    assert_eq!(
        mini.scales().pressure().value().to_bits(),
        3.0_f64.to_bits()
    );
    assert_eq!(mini.linear().algorithm(), LinearSolver::SparseLu);
    assert_eq!(mini.linear().reduction(), ReductionPolicy::Fast);
    assert_eq!(fvm.linear().reduction(), ReductionPolicy::Reproducible);
    let mini_formulation = mini.formulation();
    assert_eq!(
        mini_formulation.requested(),
        FormulationSelectionMode::Automatic
    );
    assert_eq!(mini_formulation.effective(), FormulationKind::MixedGalerkin);
    assert_eq!(
        mini_formulation.boundary_treatment(),
        "explicit-trace-flux-laws"
    );
    assert_eq!(mini_formulation.rule_ids().len(), 6);
    assert_eq!(mini_formulation.selection_reason_codes().len(), 1);
    let fvm_formulation = fvm.formulation();
    assert_eq!(
        fvm_formulation.effective(),
        FormulationKind::IntegralConservative
    );
    assert_eq!(fvm_formulation.rule_ids().len(), 7);
    assert_ne!(mini_formulation, fvm_formulation);

    let mini_zero = mini.zero_state(0.0).unwrap();
    let fvm_zero = fvm.zero_state(0.0).unwrap();
    let mini_bytes = mini_zero.to_bytes().unwrap();
    assert_eq!(
        CommonState::from_bytes(
            &mini_bytes,
            &ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
        )
        .unwrap(),
        mini_zero
    );
    let fvm_bytes = fvm_zero.to_bytes().unwrap();
    assert_eq!(
        CommonState::from_bytes(
            &fvm_bytes,
            &ResolvedCommonPlan::TransientFlow(Box::new(fvm.clone())),
        )
        .unwrap(),
        fvm_zero
    );
    let mut noncanonical = mini_bytes;
    noncanonical.push(b'\n');
    assert!(
        CommonState::from_bytes(
            &noncanonical,
            &ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
        )
        .is_err()
    );
    assert!(
        CommonState::from_bytes(
            &fvm_bytes,
            &ResolvedCommonPlan::TransientFlow(Box::new(mini.clone())),
        )
        .is_err()
    );
    assert_eq!(mini_zero.velocity_vertex_values().unwrap().len(), 12);
    assert_eq!(mini_zero.velocity_cell_values().len(), 12);
    assert_eq!(mini_zero.pressure_vertex_values().unwrap().len(), 12);
    assert!(mini_zero.method_history_values().is_empty());
    assert_eq!(fvm_zero.velocity_cell_values().len(), 6);
    assert_eq!(fvm_zero.pressure_cell_values().unwrap().len(), 6);
    assert!(!fvm_zero.method_history_values().is_empty());
    let curl = mini.cell_average_velocity_curl_2d(&mini_zero).unwrap();
    assert_eq!(curl.as_ref(), &[0.0; 12]);
    assert_eq!(
        curl,
        custom.cell_average_velocity_curl_2d(&mini_zero).unwrap(),
        "derived values exclude solve policy when the exact State and field are unchanged",
    );
    assert!(fvm.cell_average_velocity_curl_2d(&fvm_zero).is_err());
    assert_eq!(custom.state_space_identity(), mini.state_space_identity());
    assert_eq!(
        fvm_alternate_scaling.state_space_identity(),
        fvm.state_space_identity(),
        "coherent-SI State compatibility excludes numerical scaling",
    );
    assert!(
        CommonTransientRunRequest::from_steps(custom.clone(), mini_zero.clone(), 2, vec![1, 2],)
            .is_ok()
    );
    assert!(CommonTransientRunRequest::from_steps(mini.clone(), fvm_zero, 1, vec![1],).is_err());
    let by_steps =
        CommonTransientRunRequest::from_steps(mini.clone(), mini_zero.clone(), 2, vec![1, 2])
            .unwrap();
    let by_times =
        CommonTransientRunRequest::from_times(mini.clone(), mini_zero, 0.02, vec![0.01, 0.02])
            .unwrap();
    assert_eq!(by_steps.identity(), by_times.identity());
    assert!(
        CommonTransientRunRequest::from_steps(mini, by_times.state().clone(), 2, vec![2, 1],)
            .is_err()
    );

    assert!(
        resolve_common_plan(
            &model,
            affine_resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            CommonSolvePolicy::Linear(linear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        resolve_common_plan(
            &model,
            affine_resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            newton_policy(linear, nonlinear),
            Some(IncompressibleScalingRequest2d::from_si(Some(1.0), None, Some(3.0)).unwrap()),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        resolve_common_plan(
            &model,
            affine_resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            newton_policy(linear, nonlinear),
            Some(scaling),
            None,
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        resolve_common_plan(
            &model,
            resources(&geometry),
            CommonSpatialPolicy::MiniP1,
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        resolve_common_plan(
            &model,
            affine_resources(&geometry),
            CommonMethodRequest::Exact {
                spatial: CommonSpatialPolicy::MiniP1,
                formulation: FormulationKind::IntegralConservative,
            },
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
    assert!(
        resolve_common_plan(
            &model,
            resources(&geometry),
            CommonMethodRequest::Exact {
                spatial: CommonSpatialPolicy::CellCentered,
                formulation: FormulationKind::MixedGalerkin,
            },
            newton_policy(linear, nonlinear),
            Some(scaling),
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
}
