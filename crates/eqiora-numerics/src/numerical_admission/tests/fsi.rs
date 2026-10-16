use super::*;

mod nonlinear;
mod shared;
mod tetrahedral;

#[test]
pub(super) fn common_fsi_resolves_exact_scopes_initializes_and_restarts_without_pressure_gauge() {
    use eqiora_core::entity::kinds::Field;

    let geometry = fsi_geometry();
    let model = fsi_model(&geometry);
    let resources = fsi_resources(&geometry);
    let recognized = RecognizedNativeAdmission::recognize(
        &model,
        resources.clone(),
        &CommonSpatialRequest::Uniform(CommonSpatialPolicy::P1),
    )
    .unwrap();
    let RecognizedNativeModel::Spatial(equations) = &recognized.recognized else {
        panic!("coupled equations must use the common linear Plan")
    };
    let fluid = equations
        .regions
        .iter()
        .find(|region| region.form.constraint_pairs().next().is_some())
        .unwrap();
    let solid = equations
        .regions
        .iter()
        .find(|region| !region.form.kinematics().is_empty())
        .unwrap();
    let (velocity, pressure) = fluid.form.constraint_pairs().next().unwrap();
    let pair = solid.form.kinematics()[0].0;
    let fluid_domain = fluid.domain_id();
    let solid_domain = solid.domain_id();
    let field_ids = [
        velocity.downcast::<Field>().unwrap(),
        pressure.downcast::<Field>().unwrap(),
        pair.rate(),
        pair.state(),
    ];
    let digest = model.digest().unwrap();
    let scoped = CommonMethodRequest::Scoped(vec![
        CommonScopedSpatialPolicy::new(digest.clone(), fluid_domain, CommonSpatialPolicy::MiniP1),
        CommonScopedSpatialPolicy::new(digest.clone(), solid_domain, CommonSpatialPolicy::P1),
    ]);
    let requested = exact_reference_linear(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(20_000).unwrap(),
    );
    let temporal = CommonBackwardEuler::from_seconds(0.05).unwrap();
    let resolve = |scaling: Option<CommonScalingRequest>| {
        let resolved = ResolvedCommonPlan::resolve(
            &model,
            resources.clone(),
            scoped.clone(),
            CommonSolvePolicy::Linear(requested),
            scaling,
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .unwrap();
        replay_plan(resolved, &ResolveOnlyBackend)
    };
    let automatic_resolved = resolve(None);
    let automatic = automatic_resolved.as_spatial().unwrap();
    let scaling = eqiora_realization::SymmetricCongruenceScaling::new(
        equations.fields().iter().map(|(field, value_type)| {
            eqiora_realization::AlgebraicBlockScale::new(
                eqiora_solver::AlgebraicBlock::Field(field.downcast().unwrap()),
                eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(
                    2.,
                    value_type.dimension(),
                ))
                .unwrap(),
            )
        }),
        eqiora_realization::PositivePhysicalScale::new(DynQuantity::new(
            3.,
            DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).unwrap(),
        ))
        .unwrap(),
    )
    .unwrap();
    let manual_resolved = resolve(Some(CommonScalingRequest::SymmetricCongruence(scaling)));
    let manual = manual_resolved.as_spatial().unwrap();
    assert_ne!(automatic.identity(), manual.identity());
    assert_ne!(automatic.realization_digest(), manual.realization_digest());
    assert_eq!(
        automatic.realization_digest(),
        hex_bytes(&automatic.portable_realization().digest().unwrap())
    );
    assert_eq!(
        automatic.linear().algorithm(),
        LinearSolver::MinimumResidual
    );
    assert_eq!(
        automatic.solver_provider(),
        REFERENCE_LINEAR_SOLVER.provider()
    );

    let fields = vec![
        CommonInitialField::new(
            digest.clone(),
            field_ids[0],
            Some(
                CommonInitialValues::new(
                    eqiora_core::ValueShape::new([2]).unwrap(),
                    vec![[0.0; 2]; 6].into_iter().flatten().collect(),
                )
                .unwrap(),
            ),
            Some(
                CommonInitialValues::new(
                    eqiora_core::ValueShape::new([2]).unwrap(),
                    vec![[0.0; 2]; 4].into_iter().flatten().collect(),
                )
                .unwrap(),
            ),
        )
        .unwrap(),
        CommonInitialField::new(
            digest.clone(),
            field_ids[1],
            Some(
                CommonInitialValues::new(eqiora_core::ValueShape::scalar(), vec![0.25; 6]).unwrap(),
            ),
            None,
        )
        .unwrap(),
        CommonInitialField::new(
            digest.clone(),
            field_ids[2],
            Some(
                CommonInitialValues::new(
                    eqiora_core::ValueShape::new([2]).unwrap(),
                    vec![[0.0; 2]; 6].into_iter().flatten().collect(),
                )
                .unwrap(),
            ),
            None,
        )
        .unwrap(),
        CommonInitialField::new(
            digest.clone(),
            field_ids[3],
            Some(
                CommonInitialValues::new(
                    eqiora_core::ValueShape::new([2]).unwrap(),
                    vec![[0.02, 0.0]; 6].into_iter().flatten().collect(),
                )
                .unwrap(),
            ),
            None,
        )
        .unwrap(),
    ];
    let initial = automatic.initial_state(Some(0.0), fields.clone()).unwrap();
    let initial_bytes = initial.to_bytes().unwrap();
    assert_eq!(
        CommonState::from_bytes(&initial_bytes, &automatic_resolved,).unwrap(),
        initial
    );
    let pressure_offset = automatic
        .fields()
        .take_while(|(field, _)| *field != field_ids[1])
        .map(|(field, ty)| {
            automatic.field_coefficient_entities(field).unwrap().len()
                * ty.shape().component_count().unwrap()
        })
        .sum::<usize>();
    assert_eq!(
        &initial.spatial_values().unwrap()[pressure_offset..pressure_offset + 6],
        &[0.25; 6]
    );
    assert!(
        CommonTransientRunRequest::from_steps(manual_resolved.clone(), initial.clone(), 1, vec![1])
            .is_err()
    );
    let manual_initial = manual.initial_state(Some(0.), fields).unwrap();
    let accepted_step = automatic
        .advance_scalar(&initial, &REFERENCE_LINEAR_SOLVER, 0.05)
        .unwrap();
    let accepted = accepted_step.state().clone();
    let scaled = manual
        .advance_scalar(&manual_initial, &REFERENCE_LINEAR_SOLVER, 0.05)
        .unwrap();
    for (left, right) in accepted
        .spatial_values()
        .unwrap()
        .iter()
        .zip(scaled.state().spatial_values().unwrap())
    {
        assert!((left - right).abs() < 1e-10);
    }
    // The common mathematical profile has no FSI-only reduction restriction.
    // Program-controlled decisions and exact direct choices must replay without a solve.
    for objective in [
        SolverPlanningObjective::Robust,
        SolverPlanningObjective::Fast,
        SolverPlanningObjective::LowMemory,
    ] {
        let ranked = ResolvedCommonPlan::resolve(
            &model,
            resources.clone(),
            scoped.clone(),
            CommonSolvePolicy::Linear(
                CommonLinearRequest::program_controlled(
                    1e-11,
                    1e-13,
                    NonZeroUsize::new(20_000).unwrap(),
                    objective,
                )
                .unwrap(),
            ),
            None,
            Some(temporal),
            &FsiPlanningBackend,
            None,
        )
        .unwrap();
        assert_eq!(ranked.solver_planning_objective(), Some(objective));
        assert!(
            replay_plan(ranked, &FsiPlanningBackend)
                .as_spatial()
                .is_some()
        );
    }
    let direct = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::SparseLu,
            1e-11,
            1e-13,
            NonZeroUsize::new(20_000).unwrap(),
        )
        .unwrap()
        .with_preconditioner(PreconditionerPolicy::Identity)
        .with_reduction(ReductionPolicy::Fast),
        FsiPlanningBackend.provider(),
    )
    .unwrap();
    let direct = ResolvedCommonPlan::resolve(
        &model,
        resources.clone(),
        scoped.clone(),
        CommonSolvePolicy::Linear(direct),
        None,
        Some(temporal),
        &FsiPlanningBackend,
        None,
    )
    .unwrap();
    assert_eq!(
        replay_plan(direct, &FsiPlanningBackend)
            .as_spatial()
            .unwrap()
            .linear()
            .algorithm(),
        LinearSolver::SparseLu
    );

    let ten_step = CommonTransientRunRequest::from_steps(
        automatic_resolved.clone(),
        initial.clone(),
        10,
        vec![10],
    )
    .unwrap();
    let std::ops::ControlFlow::Continue(prepared_outputs) = ten_step
        .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |_, _| false)
        .unwrap()
    else {
        panic!("uncancelled prepared FSI Run must finish")
    };
    let mut independently_accepted = initial.clone();
    let mut independently_accepted_at_four = None;
    for accepted_steps in 1..=10 {
        independently_accepted = automatic
            .advance_scalar(
                &independently_accepted,
                &REFERENCE_LINEAR_SOLVER,
                accepted_steps as f64 * 0.05,
            )
            .unwrap()
            .state()
            .clone();
        if accepted_steps == 4 {
            independently_accepted_at_four = Some(independently_accepted.clone());
        }
    }
    assert_eq!(prepared_outputs.len(), 1);
    assert_eq!(prepared_outputs[0].0, 10);
    assert_eq!(
        prepared_outputs[0].1.state().spatial_values(),
        independently_accepted.spatial_values()
    );
    assert_eq!(prepared_outputs[0].1.state().time_s().unwrap(), 0.5);
    assert_eq!(
        ten_step
            .advance_accepted_actions(&REFERENCE_LINEAR_SOLVER, |accepted_steps, _| {
                accepted_steps == 4
            })
            .unwrap(),
        std::ops::ControlFlow::Break((4, independently_accepted_at_four.unwrap())),
    );
    let request = CommonTransientRunRequest::from_steps(
        automatic_resolved.clone(),
        initial.clone(),
        1,
        vec![1],
    )
    .unwrap();
    let trajectory =
        crate::CommonTrajectory::accept_spatial_transient(request, vec![(1, accepted_step)])
            .unwrap();
    let result = crate::CommonResult::accept_trajectory(0.5, trajectory.clone()).unwrap();
    let result_bytes = result.to_bytes().unwrap();
    let replayed_result =
        crate::CommonResult::from_bytes(&result_bytes, &automatic_resolved).unwrap();
    assert_eq!(replayed_result.identity(), result.identity());
    assert_eq!(replayed_result.to_bytes().unwrap(), result_bytes);
    let replayed_steps = replayed_result
        .trajectory()
        .unwrap()
        .spatial_steps()
        .unwrap();
    assert_eq!(replayed_steps.len(), 1);
    assert_eq!(replayed_steps[0].1.state().identity(), accepted.identity());
    assert!(
        replayed_result
            .solve_true_residual_norm(Some(0), None)
            .unwrap()
            <= replayed_result
                .solve_residual_target(Some(0), None)
                .unwrap()
    );
    let mut forged_result: serde_json::Value = serde_json::from_slice(&result_bytes).unwrap();
    forged_result["content"]["payload"]["fsi"] = serde_json::json!({"states": []});
    assert!(
        crate::CommonResult::from_bytes(
            &serde_json::to_vec(&forged_result).unwrap(),
            &automatic_resolved,
        )
        .is_err()
    );
    let trajectory_bytes = trajectory.to_bytes().unwrap();
    let replayed_trajectory =
        crate::CommonTrajectory::from_bytes(&trajectory_bytes, &automatic_resolved).unwrap();
    assert_eq!(replayed_trajectory.identity(), trajectory.identity());
    assert_eq!(replayed_trajectory.to_bytes().unwrap(), trajectory_bytes);
    // Retired schemas and the removed sidecar must not be accepted as aliases.
    let old_schema = |bytes: &[u8], current: &str, retired: &str| {
        let text = std::str::from_utf8(bytes).unwrap();
        let old = text.replacen(current, retired, 1);
        assert_ne!(text, old);
        old.into_bytes()
    };
    assert!(
        CommonState::from_bytes(
            &old_schema(
                &initial_bytes,
                "eqiora.common-spatial-state/v6",
                "eqiora.common-spatial-state/v2"
            ),
            &automatic_resolved,
        )
        .is_err()
    );
    assert!(
        crate::CommonTrajectory::from_bytes(
            &old_schema(
                &trajectory_bytes,
                "eqiora.common-trajectory/v6",
                "eqiora.common-trajectory/v4"
            ),
            &automatic_resolved,
        )
        .is_err()
    );
    assert!(
        crate::CommonResult::from_bytes(
            &old_schema(
                &result_bytes,
                "eqiora.common-result/v22",
                "eqiora.common-result/v15"
            ),
            &automatic_resolved,
        )
        .is_err()
    );
    let replayed_states = replayed_trajectory.spatial_steps().unwrap();
    assert_eq!(replayed_states[0].1.state().identity(), accepted.identity());
    let accepted_bytes = accepted.to_bytes().unwrap();
    let replayed_accepted = CommonState::from_bytes(&accepted_bytes, &automatic_resolved).unwrap();
    assert_eq!(replayed_accepted.identity(), accepted.identity());
    assert_eq!(
        replayed_accepted.spatial_values(),
        accepted.spatial_values()
    );
    assert!(
        automatic
            .advance_scalar(&replayed_accepted, &REFERENCE_LINEAR_SOLVER, 0.1)
            .is_ok()
    );

    let foreign = eqiora_artifact::ArtifactDigest::from_sha256([7; 32]);
    let foreign_scoped = CommonMethodRequest::Scoped(vec![
        CommonScopedSpatialPolicy::new(foreign, fluid_domain, CommonSpatialPolicy::MiniP1),
        CommonScopedSpatialPolicy::new(digest, solid_domain, CommonSpatialPolicy::P1),
    ]);
    assert!(
        ResolvedCommonPlan::resolve(
            &model,
            resources,
            foreign_scoped,
            CommonSolvePolicy::Linear(requested),
            None,
            Some(temporal),
            &ResolveOnlyBackend,
            None,
        )
        .is_err()
    );
}

pub(super) fn exercise_model_driven_common_mesh_admission_evidence() {
    scalar_q1_and_tpfa_consume_one_exact_anisotropic_common_mesh();
    scalar_linear_blocks_execute_and_replay_complete_one_two_three_field_results();
    common_elasticity_plan_consumes_exact_mesh_and_model_meaning();
    mathematical_resolution_ignores_names_and_rejects_changed_operator_or_boundary_law();
    admission_rejects_policy_and_resource_cross_wires();
    transient_common_plan_resolves_exact_mini_and_supplied_cartesian_resources();
}

/// Catalog-authentic provider with an admitted indefinite direct tuple. Execution
/// panics so resolution/rejection cannot accidentally perform numerical work.
#[derive(Debug)]
struct FsiPlanningBackend;
impl LinearSolverBackend for FsiPlanningBackend {
    fn provider(&self) -> SolverProvider {
        PlanningFaerBackend.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        ResolveOnlyBackend.capabilities()
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        unreachable!("solver planning must not execute")
    }
}
