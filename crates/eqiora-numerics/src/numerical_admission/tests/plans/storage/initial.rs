use super::*;

#[test]
fn explicit_three_component_initial_fields_are_exact_and_restartable() {
    let mut source = String::from(
        "model ExplicitInitial() {
        domain body=box(0,1,0,1,0,1);
        variable g:m on body in smooth;
        relation potential on body {g=2*coordinate(0)+3*coordinate(1)+5*coordinate(2);}
        state u:vector<1,3> on body in smooth;
        relation balance on body {1[s/m^2]*derivative(u)=div(grad(u));}",
    );
    for (axis, label) in ["x", "y", "z"].into_iter().enumerate() {
        for side in ["lower", "upper"] {
            source += &format!(
                "domain {label}_{side}=boundary(body,axis={axis},side={side});
                relation fixed_{label}_{side} on {label}_{side} {{trace(u)=trace(grad(g));}}"
            );
        }
    }
    source += "}";
    let (transaction, model, _) = eqiora_compiler::compile("explicit-initial.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&cartesian_box_3d(), &[2, 2, 2]),
        CommonSpatialPolicy::Q1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(100).unwrap(),
        )),
        None,
        Some(CommonBackwardEuler::from_seconds(0.25).unwrap()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
    let plan = resolved.as_linear().unwrap();
    let field = plan.fields().next().unwrap().0;
    let expected = (0..27).flat_map(|_| [2., 3., 5.]).collect::<Vec<_>>();
    let assignment = CommonInitialField::new(
        model.digest().unwrap(),
        field,
        Some(
            CommonInitialValues::new(eqiora_core::ValueShape::new([3]).unwrap(), expected.clone())
                .unwrap(),
        ),
        None,
    )
    .unwrap();
    assert!(
        plan.initial_state(0.0, vec![])
            .unwrap_err()
            .message()
            .contains("omits an exact stored Field")
    );
    assert!(
        plan.initial_state(0.0, vec![assignment.clone(), assignment.clone()])
            .is_err()
    );
    let wrong_shape = CommonInitialField::new(
        model.digest().unwrap(),
        field,
        Some(
            CommonInitialValues::new(eqiora_core::ValueShape::scalar(), expected.clone()).unwrap(),
        ),
        None,
    )
    .unwrap();
    assert!(
        plan.initial_state(0.0, vec![wrong_shape])
            .unwrap_err()
            .message()
            .contains("shape")
    );
    let wrong_model = CommonInitialField::new(
        eqiora_artifact::ArtifactDigest::from_hex("0".repeat(64)).unwrap(),
        field,
        assignment.vertex().cloned(),
        None,
    )
    .unwrap();
    assert!(
        plan.initial_state(0.0, vec![wrong_model])
            .unwrap_err()
            .message()
            .contains("foreign Model")
    );
    let short = CommonInitialField::new(
        model.digest().unwrap(),
        field,
        Some(
            CommonInitialValues::new(
                eqiora_core::ValueShape::new([3]).unwrap(),
                expected[..78].to_vec(),
            )
            .unwrap(),
        ),
        None,
    )
    .unwrap();
    assert!(
        plan.initial_state(0.0, vec![short])
            .unwrap_err()
            .message()
            .contains("cardinality")
    );
    let wrong_association = CommonInitialField::new(
        model.digest().unwrap(),
        field,
        assignment.vertex().cloned(),
        assignment.vertex().cloned(),
    )
    .unwrap();
    assert!(
        plan.initial_state(0.0, vec![wrong_association])
            .unwrap_err()
            .message()
            .contains("association")
    );
    let mut incompatible = expected.clone();
    incompatible[0] += 1.;
    let incompatible = CommonInitialField::new(
        model.digest().unwrap(),
        field,
        Some(
            CommonInitialValues::new(eqiora_core::ValueShape::new([3]).unwrap(), incompatible)
                .unwrap(),
        ),
        None,
    )
    .unwrap();
    assert!(
        plan.initial_state(0.0, vec![incompatible])
            .unwrap_err()
            .message()
            .contains("contradicts prescribed boundary")
    );
    let state = plan.initial_state(0.0, vec![assignment.clone()]).unwrap();
    assert_eq!(state.linear_values().unwrap(), expected);
    let state = CommonState::from_bytes(&state.to_bytes().unwrap(), &resolved).unwrap();
    // The constant vector has zero diffusion and the same boundary traces.
    let next = plan
        .advance_scalar(&state, &REFERENCE_LINEAR_SOLVER, 0.25)
        .unwrap();
    for (&actual, &expected) in next.linear_values().unwrap().iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let restart = plan.initial_state(0.25, vec![assignment]).unwrap();
    let restart = CommonState::from_bytes(&restart.to_bytes().unwrap(), &resolved).unwrap();
    assert_eq!(restart.linear_values().unwrap(), expected);
}
