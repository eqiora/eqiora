use super::*;

#[test]
fn stationary_state_owns_complete_fields_without_model_time() {
    let geometry = cartesian_interval();
    let model = scalar_box_model(
        &geometry,
        POISSON_INTERVAL,
        "PoissonInterval",
        &["left", "right"],
    );
    let plan = ResolvedCommonPlan::Spatial(Box::new(resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[4]),
        CommonSpatialPolicy::Q1,
    )));
    let spatial = plan.as_spatial().unwrap();
    let field = spatial.fields().next().unwrap().0;
    let assignment = |values| {
        CommonInitialField::new(
            model.digest().unwrap(),
            field,
            Some(CommonInitialValues::new(eqiora_core::ValueShape::scalar(), values).unwrap()),
            None,
        )
        .unwrap()
    };
    // A seed is not an accepted solution: these interior values deliberately
    // violate Poisson, while obeying the exact nodal support and boundary trace.
    let values = vec![0., 4., -2., 7., 0.];
    let state = spatial
        .initial_state(None, vec![assignment(values.clone())])
        .unwrap();
    assert_eq!(state.time_s(), None);
    assert_eq!(state.spatial_values(), Some(values.as_slice()));
    let bytes = state.to_bytes().unwrap();
    assert_eq!(CommonState::from_bytes(&bytes, &plan).unwrap(), state);
    let projected = spatial.state_fields(&state).unwrap().collect::<Vec<_>>();
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].0, field);
    assert_eq!(projected[0].3, values);

    assert!(
        spatial
            .initial_state(Some(0.), vec![assignment(values)])
            .is_err()
    );
    assert!(spatial.initial_state(None, vec![]).is_err());
    assert!(
        spatial
            .initial_state(None, vec![assignment(vec![0.; 4])])
            .is_err()
    );
    assert!(
        spatial
            .initial_state(None, vec![assignment(vec![1.; 5])])
            .is_err()
    );
    assert!(
        CommonTransientRunRequest::from_steps(plan.clone(), state.clone(), 1, vec![1]).is_err()
    );
    assert!(CommonAcceptedSpatialStep::new(&state, state.clone()).is_err());

    let other = ResolvedCommonPlan::Spatial(Box::new(resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[8]),
        CommonSpatialPolicy::Q1,
    )));
    assert!(CommonState::from_bytes(&bytes, &other).is_err());
    let obsolete = String::from_utf8(bytes)
        .unwrap()
        .replace("common-spatial-state/v6", "common-spatial-state/v5");
    assert!(CommonState::from_bytes(obsolete.as_bytes(), &plan).is_err());
}
