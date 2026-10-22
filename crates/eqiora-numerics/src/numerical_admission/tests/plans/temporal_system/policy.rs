//! Common method controls preserve spatial semantics and exact replay.
use super::*;
use eqiora_core::TimeStateCoordinate;
use eqiora_time::TimeMethod;

pub(super) fn policy(plan: &ResolvedCommonPlan, method: TimeMethod, step: f64) -> CommonOdePolicy {
    let tolerances = plan
        .as_spatial()
        .unwrap()
        .fields()
        .flat_map(|(field, ty)| {
            (0..ty.shape().component_count().unwrap()).flat_map(move |component| {
                [false, true]
                    .into_iter()
                    .take(
                        if ty.scalar_domain() == eqiora_core::ScalarDomain::Complex {
                            2
                        } else {
                            1
                        },
                    )
                    .map(move |imaginary| {
                        (
                            TimeStateCoordinate::new(field, 0, component, imaginary),
                            1e-12,
                        )
                    })
            })
        })
        .collect();
    CommonOdePolicy::new(method, step, 1e-10, tolerances).unwrap()
}

pub(super) fn with_policy(
    plan: &ResolvedCommonPlan,
    policy: CommonOdePolicy,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    let backend = match policy.method() {
        TimeMethod::ImplicitMidpoint => eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES,
        _ => TEST_TIME_BACKEND,
    };
    with_backend(plan, Some(policy.into()), Some(backend))
}

fn with_backend(
    plan: &ResolvedCommonPlan,
    temporal: Option<CommonTemporalPolicy>,
    backend: Option<eqiora_time::TimeBackendCapabilities>,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    ResolvedCommonPlan::resolve(
        plan.model_artifact(),
        plan.authenticated_mesh().unwrap(),
        plan.canonical_method_request().unwrap(),
        plan.canonical_solve_request().unwrap(),
        plan.canonical_scaling_request(),
        temporal,
        &REFERENCE_LINEAR_SOLVER,
        backend,
        None,
    )
}

#[test]
fn controlled_spatial_plans_replay_without_binding_a_backward_euler_step() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let base = resolve(
                complex,
                spatial,
                0.25,
                "1[s/m^2]*derivative(u)-div(grad(u))=0[1/m^2];",
                Some("0"),
            );
            let mut identities = BTreeSet::new();
            for method in [TimeMethod::ImplicitMidpoint, TimeMethod::Tsitouras45] {
                for step in [0.125, 2.] {
                    let requested = policy(&base, method, step);
                    let plan = with_policy(&base, requested.clone()).unwrap();
                    assert!(plan.is_temporal());
                    assert!(plan.backward_euler().is_none());
                    assert_eq!(plan.ode_temporal(), Some(&requested));
                    assert!(identities.insert(plan.identity().to_owned()));
                    let bytes = plan.to_bytes().unwrap();
                    let replay = replay_plan(plan, &REFERENCE_LINEAR_SOLVER);
                    assert_eq!(replay.to_bytes().unwrap(), bytes);
                    assert_eq!(replay.ode_temporal(), Some(&requested));
                    let spatial_plan = replay.as_spatial().unwrap();
                    spatial_plan.admission.revalidate().unwrap();
                    let (field, ty) = spatial_plan.fields().next().unwrap();
                    let one = C::new(1., if complex { 2. } else { 0. });
                    let values = if spatial == CommonSpatialPolicy::Q1 {
                        vec![C::new(0., 0.), one, C::new(0., 0.)]
                    } else {
                        vec![one; 2]
                    };
                    let values = CommonInitialValues::new(ty.shape().clone(), values).unwrap();
                    let assignment = CommonInitialField::new(
                        replay.model_artifact().digest().unwrap(),
                        field,
                        (spatial == CommonSpatialPolicy::Q1).then(|| values.clone()),
                        (spatial == CommonSpatialPolicy::CellCenteredTpfa).then_some(values),
                    )
                    .unwrap();
                    let initial = spatial_plan
                        .initial_state(Some(0.), vec![assignment])
                        .unwrap();
                    let initial_bytes = initial.to_bytes().unwrap();
                    let restored = CommonState::from_bytes(&initial_bytes, &replay).unwrap();
                    assert_eq!(restored.to_bytes().unwrap(), initial_bytes);
                    let error =
                        CommonTransientRunRequest::from_steps(replay.clone(), restored, 1, vec![1])
                            .unwrap_err();
                    assert!(error.message().contains("fixed-grid"));
                    assert_eq!(initial.to_bytes().unwrap(), initial_bytes);
                    let system = system(spatial_plan);
                    let count = system.dimension();
                    let mut residual = vec![f64::NAN; count];
                    // Constant free values 1 and rates 3: Q1 M=1/3,K=4;
                    // TPFA M=I/2 and each K row sums to 4. Both real channels
                    // are independent; these integrals do not use the policy step.
                    system
                        .residual(0.3, &vec![1.; count], &vec![3.; count], &mut residual)
                        .unwrap();
                    close(
                        &residual,
                        &vec![
                            if spatial == CommonSpatialPolicy::Q1 {
                                5.
                            } else {
                                5.5
                            };
                            count
                        ],
                    );
                }
            }
        }
    }
}

#[test]
fn spatial_time_tolerances_bind_exact_fields_components_and_complex_channels() {
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let base = resolve(true, spatial, 0.25, "1[s]*derivative(u)+u=0;", None);
        let valid = policy(&base, TimeMethod::ImplicitMidpoint, 0.125);
        let entries = valid.absolute_tolerances();
        let field = entries[0].0.field();
        let invalid = [
            vec![entries[0]],
            vec![
                entries[0],
                (TimeStateCoordinate::new(field, 0, 1, true), 1e-12),
            ],
            vec![
                entries[0],
                (TimeStateCoordinate::new(field, 1, 0, true), 1e-12),
            ],
        ];
        for tolerances in invalid {
            let request =
                CommonOdePolicy::new(TimeMethod::ImplicitMidpoint, 0.125, 1e-10, tolerances)
                    .unwrap();
            let error = with_policy(&base, request).unwrap_err();
            assert!(error.message().contains("cover exactly"), "{error}");
        }
        let plan = with_policy(&base, valid.clone()).unwrap();
        let mut changed = valid.absolute_tolerances().to_vec();
        changed[0].1 *= 10.;
        let changed = CommonOdePolicy::new(
            valid.method(),
            valid.initial_step_s(),
            valid.relative_tolerance(),
            changed,
        )
        .unwrap();
        assert_ne!(
            with_policy(&base, changed).unwrap().identity(),
            plan.identity()
        );
        let changed = CommonOdePolicy::new(
            valid.method(),
            valid.initial_step_s(),
            1e-9,
            valid.absolute_tolerances().to_vec(),
        )
        .unwrap();
        assert_ne!(
            with_policy(&base, changed).unwrap().identity(),
            plan.identity()
        );
    }
}

#[test]
fn spatial_time_plan_rejects_retired_schema_and_changed_controls() {
    let base = resolve(
        false,
        CommonSpatialPolicy::Q1,
        0.25,
        "1[s]*derivative(u)+u=0;",
        None,
    );
    let requested = policy(&base, TimeMethod::ImplicitMidpoint, 0.125);
    let plan = with_policy(&base, requested).unwrap();
    let original = String::from_utf8(plan.to_bytes().unwrap()).unwrap();
    assert!(original.contains("eqiora.resolved-common-plan/v21"));
    for (before, after, diagnostic) in [
        (
            "eqiora.resolved-common-plan/v21",
            "eqiora.resolved-common-plan/v20",
            "schema",
        ),
        (
            "\"relative_tolerance\":1e-10",
            "\"relative_tolerance\":1e-8",
            "canonical",
        ),
    ] {
        // Preserve the exact encoding: only the selected semantic datum changes.
        assert_eq!(original.matches(before).count(), 1);
        let wire = original.replacen(before, after, 1);
        let error = ResolvedCommonPlan::from_bytes(
            wire.as_bytes(),
            &REFERENCE_LINEAR_SOLVER,
            eqiora_time::ImplicitMidpointTimeBackend::CAPABILITIES,
        )
        .unwrap_err();
        assert!(error.message().contains(diagnostic), "{error}");
    }
}

#[test]
fn spatial_time_backend_admission_binds_identity_and_available_domains() {
    use eqiora_time::{TimeBackendCapabilities, TimeBackendIdentity};
    let base = resolve(
        true,
        CommonSpatialPolicy::Q1,
        0.25,
        "1[s]*derivative(u)+u=0;",
        None,
    );
    let adaptive = policy(&base, TimeMethod::Tsitouras45, 0.125);
    let resolved = with_backend(
        &base,
        Some(adaptive.clone().into()),
        Some(TEST_TIME_BACKEND),
    )
    .unwrap();
    let real_only = TimeBackendCapabilities::new(
        TEST_TIME_BACKEND.identity(),
        &[eqiora_core::ScalarDomain::Real],
        &[eqiora_core::ScalarType::F64],
    );
    let wrong_precision = TimeBackendCapabilities::new(
        TEST_TIME_BACKEND.identity(),
        &[eqiora_core::ScalarDomain::Complex],
        &[eqiora_core::ScalarType::F32],
    );
    for backend in [real_only, wrong_precision] {
        let error = with_backend(&base, Some(adaptive.clone().into()), Some(backend)).unwrap_err();
        assert!(error.message().contains("does not support"), "{error}");
        let error = ResolvedCommonPlan::from_bytes(
            &resolved.to_bytes().unwrap(),
            &REFERENCE_LINEAR_SOLVER,
            backend,
        )
        .unwrap_err();
        assert!(error.message().contains("does not support"), "{error}");
    }
    for identity in [
        TimeBackendIdentity::new("eqiora.test.other-time", "1"),
        TimeBackendIdentity::new("eqiora.test.time", "2"),
    ] {
        let other = TimeBackendCapabilities::new(
            identity,
            &[
                eqiora_core::ScalarDomain::Real,
                eqiora_core::ScalarDomain::Complex,
            ],
            &[eqiora_core::ScalarType::F64],
        );
        let changed = with_backend(&base, Some(adaptive.clone().into()), Some(other)).unwrap();
        assert_ne!(changed.identity(), resolved.identity());
        let bytes = changed.to_bytes().unwrap();
        assert_eq!(
            ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, other)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let error =
            ResolvedCommonPlan::from_bytes(&bytes, &REFERENCE_LINEAR_SOLVER, TEST_TIME_BACKEND)
                .unwrap_err();
        assert!(
            error.message().contains("time backend identity or version"),
            "{error}"
        );
    }
    for (temporal, backend, message) in [
        (
            Some(adaptive.into()),
            None,
            "requires an available time backend",
        ),
        (
            Some(policy(&base, TimeMethod::ImplicitMidpoint, 0.125).into()),
            Some(TEST_TIME_BACKEND),
            "available host backend identity",
        ),
        (
            base.backward_euler().map(Into::into),
            Some(TEST_TIME_BACKEND),
            "requires a controlled time policy",
        ),
    ] {
        let error = with_backend(&base, temporal, backend).unwrap_err();
        assert!(error.message().contains(message), "{error}");
    }
}

#[test]
fn spatial_time_backend_payload_rejects_missing_or_changed_provider() {
    let base = resolve(
        false,
        CommonSpatialPolicy::Q1,
        0.25,
        "1[s]*derivative(u)+u=0;",
        None,
    );
    let resolved = with_policy(&base, policy(&base, TimeMethod::Tsitouras45, 0.125)).unwrap();
    let original = String::from_utf8(resolved.to_bytes().unwrap()).unwrap();
    let provider = r#""time_backend":{"id":"eqiora.test.time","version":"1"},"#;
    assert_eq!(original.matches(provider).count(), 1);
    for (before, after, message) in [
        (provider, "", "must be present together"),
        (
            "eqiora.test.time",
            "eqiora.test.foreign",
            "time backend identity or version",
        ),
        (
            r#""version":"1""#,
            r#""version":"2""#,
            "time backend identity or version",
        ),
    ] {
        assert_eq!(original.matches(before).count(), 1);
        let changed = original.replacen(before, after, 1);
        let error = ResolvedCommonPlan::from_bytes(
            changed.as_bytes(),
            &REFERENCE_LINEAR_SOLVER,
            TEST_TIME_BACKEND,
        )
        .unwrap_err();
        assert!(error.message().contains(message), "{error}");
    }
    let original = String::from_utf8(base.to_bytes().unwrap()).unwrap();
    let changed = original.replacen(r#""backend":"#, &format!("{provider}\"backend\":"), 1);
    let error = ResolvedCommonPlan::from_bytes(
        changed.as_bytes(),
        &REFERENCE_LINEAR_SOLVER,
        TEST_TIME_BACKEND,
    )
    .unwrap_err();
    assert!(
        error.message().contains("must be present together"),
        "{error}"
    );
}
