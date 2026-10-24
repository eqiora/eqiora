use super::vector::open_resources;
use super::*;

mod history;
mod nonlinear;
mod observations;
mod step_products;
mod time_data;

const FLOW: &str = r#"
public model Throughflow(
    support body:volume(ambient_dimension=2),
    support left:boundary(parent=body),
    support right:boundary(parent=body),
    support bottom:boundary(parent=body),
    support top:boundary(parent=body)
) {
    variable u:vector<m/s,2> on body in smooth;
    variable p:kg/(m*s^2) on body in h1;
    variable g:m^2/s on body in smooth;
    variable f:kg/(m*s^2) on body in smooth;
    relation prescribed_velocity on body {g=1[m/s]*coordinate(0);}
    relation force on body {f=0[kg/(m*s^2)];}
    relation momentum on body {-div(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))-grad(f)=0;}
    relation constraint on body {div(u)=0;}
    relation inlet on left {trace(u)=trace(grad(g));}
    relation lower on bottom {trace(u)=trace(grad(g));}
    relation upper on top {trace(u)=trace(grad(g));}
    relation outlet on right {normal(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))=0;}
}
"#;

#[test]
fn mini_pressure_pair_uses_the_common_linear_plan() {
    // Exact barycentric integration on this five-cell mesh gives
    // ||A^-1||_inf = 225764530027697/744320367054 < 304 for its 20 free
    // MINI/P1 coordinates. Thus ||r||_2 <= 1e-13 bounds coefficient error
    // by 3.04e-11, below the 1e-10 checks. A relative residual target alone
    // does not establish that forward accuracy for this saddle system.
    let residual_target = 1e-13;
    let source = FLOW;
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
    for (source, pressure_gradient) in [
        (source.to_owned(), 0.0),
        (source.replace("Throughflow", "RenamedFlow"), 0.0),
        (
            source.replace(
                "f=0[kg/(m*s^2)]",
                "f=1[kg/(m^2*s^2)]*coordinate(0)-1[kg/(m*s^2)]",
            ),
            1.0,
        ),
    ] {
        let entry = if source.contains("model RenamedFlow") {
            "RenamedFlow"
        } else {
            "Throughflow"
        };
        let model = compile_model("common-mini.eqi", &source, &geometry, entry, &supports, &[]);
        let unpaired = compile_model(
            "unpaired-mini.eqi",
            &source.replace("isotropic_lift(p)", "2*isotropic_lift(p)"),
            &geometry,
            entry,
            &supports,
            &[],
        );
        let resolve_unpaired = |algorithm| {
            ResolvedCommonPlan::resolve(
                &unpaired,
                open_resources(&geometry, false),
                CommonSpatialPolicy::MiniP1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    algorithm,
                    0.0,
                    1e-12,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
        };
        // Doubling only the pressure coupling defeats the signed-transpose proof.
        // It remains executable as a general system, but cannot claim MINRES.
        resolve_unpaired(LinearSolver::BiConjugateGradientStabilized).unwrap();
        assert!(resolve_unpaired(LinearSolver::MinimumResidual).is_err());
        for permuted in [false, true] {
            let resolved = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonSpatialPolicy::MiniP1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::BiConjugateGradientStabilized,
                    0.0,
                    residual_target,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let resolved = replay_plan(resolved, &REFERENCE_LINEAR_SOLVER);
            let plan = resolved
                .as_spatial()
                .expect("MINI must use the shared linear owner");
            let result = plan.run(None, &REFERENCE_LINEAR_SOLVER).unwrap();
            let minres = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonSpatialPolicy::MiniP1,
                CommonSolvePolicy::Linear(exact_reference_linear(
                    LinearSolver::MinimumResidual,
                    0.0,
                    residual_target,
                    NonZeroUsize::new(1000).unwrap(),
                )),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let minres = replay_plan(minres, &REFERENCE_LINEAR_SOLVER);
            let minres_result = minres
                .as_spatial()
                .unwrap()
                .run(None, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            let minres_bytes = minres
                .as_spatial()
                .unwrap()
                .run_result(None, &REFERENCE_LINEAR_SOLVER)
                .unwrap()
                .to_bytes()
                .unwrap();
            assert_eq!(
                crate::CommonResult::from_bytes(&minres_bytes, &minres)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                minres_bytes
            );
            assert_eq!(result.fields.len(), 2);
            let published = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
            for (index, (_, ty, values, _)) in result.fields.iter().enumerate() {
                let (association, vertex_values, shape) = published.field_block(index, 0).unwrap();
                assert_eq!(association, "vertex");
                if ty.shape().is_scalar() {
                    assert_eq!(shape, &[6]);
                    assert_eq!(vertex_values, values);
                } else {
                    assert_eq!(shape, &[6, 2]);
                    assert_eq!(vertex_values, &values[..12]);
                    let (association, bubbles, shape) = published.field_block(index, 1).unwrap();
                    assert_eq!(association, "cell-bubble");
                    assert_eq!(shape, &[5, 2]);
                    assert_eq!(bubbles, &values[12..]);
                }
            }
            let bytes = published.to_bytes().unwrap();
            assert_eq!(
                crate::CommonResult::from_bytes(&bytes, &resolved)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            let scoped = CommonMethodRequest::Scoped(vec![CommonScopedSpatialPolicy::new(
                model.digest().unwrap(),
                plan.admission.discretizations[0].domain(),
                CommonSpatialPolicy::MiniP1,
            )]);
            let scoped_plan = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                scoped.clone(),
                resolved.canonical_solve_request().unwrap(),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let scoped_plan = replay_plan(scoped_plan, &REFERENCE_LINEAR_SOLVER);
            assert_eq!(scoped_plan.canonical_method_request(), Some(scoped));
            let scoped_result = scoped_plan
                .as_spatial()
                .unwrap()
                .run(None, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            assert_eq!(scoped_result.fields, result.fields);
            let domain = plan.admission.discretizations[0].domain();
            let binding = CommonScopedSpatialPolicy::new(
                model.digest().unwrap(),
                domain,
                CommonSpatialPolicy::MiniP1,
            );
            let foreign_domain =
                eqiora_core::Id::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap());
            assert_ne!(domain, foreign_domain);
            for (bindings, diagnostic) in [
                (vec![], "nonempty exact Domain bindings"),
                (
                    vec![binding.clone(), binding.clone()],
                    "unique Domains of this exact Model",
                ),
                (
                    vec![CommonScopedSpatialPolicy::new(
                        eqiora_artifact::ArtifactDigest::from_sha256([7; 32]),
                        domain,
                        CommonSpatialPolicy::MiniP1,
                    )],
                    "unique Domains of this exact Model",
                ),
                (
                    vec![CommonScopedSpatialPolicy::new(
                        model.digest().unwrap(),
                        foreign_domain,
                        CommonSpatialPolicy::MiniP1,
                    )],
                    "cover exactly the executable Regions",
                ),
                (
                    vec![
                        binding,
                        CommonScopedSpatialPolicy::new(
                            model.digest().unwrap(),
                            foreign_domain,
                            CommonSpatialPolicy::MiniP1,
                        ),
                    ],
                    "cover exactly the executable Regions",
                ),
                (
                    vec![CommonScopedSpatialPolicy::new(
                        model.digest().unwrap(),
                        domain,
                        CommonSpatialPolicy::P1,
                    )],
                    "require the paired MINI/P1 policy",
                ),
            ] {
                let error = ResolvedCommonPlan::resolve(
                    &model,
                    open_resources(&geometry, permuted),
                    CommonMethodRequest::Scoped(bindings),
                    resolved.canonical_solve_request().unwrap(),
                    None,
                    None,
                    &REFERENCE_LINEAR_SOLVER,
                    None,
                    None,
                )
                .unwrap_err();
                assert!(error.message().contains(diagnostic), "{error:?}");
            }
            let description = resolved.formulation().unwrap();
            assert_eq!(description.effective(), FormulationKind::MixedGalerkin);
            assert_eq!(description.requested(), FormulationSelectionMode::Automatic);
            assert!(
                description
                    .rule_ids()
                    .contains(&"fem.derive.v1.divergence-by-parts")
            );
            assert!(
                description
                    .rule_ids()
                    .contains(&"fem.mixed.v1.continuity-constraint-pairing")
            );
            let exact_request = CommonMethodRequest::Exact {
                spatial: CommonSpatialPolicy::MiniP1,
                formulation: FormulationKind::MixedGalerkin,
            };
            let exact = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                exact_request.clone(),
                resolved.canonical_solve_request().unwrap(),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap();
            let exact = replay_plan(exact, &REFERENCE_LINEAR_SOLVER);
            assert_eq!(exact.canonical_method_request(), Some(exact_request));
            assert_eq!(
                exact.formulation().unwrap().requested(),
                FormulationSelectionMode::Exact
            );
            assert_eq!(
                exact
                    .as_spatial()
                    .unwrap()
                    .run(None, &REFERENCE_LINEAR_SOLVER)
                    .unwrap()
                    .fields,
                result.fields
            );
            let error = ResolvedCommonPlan::resolve(
                &model,
                open_resources(&geometry, permuted),
                CommonMethodRequest::Exact {
                    spatial: CommonSpatialPolicy::MiniP1,
                    formulation: FormulationKind::PrimalGalerkin,
                },
                resolved.canonical_solve_request().unwrap(),
                None,
                None,
                &REFERENCE_LINEAR_SOLVER,
                None,
                None,
            )
            .unwrap_err();
            assert!(error.message().contains("MixedGalerkin"));
            let mut drift = plan.clone();
            drift.formulation.as_mut().unwrap().kind = FormulationKind::PrimalGalerkin;
            assert!(
                drift
                    .run(None, &REFERENCE_LINEAR_SOLVER)
                    .unwrap_err()
                    .message()
                    .contains("mixed Region formulation")
            );

            // u=(1,0) has zero symmetric gradient and divergence. Momentum
            // requires grad(p)=grad(f); zero right traction fixes p(1)=0.
            // Thus p=pressure_gradient*(x-1). P1 coefficients reproduce the
            // constant velocity; all five hierarchical bubble coefficients vanish.
            for (_, value_type, values, space) in result.fields.iter().chain(&minres_result.fields)
            {
                if value_type.shape().is_scalar() {
                    assert_eq!(
                        *space,
                        Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                    );
                    assert_eq!(values.len(), 6);
                    let owner = open_resources(&geometry, permuted);
                    let vertices = owner.simplicial_mesh().unwrap().mesh().vertices();
                    for (value, point) in values.iter().zip(vertices) {
                        assert!(
                            (value - pressure_gradient * (point[0] - 1.0)).abs() < 1e-10,
                            "pressure={value:e}, point={point:?}, gradient={pressure_gradient}, permuted={permuted}; BiCGSTAB={:?}; MINRES={:?}",
                            result.solve_report,
                            minres_result.solve_report,
                        );
                    }
                } else {
                    assert_eq!(*space, Space::simplex_p1_bubble());
                    assert_eq!(values.len(), 22);
                    for value in values[..12].as_chunks::<2>().0 {
                        assert!((value[0] - 1.).abs() < 1e-10);
                        assert!(value[1].abs() < 1e-10);
                    }
                    assert!(values[12..].iter().all(|bubble| bubble.abs() < 1e-10));
                }
            }
        }
    }
}

#[test]
fn closed_velocity_boundary_cannot_publish_an_arbitrary_constant_pressure() {
    // With zero velocity on the whole boundary and zero forcing, every
    // constant pressure gives the same zero residual. A zero RHS solver can
    // return immediately, so convergence alone cannot establish uniqueness.
    let source = FLOW
        .replace("g=1[m/s]*coordinate(0)", "g=0[m^2/s]")
        .replace(
            "normal(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))=0",
            "trace(u)=trace(grad(g))",
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
        "closed-common-mini.eqi",
        &source,
        &geometry,
        "Throughflow",
        &supports,
        &[],
    );
    let resolved = ResolvedCommonPlan::resolve(
        &model,
        open_resources(&geometry, false),
        CommonSpatialPolicy::MiniP1,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-11,
            1e-13,
            NonZeroUsize::new(1000).unwrap(),
        )),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
        None,
    )
    .unwrap();
    let plan = resolved.as_spatial().expect("shared linear owner");
    // Result replay uses this same admission check even without a fresh solve.
    let replay_error = plan
        .reaccept_static_coefficients(&zero_real_coefficients(plan), None, None, 0.)
        .unwrap_err();
    assert!(
        replay_error
            .message()
            .contains("unclosed constant multiplier")
    );

    let error = plan
        .run_result(None, &REFERENCE_LINEAR_SOLVER)
        .expect_err("unclosed pressure must reject before publication");
    assert!(
        error.message().contains("unclosed constant multiplier"),
        "{error:?}"
    );
}
