//! Ordinary quadratic admission/topology; manufactured solves have separate oracles.
use super::*;

#[test]
fn quadratic_plan_keeps_shared_edges_faces_and_cell_nodes_through_replay() {
    let geometry = cartesian_box_3d();
    let names = [
        "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
    ];
    let model = scalar_box_model(&geometry, POISSON_BOX, "PoissonBox", &names);
    let plan = resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[2, 3, 2]),
        CommonSpatialPolicy::Q2,
    );
    let replayed = replay_plan(
        ResolvedCommonPlan::Spatial(Box::new(plan)),
        &ResolveOnlyBackend,
    )
    .as_spatial()
    .cloned()
    .unwrap();
    let field = replayed.fields().next().unwrap().0;
    let entities = replayed.field_coefficient_entities(field).unwrap();
    // For nx=2,ny=3,nz=2: V=3*4*3, E=2*4*3+3*3*3+3*4*2,
    // F=2*3*3+2*4*2+3*3*2, C=2*3*2. Shared entities occur once.
    assert_eq!(
        (0..=3)
            .map(|d| entities
                .iter()
                .filter(|entity| entity.dimension() == d)
                .count())
            .collect::<Vec<_>>(),
        [36, 75, 52, 12]
    );
    assert_eq!(entities.len(), 5 * 7 * 5);
    assert!(entities.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        replayed.field_space(field).unwrap(),
        Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap())
    );
}

fn quadrature_plan(
    source: &str,
    spatial: CommonSpatialPolicy,
    quadrature: Option<eqiora_realization::QuadraturePolicy>,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    let geometry = cartesian_box_3d();
    let model = scalar_box_model(
        &geometry,
        source,
        "PoissonBox",
        &[
            "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
        ],
    );
    ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2, 2, 2]),
        spatial,
        CommonSolvePolicy::Linear(exact_reference_linear(
            LinearSolver::BiConjugateGradientStabilized,
            1e-10,
            1e-12,
            NonZeroUsize::new(10000).unwrap(),
        )),
        None,
        None,
        &ResolveOnlyBackend,
        None,
        None,
        quadrature,
    )
}

fn gauss(points: usize) -> eqiora_realization::QuadraturePolicy {
    eqiora_realization::QuadraturePolicy::GaussLegendre {
        points_per_axis: NonZeroUsize::new(points).unwrap(),
    }
}

#[test]
fn quadratic_default_and_explicit_quadrature_are_effective_replayed_policies() {
    for (requested, expected) in [(None, 4), (Some(gauss(3)), 3), (Some(gauss(5)), 5)] {
        let plan = quadrature_plan(POISSON_BOX, CommonSpatialPolicy::Q2, requested).unwrap();
        assert_eq!(
            plan.as_spatial().unwrap().admission.assembly_quadrature(),
            Some(gauss(expected))
        );
        let wire: serde_json::Value = serde_json::from_slice(&plan.to_bytes().unwrap()).unwrap();
        assert_eq!(wire["schema"], "eqiora.resolved-common-plan/v28");
        assert_eq!(
            wire["assembly_quadrature"],
            serde_json::to_value(gauss(expected)).unwrap()
        );
        let replayed = replay_plan(plan, &ResolveOnlyBackend);
        assert_eq!(
            replayed
                .as_spatial()
                .unwrap()
                .admission
                .assembly_quadrature(),
            Some(gauss(expected))
        );
        replayed
            .as_spatial()
            .unwrap()
            .admission
            .revalidate()
            .unwrap();
    }
    assert!(quadrature_plan(POISSON_BOX, CommonSpatialPolicy::Q2, Some(gauss(2))).is_err());
    assert!(
        quadrature_plan(
            POISSON_BOX,
            CommonSpatialPolicy::Q2,
            Some(gauss(usize::MAX))
        )
        .is_err()
    );
    assert!(
        quadrature_plan(
            POISSON_BOX,
            CommonSpatialPolicy::Q2,
            Some(eqiora_realization::QuadraturePolicy::CellCentroid)
        )
        .is_err()
    );
    assert!(
        quadrature_plan(
            POISSON_BOX,
            CommonSpatialPolicy::Q2,
            Some(
                eqiora_realization::QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(3).unwrap(),
                    points_per_axis: NonZeroUsize::new(4).unwrap(),
                }
            )
        )
        .is_err()
    );
    assert_eq!(
        quadrature_plan(POISSON_BOX, CommonSpatialPolicy::Q1, None)
            .unwrap()
            .as_spatial()
            .unwrap()
            .admission
            .assembly_quadrature(),
        Some(gauss(2))
    );
}

#[test]
fn natural_polynomial_minimum_raises_one_effective_volume_and_facet_rule() {
    // A degree-six natural datum times a linear trace has degree seven: four Gauss points.
    let source = POISSON_BOX.replace("potential: 1 on body in h1", "potential: 1 on body in smooth").replace(
        "relation x_upper_value on x_upper { trace(potential) = 0; }",
        "relation x_upper_value on x_upper { normal(grad(potential)) = (coordinate(1)/1[m])^6/1[m]; }",
    );
    assert!(quadrature_plan(&source, CommonSpatialPolicy::Q1, Some(gauss(3))).is_err());
    let plan = quadrature_plan(&source, CommonSpatialPolicy::Q1, None).unwrap();
    let plan = plan.as_spatial().unwrap();
    assert_eq!(plan.admission.assembly_quadrature(), Some(gauss(4)));
    let RecognizedNativeModel::Spatial(equations) = plan.admission.recognized_model() else {
        panic!("scalar fixture")
    };
    let (_, input) = equations
        .static_assembly(
            plan.admission.resources(),
            &plan.admission.discretizations,
            &plan.admission.system_scaling,
            plan.admission.operator_properties,
            plan.admission.assembly_quadrature(),
        )
        .unwrap();
    assert!(
        input
            .forms
            .iter()
            .all(|(_, rule)| rule.points().len() == 4usize.pow(3))
    );
    for (region, (form, _)) in equations.regions.iter().zip(&input.forms) {
        assert_eq!(form.retained_assembly_points(), Some(4));
        for (&field, laws) in region.form.boundary_laws() {
            for law in laws.values().filter(|law| {
                law.quantity != crate::canonical_boundary::PhysicalBoundaryQuantity::Trace
            }) {
                assert_eq!(
                    form.natural_facet_quadrature(field, law.polynomial_degree())
                        .unwrap()
                        .points()
                        .len(),
                    4usize.pow(2)
                );
            }
        }
    }
    let replayed = replay_plan(
        ResolvedCommonPlan::Spatial(Box::new(plan.clone())),
        &ResolveOnlyBackend,
    );
    assert_eq!(
        replayed
            .as_spatial()
            .unwrap()
            .admission
            .assembly_quadrature(),
        Some(gauss(4))
    );
}

#[test]
fn quadrature_request_preserves_tpfa_centroid_owner_and_rejects_override() {
    let plan = quadrature_plan(POISSON_BOX, CommonSpatialPolicy::CellCenteredTpfa, None).unwrap();
    let spatial = plan.as_spatial().unwrap();
    assert_eq!(spatial.admission.assembly_quadrature(), None);
    assert!(
        spatial
            .portable_realization()
            .domains()
            .iter()
            .all(|domain| domain.discretization().quadrature()
                == eqiora_realization::QuadraturePolicy::CellCentroid)
    );
    assert!(
        quadrature_plan(
            POISSON_BOX,
            CommonSpatialPolicy::CellCenteredTpfa,
            Some(gauss(4))
        )
        .is_err()
    );
    replay_plan(plan, &ResolveOnlyBackend);
}
