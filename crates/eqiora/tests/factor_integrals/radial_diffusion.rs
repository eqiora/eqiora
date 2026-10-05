//! Radial conservation names its coordinate, center flux and spherical observation.
use eqiora::api::ModelDocument;
use eqiora::compiler::StaticBindingValue;
use eqiora::kernel::AxisBounds;
use eqiora::{DimExponents, DynQuantity};

const SOURCE: &str =
    include_str!("../../../../verify/language/factor-integrals/models/radial-diffusion.eqi");

fn document(source: &str) -> ModelDocument {
    document_with_radius(source, 1.0)
}

fn document_with_radius(source: &str, outer: f64) -> ModelDocument {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    ModelDocument::compile_selected(
        "radial-diffusion.eqi",
        source,
        "Particle",
        &[(
            "radius",
            StaticBindingValue::CoordinateInterval(
                AxisBounds::new(
                    DynQuantity::new(0.0, length),
                    DynQuantity::new(outer, length),
                )
                .unwrap(),
            ),
        )],
    )
    .unwrap()
}

#[test]
fn radial_conservation_retains_center_surface_and_measure_through_model_replay() {
    let document = document(SOURCE);
    let replay = ModelDocument::replay(&document.canonical_json().unwrap()).unwrap();
    assert_eq!(document.program(), replay.program());
}

#[test]
fn radial_diffusion_result_replays_and_converges_to_the_spherical_average() {
    use eqiora::artifact::ModelEnvelope;
    use eqiora::meshing::QuadratureRule;
    use eqiora::solver::{
        LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, SolverPlan,
    };
    use eqiora_numerics::{
        AuthenticatedCommonMesh, CommonLinearRequest, CommonResult, CommonSolvePolicy,
        CommonSpatialPolicy, ResolvedCommonPlan, resolve_common_plan,
    };
    use std::{collections::HashMap, num::NonZeroUsize};
    let document = document(SOURCE);
    let original = ModelEnvelope::from_program(document.program()).unwrap();
    let model =
        ModelEnvelope::from_json(&original.canonical_json().unwrap(), Default::default()).unwrap();
    let radius = document.aliases()["radius"].downcast().unwrap();
    let observable = |name| document.aliases()[name].downcast().unwrap();
    for n in [1, 2, 4, 8, 16] {
        let mesh = AuthenticatedCommonMesh::coordinate_factors(&model, radius, &[n]).unwrap();
        let mesh = AuthenticatedCommonMesh::from_bytes(&mesh.to_bytes().unwrap()).unwrap();
        let linear = CommonLinearRequest::exact(
            SolverPlan::new(
                LinearSolver::ConjugateGradient,
                1e-12,
                1e-14,
                NonZeroUsize::new(1000).unwrap(),
            )
            .unwrap(),
            REFERENCE_SOLVER_PROVIDER,
        )
        .unwrap();
        let plan = resolve_common_plan(
            &model,
            mesh,
            CommonSpatialPolicy::CellCentered,
            CommonSolvePolicy::Linear(linear),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap();
        let plan = ResolvedCommonPlan::from_bytes(
            &plan.to_bytes().unwrap(),
            &REFERENCE_LINEAR_SOLVER,
            eqiora_time::TimeBackendCapabilities::new(
                eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
                &[
                    eqiora_core::ScalarDomain::Real,
                    eqiora_core::ScalarDomain::Complex,
                ],
                &[eqiora_core::ScalarType::F64],
            ),
        )
        .unwrap();
        let result = plan
            .as_scalar()
            .unwrap()
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let result = CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap();
        let rules = HashMap::from([(radius, QuadratureRule::gauss_legendre(2).unwrap())]);
        // Integrating (r²*j)'=6r² and imposing regular j(0)=0 gives j=2r.
        // Then c'=-2r and c(1)=0 give c=1-r², whose spherical mean is 2/5.
        // On a uniform cell-centered grid, the surface half-cell closure gives
        // c_i=1-r_i²+h²/4. Summing exact spherical cell volumes yields
        // mean_h=2/5+2h²/3-h⁴/15. This separates reconstruction from quadrature error.
        let h = 1.0 / n as f64;
        let mean = 2.0 / 5.0 + 2.0 * h * h / 3.0 - h.powi(4) / 15.0;
        for (name, expected) in [
            ("average", mean),
            ("volume", 4.0 * std::f64::consts::PI / 3.0),
            ("amount", mean * 4.0 * std::f64::consts::PI / 3.0),
        ] {
            let value = result.observe(&model, observable(name), &rules).unwrap();
            let actual = value.value().real_scalar_value().unwrap().value();
            assert!(
                (actual - expected).abs() < 1e-10,
                "n={n}, {name}: {actual} vs {expected}"
            );
        }
        for (field_index, (id, _)) in plan.as_scalar().unwrap().fields().enumerate() {
            let (_, values, shape) = result.field_block(field_index, 0).unwrap();
            assert_eq!(shape, [n]);
            for (i, value) in values.iter().enumerate() {
                let r = (i as f64 + 0.5) * h;
                let expected = if id.erase() == document.aliases()["concentration"] {
                    1.0 - r * r + h * h / 4.0
                } else {
                    assert_eq!(id.erase(), document.aliases()["flux"]);
                    2.0 * r
                };
                assert!(
                    (value - expected).abs() < 1e-10,
                    "n={n}, cell={i}: {value} vs {expected}"
                );
            }
        }
    }
}

fn select(
    source: &str,
) -> Result<
    (
        eqiora::artifact::ModelEnvelope,
        ModelDocument,
        eqiora_numerics::ResolvedCommonPlan,
    ),
    eqiora::Diagnostic,
> {
    select_document(document(source))
}

fn select_document(
    document: ModelDocument,
) -> Result<
    (
        eqiora::artifact::ModelEnvelope,
        ModelDocument,
        eqiora_numerics::ResolvedCommonPlan,
    ),
    eqiora::Diagnostic,
> {
    use eqiora::solver::{
        LinearSolver, REFERENCE_LINEAR_SOLVER, REFERENCE_SOLVER_PROVIDER, SolverPlan,
    };
    use eqiora_numerics::{
        AuthenticatedCommonMesh, CommonLinearRequest, CommonSolvePolicy, CommonSpatialPolicy,
        resolve_common_plan,
    };
    let model = eqiora::artifact::ModelEnvelope::from_program(document.program()).unwrap();
    let radius = document.aliases()["radius"].downcast().unwrap();
    let mesh = AuthenticatedCommonMesh::coordinate_factors(&model, radius, &[4]).unwrap();
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1e-12,
            1e-30,
            std::num::NonZeroUsize::new(1000).unwrap(),
        )
        .unwrap(),
        REFERENCE_SOLVER_PROVIDER,
    )
    .unwrap();
    let plan = resolve_common_plan(
        &model,
        mesh,
        CommonSpatialPolicy::CellCentered,
        CommonSolvePolicy::Linear(linear),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )?;
    Ok((model, document, plan))
}

#[test]
fn radial_admission_rejects_lost_weights_irregular_centers_and_non_diffusive_laws() {
    select(SOURCE).unwrap();
    for (from, to, message) in [
        (
            "partial(r*r*flux,wrt=r)=production*r*r",
            "partial(flux,wrt=r)=production",
            "radial relations",
        ),
        (
            "side=upper)=0[1/m^2/s]",
            "side=upper)=1[1/m^2/s]",
            "zero center flux",
        ),
        (
            "side=upper)=0[1/m^2/s]",
            "side=lower)=0[1/m^2/s]",
            "zero center flux",
        ),
        ("r=1[m]", "r=0.75[m]", "inward surface value"),
        (
            "diffusivity:m^2/s=1[m^2/s]",
            "diffusivity:m^2/s=0[m^2/s]",
            "positive D",
        ),
        (
            "diffusivity:m^2/s=1[m^2/s]",
            "diffusivity:m^2/s=-1[m^2/s]",
            "positive D",
        ),
        (
            "flux=-diffusivity*partial(concentration,wrt=r)",
            "flux=-diffusivity*partial(concentration,wrt=r)+1[m^4/s]*concentration^2",
            "radial relations",
        ),
    ] {
        let error = select(&SOURCE.replace(from, to)).unwrap_err();
        assert!(error.message().contains(message), "{from}: {error:?}");
    }
}

#[test]
fn radial_fixed_parameters_keep_small_physical_scales_and_ignore_field_names() {
    use eqiora::meshing::QuadratureRule;
    use eqiora::solver::REFERENCE_LINEAR_SOLVER;
    use std::collections::HashMap;
    for (source, expected) in [
        (
            SOURCE
                .replace(
                    "diffusivity:m^2/s=1[m^2/s]",
                    "diffusivity:m^2/s=1e-14[m^2/s]",
                )
                .replace(
                    "production:1/m^3/s=6[1/m^3/s]",
                    "production:1/m^3/s=6e-14[1/m^3/s]",
                ),
            113.0 / 256.0,
        ),
        (
            SOURCE
                .replace("concentration", "stored")
                .replace("flux", "transfer"),
            113.0 / 256.0,
        ),
        (
            SOURCE
                .replace(
                    "production:1/m^3/s=6[1/m^3/s]",
                    "production:1/m^3/s=0[1/m^3/s]",
                )
                .replace("side=lower)=0[1/m^3]", "side=lower)=5[1/m^3]"),
            5.0,
        ),
    ] {
        let (model, document, plan) = select(&source).unwrap();
        let result = plan
            .as_scalar()
            .unwrap()
            .run_result(&REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let rules = HashMap::from([(
            document.aliases()["radius"].downcast().unwrap(),
            QuadratureRule::gauss_legendre(2).unwrap(),
        )]);
        let value = result
            .observe(
                &model,
                document.aliases()["average"].downcast().unwrap(),
                &rules,
            )
            .unwrap();
        assert!((value.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-10);
    }
}

#[test]
fn radial_operator_cannot_claim_positive_definiteness_after_conductance_underflow() {
    let source = SOURCE
        .replace(
            "diffusivity:m^2/s=1[m^2/s]",
            "diffusivity:m^2/s=5e-324[m^2/s]",
        )
        .replace(
            "production:1/m^3/s=6[1/m^3/s]",
            "production:1/m^3/s=0[1/m^3/s]",
        );
    let (_, _, plan) = select(&source).unwrap();
    let error = plan
        .as_scalar()
        .unwrap()
        .run_result(&eqiora::solver::REFERENCE_LINEAR_SOLVER)
        .unwrap_err();
    assert!(error.message().contains("no nonzero entries"), "{error:?}");
}

#[test]
fn radial_measure_scales_with_the_declared_radius() {
    use eqiora::meshing::QuadratureRule;
    use std::collections::HashMap;
    let source = SOURCE.replace("r=1[m]", "r=2[m]");
    let (model, document, plan) = select_document(document_with_radius(&source, 2.0)).unwrap();
    let result = plan
        .as_scalar()
        .unwrap()
        .run_result(&eqiora::solver::REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let rules = HashMap::from([(
        document.aliases()["radius"].downcast().unwrap(),
        QuadratureRule::gauss_legendre(2).unwrap(),
    )]);
    let value = result
        .observe(
            &model,
            document.aliases()["average"].downcast().unwrap(),
            &rules,
        )
        .unwrap();
    // c=R²-r² scales the unit-radius four-cell average by R².
    assert!((value.value().real_scalar_value().unwrap().value() - 113.0 / 64.0).abs() < 1e-10);
}

#[test]
fn radial_measure_rejects_unrepresentable_positive_cell_volumes() {
    let source = SOURCE
        .replace(
            "coordinate r:m",
            "parameter outer:m=1e-150[m]; coordinate r:m",
        )
        .replace("r=1[m]", "r=outer");
    let (_, _, plan) = select_document(document_with_radius(&source, 1e-150)).unwrap();
    let error = plan
        .as_scalar()
        .unwrap()
        .run_result(&eqiora::solver::REFERENCE_LINEAR_SOLVER)
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("positive finite radial cell measure"),
        "{error:?}"
    );
}
