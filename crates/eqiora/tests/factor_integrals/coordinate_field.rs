//! Independent cell antiderivatives exercised through public Model/Mesh/Plan/Result owners.
use super::*;
use eqiora_numerics::{AuthenticatedCommonMesh, CommonSpatialPolicy, ResolvedCommonPlan};
use std::collections::HashMap;

const FIELD_SOURCE: &str =
    include_str!("../../../../verify/language/factor-integrals/models/coordinate-field.eqi");

fn plan(
    model: &ModelEnvelope,
    mesh: AuthenticatedCommonMesh,
    spatial: CommonSpatialPolicy,
) -> Result<ResolvedCommonPlan, eqiora_core::Diagnostic> {
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::SparseLu,
            1e-12,
            1e-12,
            NonZeroUsize::new(10).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast),
        FaerLinearSolver.provider(),
    )
    .unwrap();
    eqiora_numerics::ResolvedCommonPlan::resolve(
        model,
        mesh,
        spatial,
        CommonSolvePolicy::Linear(linear),
        None,
        None,
        &FaerLinearSolver,
        None,
    )
}

#[test]
fn prescribed_phase_field_moments_conserve_mass_and_match_cell_antiderivatives() {
    let (original, symbols) = model(FIELD_SOURCE, [-2.0, 4.0]);
    let model =
        ModelEnvelope::from_json(&original.canonical_json().unwrap(), Default::default()).unwrap();
    let domain = |name| symbols.get(name).unwrap().downcast().unwrap();
    let observable = |name| symbols.get(name).unwrap().downcast().unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    for n in [1, 3, 6] {
        let mesh =
            AuthenticatedCommonMesh::coordinate_factors(&model, domain("phase"), &[2, n]).unwrap();
        let mesh = AuthenticatedCommonMesh::from_bytes(&mesh.to_bytes().unwrap()).unwrap();
        assert!(mesh.geometry().is_none());
        let selected = plan(&model, mesh, CommonSpatialPolicy::CellCentered).unwrap();
        let selected = ResolvedCommonPlan::from_bytes(
            &selected.to_bytes().unwrap(),
            &FaerLinearSolver,
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
        let solved = selected
            .as_scalar()
            .unwrap()
            .run_result(&FaerLinearSolver)
            .unwrap();
        let result = CommonResult::from_bytes(&solved.to_bytes().unwrap(), &selected).unwrap();
        let (association, coefficients, shape) = result.field_block(0, 0).unwrap();
        assert_eq!(association, "cell");
        assert_eq!(shape, [2, n]);
        assert_eq!(coefficients.len(), 2 * n);
        let velocity_rule = HashMap::from([(
            domain("velocity"),
            QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
        )]);
        let n2 = (n * n) as f64;
        let moments = [
            7.5,
            39.0 / 4.0 - 9.0 / (4.0 * n2),
            186.0 / 5.0 - 18.0 / n2 + 54.0 / (5.0 * n2 * n2),
        ];
        for (x, cell_x) in [(0.0, 0.5), (0.5, 0.5), (1.0, 1.5), (2.0, 1.5)] {
            for (order, name) in ["density", "current", "second"].into_iter().enumerate() {
                let value = result
                    .observe_at(
                        &model,
                        observable(name),
                        &[DynQuantity::new(x, length)],
                        &velocity_rule,
                    )
                    .unwrap();
                let quantity = value.value().real_scalar_value().unwrap();
                let exponent = order as i32;
                assert_eq!(
                    quantity.dim(),
                    DimExponents::from_integers([0, exponent - 1, -exponent, 0, 0, 0, 0]).unwrap()
                );
                assert!(
                    (quantity.value() - 3.0 * (1.0 + cell_x / 2.0) * moments[order]).abs() < 1e-10
                );
            }
            let mean = result
                .observe_at(
                    &model,
                    observable("mean"),
                    &[DynQuantity::new(x, length)],
                    &velocity_rule,
                )
                .unwrap();
            assert!(
                (mean.value().real_scalar_value().unwrap().value() - moments[1] / moments[0]).abs()
                    < 1e-10
            );
        }
        for (name, rules) in [
            (
                "mass",
                HashMap::from([(
                    domain("phase"),
                    QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap(),
                )]),
            ),
            (
                "nested",
                HashMap::from([
                    (
                        domain("position"),
                        QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
                    ),
                    (
                        domain("velocity"),
                        QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
                    ),
                ]),
            ),
        ] {
            let total = result.observe(&model, observable(name), &rules).unwrap();
            assert!((total.value().real_scalar_value().unwrap().value() - 67.5).abs() < 1e-10);
        }
        let h = 6.0 / n as f64;
        let v = -2.0 + h / 2.0;
        let radial = result
            .observe_at(
                &model,
                observable("radial"),
                &[DynQuantity::new(v, speed)],
                &HashMap::from([(
                    domain("position"),
                    QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
                )]),
            )
            .unwrap();
        // Weighted x-cell averages give 54*pi; the first v-cell mean-square is v_mid^2+h^2/12.
        let expected = 54.0 * std::f64::consts::PI * (1.0 + (v * v + h * h / 12.0) / 16.0);
        assert!((radial.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-10);
        assert!(
            result
                .observe_at(
                    &model,
                    observable("density"),
                    &[DynQuantity::new(0.5, speed)],
                    &velocity_rule
                )
                .is_err()
        );
    }
}

#[test]
fn coordinate_field_rejects_foreign_factor_units_support_and_methods() {
    let (model, symbols) = model(FIELD_SOURCE, [-2.0, 4.0]);
    let domain = |name| symbols.get(name).unwrap().downcast().unwrap();
    let grid =
        AuthenticatedCommonMesh::coordinate_factors(&model, domain("phase"), &[2, 3]).unwrap();
    assert!(
        plan(&model, grid.clone(), CommonSpatialPolicy::Q1)
            .unwrap_err()
            .message()
            .contains("CellCentered")
    );
    let wrong_support =
        AuthenticatedCommonMesh::coordinate_factors(&model, domain("velocity"), &[3]).unwrap();
    assert!(
        plan(&model, wrong_support, CommonSpatialPolicy::CellCentered)
            .unwrap_err()
            .message()
            .contains("exact Field and Relation support")
    );
    let bytes = String::from_utf8(grid.to_bytes().unwrap()).unwrap();
    let speed = "[[0,1],[1,1],[-1,1],[0,1],[0,1],[0,1],[0,1]]";
    let length = "[[0,1],[1,1],[0,1],[0,1],[0,1],[0,1],[0,1]]";
    assert!(bytes.contains(speed));
    let changed = bytes.replace(speed, length);
    // The independently canonical artifact remains a valid unit-bearing grid, but not this Model's grid.
    let substituted = AuthenticatedCommonMesh::from_bytes(changed.as_bytes()).unwrap();
    assert_eq!(substituted.cartesian_mesh(), grid.cartesian_mesh());
    assert_ne!(
        substituted.source_digest().unwrap(),
        grid.source_digest().unwrap()
    );
    assert!(
        plan(&model, substituted, CommonSpatialPolicy::CellCentered)
            .unwrap_err()
            .message()
            .contains("units or bounds")
    );
}
