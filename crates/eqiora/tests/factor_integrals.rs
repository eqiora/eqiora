//! Independent monomial antiderivatives for a bounded 1x1v density on an accepted Result.
#[path = "support/factor_integral_model.rs"]
mod model_replay;

use eqiora_artifact::ModelEnvelope;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_compiler::{CompiledModel, ModelSymbols, StaticBindingValue};
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::QuadratureRule;
use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonResult, CommonSolvePolicy};
use eqiora_schema::kernel::AxisBounds;
use eqiora_solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use std::num::NonZeroUsize;

const SOURCE: &str =
    include_str!("../../../verify/language/factor-integrals/models/distribution.eqi");

fn model(source: &str, velocity_bounds: [f64; 2]) -> (ModelEnvelope, ModelSymbols) {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let interval = |lower, upper, unit| {
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(DynQuantity::new(lower, unit), DynQuantity::new(upper, unit)).unwrap(),
        )
    };
    let compiled = CompiledModel::compile_selected(
        "moments.eqi",
        source,
        "Distribution",
        &[
            ("position", interval(0.0, 2.0, length)),
            (
                "velocity",
                interval(velocity_bounds[0], velocity_bounds[1], speed),
            ),
        ],
    )
    .unwrap();
    let symbols = compiled.symbols().clone();
    let (transaction, model_id, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = eqiora_sem::KernelProgram::from_snapshot(&store.snapshot(), model_id).unwrap();
    let model = ModelEnvelope::from_program(&program).unwrap();
    (model, symbols)
}

fn solve(source: &str, velocity_bounds: [f64; 2]) -> (ModelEnvelope, ModelSymbols, CommonResult) {
    let (model, symbols) = model(source, velocity_bounds);
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(
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
            .unwrap(),
        ),
        None,
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    (model, symbols, result)
}

#[test]
fn factor_integrals_evaluate_density_moments_and_composition_on_an_accepted_result() {
    let (model, symbols, result) = solve(SOURCE, [-2.0, 4.0]);
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let domain = |name| symbols.get(name).unwrap().downcast().unwrap();
    let observable = |name| symbols.get(name).unwrap().downcast().unwrap();
    let velocity_rule = std::collections::HashMap::from([(
        domain("velocity"),
        QuadratureRule::gauss_legendre(3).unwrap(),
    )]);
    // At x=1 m, f=(9/2)+(9/32)v² in SI units. Integrate v^k f on [-2,4]
    // by [v^(p+1)/(p+1)] at both endpoints, independently of numerical output.
    for (name, expected) in [
        ("density", 135.0 / 4.0),
        ("current", 351.0 / 8.0),
        ("second", 837.0 / 5.0),
        ("mean", 13.0 / 10.0),
    ] {
        let observation = result
            .observe_at(
                &model,
                observable(name),
                &[DynQuantity::new(1.0, length)],
                &velocity_rule,
            )
            .unwrap();
        let actual = observation.value().real_scalar_value().unwrap().value();
        assert!((actual - expected).abs() < 1e-11, "{name}: {actual}");
        assert_eq!(observation.point().unwrap().0, domain("position"));
        assert!(
            result
                .observe(&model, observable(name), &velocity_rule)
                .is_err()
        );
    }
    let full = std::collections::HashMap::from([(
        domain("phase"),
        QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap(),
    )]);
    let mut composed = velocity_rule.clone();
    composed.insert(
        domain("position"),
        QuadratureRule::gauss_legendre(2).unwrap(),
    );
    for (name, rules) in [("mass", &full), ("composed", &composed)] {
        let observation = result.observe(&model, observable(name), rules).unwrap();
        // Integral of (45/2)(1+x/2) over [0,2] is 135/2.
        assert!(
            (observation.value().real_scalar_value().unwrap().value() - 135.0 / 2.0).abs() < 1e-11
        );
        assert!(observation.point().is_none());
    }
    for coordinates in [
        vec![],
        vec![DynQuantity::new(1.0, speed)],
        vec![DynQuantity::new(3.0, length)],
    ] {
        assert!(
            result
                .observe_at(&model, observable("density"), &coordinates, &velocity_rule)
                .is_err()
        );
    }
}

#[test]
fn factor_integrals_gaussian_truncation_has_a_separate_quadrature_error_bound() {
    let source = SOURCE.replace(
        "amplitude*(1+x/2[m])*(1+(v/4[m/s])^2)",
        "amplitude*math.exp(-0.5*(v/1[m/s])^2)",
    );
    let (model, symbols, result) = solve(&source, [-3.0, 3.0]);
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let cells = 1024u32;
    let axis = QuadratureRule::gauss_legendre(2).unwrap();
    let points = (0..cells)
        .flat_map(|cell| {
            axis.points()
                .iter()
                .map(move |point| eqiora_meshing::QuadraturePoint {
                    coordinates: vec![
                        -1.0 + (2.0 * f64::from(cell) + 1.0 + point.coordinates[0])
                            / f64::from(cells),
                    ],
                    weight: point.weight / f64::from(cells),
                })
        })
        .collect();
    let rule = QuadratureRule::new(
        eqiora_meshing::ReferenceCell::hypercube(1).unwrap(),
        Some(3),
        points,
    )
    .unwrap();
    let rules = std::collections::HashMap::from([(
        symbols.get("velocity").unwrap().downcast().unwrap(),
        rule,
    )]);
    // Independent analytic moments on [-3,3]: I0=sqrt(2π) erf(3/sqrt(2)),
    // I1=0, I2=I0-6 exp(-9/2). Values below come from these formulas, not quadrature.
    let expected = [2.499_860_889_483_094_7, 0.0, 2.433_206_910_253_640_7];
    // For g=exp(-v²/2), fourth derivatives of g, vg and v²g are g times
    // v⁴-6v²+3; v⁵-10v³+15v; v⁶-14v⁴+39v²-12, respectively.
    // |v|<=3 and g<=1 give M4 <= 138, 558, 2226. Two-point Gauss on
    // each cell has error <= h⁵ M4/4320, so the length-six composite error
    // for density 3g is <= 3*6*h⁴*M4/4320. 1e-12 covers floating-point rounding.
    let h = 6.0 / f64::from(cells);
    for ((name, exact), fourth_bound) in ["density", "current", "second"]
        .into_iter()
        .zip(expected)
        .zip([138.0, 558.0, 2226.0])
    {
        let tolerance = 3.0 * 6.0 * h.powi(4) * fourth_bound / 4320.0 + 1e-12;
        let observation = result
            .observe_at(
                &model,
                symbols.get(name).unwrap().downcast().unwrap(),
                &[DynQuantity::new(1.0, length)],
                &rules,
            )
            .unwrap();
        let actual = observation.value().real_scalar_value().unwrap().value();
        assert!(
            (actual - 3.0 * exact).abs() <= tolerance,
            "{name}: {actual}, tolerance {tolerance}"
        );
        if name != "current" {
            // Integration by parts gives tail0 < 2 exp(-a²/2)/a and
            // tail2 = 2a exp(-a²/2)+tail0. This is distinct from quadrature error.
            let infinite = 3.0 * std::f64::consts::TAU.sqrt();
            let tail_bound = 6.0
                * (-4.5f64).exp()
                * (if name == "density" {
                    1.0 / 3.0
                } else {
                    3.0 + 1.0 / 3.0
                });
            assert!(infinite - actual > 100.0 * tolerance);
            assert!(infinite - actual < tail_bound + tolerance);
        }
    }
}

#[test]
fn factor_integrals_require_explicit_nonzero_normalization_and_exact_rules() {
    let (model, symbols, result) =
        solve(&SOURCE.replace("amplitude=3", "amplitude=0"), [-2.0, 4.0]);
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let density = symbols.get("density").unwrap().downcast().unwrap();
    let mean = symbols.get("mean").unwrap().downcast().unwrap();
    let velocity = symbols.get("velocity").unwrap().downcast().unwrap();
    let point = [DynQuantity::new(1.0, length)];
    let rules =
        std::collections::HashMap::from([(velocity, QuadratureRule::gauss_legendre(3).unwrap())]);
    assert_eq!(
        result
            .observe_at(&model, density, &point, &rules)
            .unwrap()
            .value()
            .real_scalar_value()
            .unwrap()
            .value(),
        0.0
    );
    assert!(result.observe_at(&model, mean, &point, &rules).is_err());
    let wrong = std::collections::HashMap::from([(
        velocity,
        QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap(),
    )]);
    assert!(
        result
            .observe_at(&model, density, &point, &wrong)
            .unwrap_err()
            .message()
            .contains("quadrature dimension")
    );
    let mut unused = rules;
    unused.insert(
        symbols.get("position").unwrap().downcast().unwrap(),
        QuadratureRule::gauss_legendre(2).unwrap(),
    );
    assert!(
        result
            .observe_at(&model, density, &point, &unused)
            .unwrap_err()
            .message()
            .contains("unused integration Domain")
    );
}

#[test]
fn finite_plan_does_not_erase_spatial_unknowns_or_equation_support() {
    for source in [
        SOURCE
            .replace(
                "variable amplitude:s/m^2;",
                "variable amplitude:s/m^2 on phase;",
            )
            .replace(
                "relation amplitude_value {",
                "relation amplitude_value on phase {",
            ),
        SOURCE
            .replace(
                "relation amplitude_value {",
                "relation amplitude_value on phase {",
            )
            .replace("amplitude=3", "amplitude+0[s/m^3]*x=3"),
    ] {
        let (model, _) = model(&source, [-2.0, 4.0]);
        let request = CommonLinearRequest::exact(
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
        let error = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(request),
            None,
            None,
            &FaerLinearSolver,
        )
        .unwrap_err();
        assert!(error.message().contains("spatial support"), "{error:?}");
    }
}

#[test]
fn factor_integrals_do_not_accept_poles_missed_by_quadrature_samples() {
    for density in ["amplitude*(1[m/s]/v)", "amplitude*(1[m]/x)"] {
        let source = SOURCE.replace("amplitude*(1+x/2[m])*(1+(v/4[m/s])^2)", density);
        let (model, symbols, result) = solve(&source, [-2.0, 4.0]);
        let rules = std::collections::HashMap::from([(
            symbols.get("velocity").unwrap().downcast().unwrap(),
            QuadratureRule::gauss_legendre(2).unwrap(),
        )]);
        let point = [DynQuantity::new(
            1.0,
            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
        )];
        let error = result
            .observe_at(
                &model,
                symbols.get("density").unwrap().downcast().unwrap(),
                &point,
                &rules,
            )
            .unwrap_err();
        assert!(
            error.message().contains("regular factor density"),
            "{error:?}"
        );
    }
}

#[test]
fn factor_integrals_retain_support_for_a_constant_declared_density() {
    let source = SOURCE.replace(
        "let f:s/m^2 on phase=amplitude*(1+x/2[m])*(1+(v/4[m/s])^2);",
        "observable f:s/m^2 on phase=amplitude;",
    );
    let (model, symbols, result) = solve(&source, [-2.0, 4.0]);
    let velocity = symbols.get("velocity").unwrap().downcast().unwrap();
    let rules =
        std::collections::HashMap::from([(velocity, QuadratureRule::gauss_legendre(1).unwrap())]);
    let point = [DynQuantity::new(
        1.0,
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
    )];
    let value = result
        .observe_at(
            &model,
            symbols.get("density").unwrap().downcast().unwrap(),
            &point,
            &rules,
        )
        .unwrap();
    assert_eq!(value.value().real_scalar_value().unwrap().value(), 18.0);
}
