//! Independent monomial antiderivatives for a bounded 1x1v density on an accepted Result.
#[path = "factor_integrals/coordinate_field.rs"]
mod coordinate_field;
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

fn compile(
    source: &str,
    velocity_bounds: [f64; 2],
) -> Result<CompiledModel, Vec<eqiora_core::Diagnostic>> {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    let interval = |lower, upper, unit| {
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(DynQuantity::new(lower, unit), DynQuantity::new(upper, unit)).unwrap(),
        )
    };
    CompiledModel::compile_selected(
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
}

fn model(source: &str, velocity_bounds: [f64; 2]) -> (ModelEnvelope, ModelSymbols) {
    let compiled = compile(source, velocity_bounds).unwrap();
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
    let plan = resolve(&model).unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    (model, symbols, result)
}

fn resolve(model: &ModelEnvelope) -> Result<CommonAlgebraicPlan, eqiora_core::Diagnostic> {
    CommonAlgebraicPlan::resolve(
        model,
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

#[test]
fn fixed_integral_partials_follow_remaining_factor_and_independent_parameter() {
    let source = SOURCE
        .replace("variable amplitude:s/m^2;", "parameter amplitude:s/m^2=3; variable anchor:1;")
        .replace("relation amplitude_value { amplitude=3[s/m^2]; }", "relation anchor_value { anchor=1; }")
        .replace("observable density:1/m", "coordinate retained_x:m on position from position[0];\n    observable density:1/m")
        .replace("observable mass:1=", "observable dx:1/m^2 on position=partial(density,wrt=retained_x);\n    observable da:m/s on position=partial(density,wrt=amplitude);\n    observable mass:1=");
    let source = source.replace(
        "observable composed:1=",
        "observable dm:m^2/s=partial(mass,wrt=amplitude);\n    observable composed:1=",
    );
    let original = model(&source, [-2.0, 4.0]).0.to_program().unwrap();
    let declaration = "observable density:1/m on position=integral(f,measure(velocity));";
    let reordered = source.replace(declaration, "");
    let (prefix, _) = reordered.rsplit_once('}').unwrap();
    let reordered = format!("{prefix}{declaration} }}");
    for equivalent in [
        source
            .replace("retained_x", "coordinate_alias")
            .replace("density", "number_density"),
        reordered,
    ] {
        let replay = model(&equivalent, [-2.0, 4.0]).0.to_program().unwrap();
        assert_eq!(
            eqiora_artifact::StructuralSemanticFingerprint::from_program(&original).unwrap(),
            eqiora_artifact::StructuralSemanticFingerprint::from_program(&replay).unwrap()
        );
    }
    let (model, symbols, result) = solve(&source, [-2.0, 4.0]);
    let model =
        ModelEnvelope::from_json(&model.canonical_json().unwrap(), Default::default()).unwrap();
    let full_rules = std::collections::HashMap::from([(
        symbols.get("phase").unwrap().downcast().unwrap(),
        QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap(),
    )]);
    let mass_partial = result
        .observe(
            &model,
            symbols.get("dm").unwrap().downcast().unwrap(),
            &full_rules,
        )
        .unwrap();
    // Integral over x in [0,2] of (15/2)*(1+x/2) is 45/2 m²/s.
    let total = mass_partial.value().real_scalar_value().unwrap();
    assert!((total.value() - 45.0 / 2.0).abs() <= 1e-11);
    assert_eq!(
        total.dim(),
        DimExponents::from_integers([0, 2, -1, 0, 0, 0, 0]).unwrap()
    );
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let rules = std::collections::HashMap::from([(
        symbols.get("velocity").unwrap().downcast().unwrap(),
        QuadratureRule::gauss_legendre(3).unwrap(),
    )]);
    // Independently, n(x,A)=A*(15/2)*(1+x/2) in coherent SI coordinates.
    // Therefore n_x=15A/4 and n_A=(15/2)*(1+x/2), at fixed velocity bounds.
    for x in [0.0, 1.0, 2.0] {
        for (name, expected, dimension) in [
            (
                "dx",
                45.0 / 4.0,
                DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).unwrap(),
            ),
            (
                "da",
                (15.0 / 2.0) * (1.0 + x / 2.0),
                DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap(),
            ),
        ] {
            let observed = result
                .observe_at(
                    &model,
                    symbols.get(name).unwrap().downcast().unwrap(),
                    &[DynQuantity::new(x, length)],
                    &rules,
                )
                .unwrap();
            let value = observed.value().real_scalar_value().unwrap();
            assert_eq!(value.dim(), dimension);
            assert!(
                (value.value() - expected).abs() <= 1e-11,
                "{name} at {x}: {}",
                value.value()
            );
        }
    }
}

#[test]
fn fixed_integral_partials_reject_moving_bounds_and_nonregular_density() {
    let bounded = "model Moving() { parameter length:m=2; domain body=box(0,length); variable anchor:1; relation value {anchor=1;} observable total:m=integral(1,measure(body)); }";
    eqiora_compiler::compile("moving.eqi", bounded).unwrap();
    let changed = bounded.replace(
        "observable total:m=",
        "observable derivative:1=partial(total,wrt=length); observable total:m=",
    );
    let errors = eqiora_compiler::compile("moving.eqi", &changed).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("moving bounds")),
        "{errors:?}"
    );
    let source = SOURCE
        .replace(
            "observable density:1/m",
            "coordinate retained_x:m on position from position[0]; observable density:1/m",
        )
        .replace(
            "observable mass:1=",
            "observable dx:1/m^2 on position=partial(density,wrt=retained_x); observable mass:1=",
        )
        .replace(
            "amplitude*(1+x/2[m])*(1+(v/4[m/s])^2)",
            "amplitude*(1+x/2[m])*(1[m/s]/v)",
        );
    let errors = compile(&source, [-2.0, 4.0]).unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("fixed nonzero real literal denominator")),
        "{errors:?}"
    );
}

#[test]
fn fixed_integral_partial_requires_spatial_field_regularity() {
    let source = SOURCE
        .replace(
            "variable amplitude:s/m^2;",
            "variable amplitude:s/m^2 on phase;",
        )
        .replace(
            "relation amplitude_value {",
            "relation amplitude_value on phase {",
        );
    compile(&source, [-2.0, 4.0]).unwrap();
    let differentiated = source.replace(
        "observable mass:1=",
        "coordinate retained_x:m on position from position[0]; observable dx:1/m^2 on position=partial(density,wrt=retained_x); observable mass:1=",
    );
    let errors = compile(&differentiated, [-2.0, 4.0]).unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("spatial Field requires admitted differentiation-under-integral regularity")),
        "{errors:?}"
    );
}

#[test]
fn fixed_integral_partial_rejects_an_integrated_coordinate_as_a_free_selector() {
    let source = SOURCE.replace("observable mass:1=", "coordinate bounded_v:m/s on velocity from velocity[0]; observable forbidden:s/m on velocity=partial(mass,wrt=bounded_v); observable mass:1=");
    let errors = compile(&source, [-2.0, 4.0]).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("integrated coordinates are bound")),
        "{errors:?}"
    );
}

#[test]
fn fixed_integral_partial_keeps_shared_literal_multipliers_and_divisors_distinct() {
    let source = SOURCE
        .replace(
            "let f:s/m^2",
            "let scale:m=2[m]; coordinate retained_x:m on position from position[0]; let f:s/m^2",
        )
        .replace("(1+x/2[m])", "(x/scale+scale*x/1[m^2])")
        .replace(
            "observable mass:1=",
            "observable dx:1/m^2 on position=partial(density,wrt=retained_x); observable mass:1=",
        );
    for (scale, expected) in [(2, 225.0 / 4.0), (-2, -225.0 / 4.0)] {
        let (model, symbols, result) = solve(
            &source.replace("scale:m=2[m]", &format!("scale:m={scale}[m]")),
            [-2.0, 4.0],
        );
        let rules = std::collections::HashMap::from([(
            symbols.get("velocity").unwrap().downcast().unwrap(),
            QuadratureRule::gauss_legendre(3).unwrap(),
        )]);
        let value = result
            .observe_at(
                &model,
                symbols.get("dx").unwrap().downcast().unwrap(),
                &[DynQuantity::new(
                    1.0,
                    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                )],
                &rules,
            )
            .unwrap();
        // d/dx [3*(x/2+2*x)*(15/2)] = (45/2)*(1/2+2) = 225/4.
        assert!((value.value().real_scalar_value().unwrap().value() - expected).abs() <= 1e-11);
    }
    let errors = compile(&source.replace("scale:m=2[m]", "scale:m=0[m]"), [-2.0, 4.0]).unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .message()
            .contains("fixed nonzero real literal denominator")),
        "{errors:?}"
    );
}

#[test]
fn spherical_density_integrals_replay_the_declared_radial_measure() {
    let source =
        "model Distribution(support position:interval(m), support velocity:interval(m/s)) {
        coordinate r:m on position from position[0];
        variable anchor:1; relation value {anchor=1;}
        let c:1/m^3=2[1/m^3]+3[1/m^5]*r^2;
        observable total:1=integral(c,spherical_measure(position));
        observable volume:m^3=integral(1,spherical_measure(position));
        observable average:1/m^3=total/volume;
        observable constant_total:1=integral(2[1/m^3],spherical_measure(position));
        observable constant_average:1/m^3=constant_total/volume;
        observable line:1/m^2=integral(c,measure(position));
    }";
    let (original, symbols, result) = solve(source, [-2.0, 4.0]);
    let bytes = original.canonical_json().unwrap();
    assert!(
        std::str::from_utf8(&bytes)
            .unwrap()
            .contains("spherical-volume-integral")
    );
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(
        ModelEnvelope::from_json(
            text.replace("eqiora.model-envelope/v41", "eqiora.model-envelope/v35")
                .as_bytes(),
            Default::default(),
        )
        .is_err()
    );
    let dropped = text.replace("spherical-volume-integral", "volume-integral");
    let errors = match ModelEnvelope::from_json(dropped.as_bytes(), Default::default()) {
        Ok(model) => model.to_program().unwrap_err(),
        Err(error) => vec![error],
    };
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("type") || error.message().contains("measure")),
        "{errors:?}"
    );
    let model = ModelEnvelope::from_json(&bytes, Default::default()).unwrap();
    let rules = std::collections::HashMap::from([(
        symbols.get("position").unwrap().downcast().unwrap(),
        QuadratureRule::gauss_legendre(3).unwrap(),
    )]);
    // Independently integrate (2+3r²)*4πr² on [0,2]:
    // total = 4π(2R³/3 + 3R⁵/5), volume = 4πR³/3.
    // Three-point Gauss is exact for the degree-four weighted polynomial.
    // The unweighted line result is 2R + R³, so omitting r² is observable.
    for (name, expected, length_power) in [
        ("total", 1472.0 * std::f64::consts::PI / 15.0, 0),
        ("volume", 32.0 * std::f64::consts::PI / 3.0, 3),
        ("average", 46.0 / 5.0, -3),
        ("constant_total", 64.0 * std::f64::consts::PI / 3.0, 0),
        ("constant_average", 2.0, -3),
        ("line", 12.0, -2),
    ] {
        let observed = result
            .observe(
                &model,
                symbols.get(name).unwrap().downcast().unwrap(),
                &rules,
            )
            .unwrap();
        let value = observed.value().real_scalar_value().unwrap();
        assert_eq!(
            value.dim(),
            DimExponents::from_integers([0, length_power, 0, 0, 0, 0, 0]).unwrap()
        );
        assert!(
            (value.value() - expected).abs() <= 1e-11,
            "{name}: {} expected {expected}",
            value.value()
        );
    }
}

#[path = "support/factor_integral_coupling.rs"]
mod model_coupling;

#[path = "factor_integrals/radial_diffusion.rs"]
mod radial_diffusion;

#[path = "factor_integrals/moving_endpoints.rs"]
mod moving_endpoints;

#[path = "factor_integrals/nonlocal.rs"]
mod nonlocal;
