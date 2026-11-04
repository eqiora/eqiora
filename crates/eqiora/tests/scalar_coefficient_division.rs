mod support;

use eqiora::api::DifferentiableProgram;
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::{CommonSpatialPolicy, ResolvedCommonPlan};
use support::derivative_products::FixtureCoordinates;

const SOURCE: &str = r#"
public component DifferentiatedPoisson(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
  parameter diffusion: 1,
  parameter wave_number: m,
  parameter source_scale: K,
  parameter boundary_offset: K
) {
  variable potential: K on square in h1;
  relation balance on square {
    -div(grad(potential)) = -6 * source_scale * coordinate(0) / wave_number ^ 3;
  }
  relation x_lower_value on x_lower { trace(potential) = source_scale * (coordinate(0) / wave_number) ^ 3; }
  relation x_upper_value on x_upper { trace(potential) = source_scale * (coordinate(0) / wave_number) ^ 3; }
  relation y_lower_value on y_lower { trace(potential) = source_scale * (coordinate(0) / wave_number) ^ 3; }
  relation y_upper_value on y_upper { trace(potential) = source_scale * (coordinate(0) / wave_number) ^ 3; }
}
"#;

#[test]
fn independent_coefficient_division_solves_and_rebinds_cubic_temperature() {
    let (document, plan) = support::common_scalar_plan::document_and_plan_with_source(
        CommonSpatialPolicy::Q1,
        SOURCE,
    );
    let output = document.field_ref(&plan.fields().next().unwrap().0.ulid().to_string()).unwrap();
    let program = DifferentiableProgram::compile(
        ResolvedCommonPlan::Spatial(Box::new(plan)),
        &[document.parameter_ref("source_scale").unwrap(), document.parameter_ref("wave_number").unwrap()],
        &[output.erase()], Default::default(), None, &REFERENCE_LINEAR_SOLVER,
    ).unwrap();
    let values = program.primal().output().fixture_coordinates();
    assert_eq!(values.len(), 169);
    // -Delta(A*x^3/L^3) = -6*A*x/L^3; Q1 on this tensor grid reproduces
    // its nodal values with exact integration of the linear source.
    for (index, value) in values.iter().enumerate() {
        let x = (index % 13) as f64 / 12.0;
        let expected = 2.0 * x.powi(3) / std::f64::consts::PI;
        assert!((value - expected).abs() < 1e-10, "{index}: {value} != {expected}");
    }
}
