use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};

use super::*;

// Euler operations below supply Primitive/Conservative states explicitly.
// This recognition fixture declares no fresh-initialization equations.
const SOURCE: &str = r#"
public operator scalar_flux(input value: scalar): spatial[1]
  = component(value);

model Main() {
  domain interval = box(0, 1);
  domain lower = boundary(interval, axis = 0, side = lower);
  domain upper = boundary(interval, axis = 0, side = upper);

  state density: kg / m ^ 3 on interval;
  state momentum: kg / (m ^ 2 * s) on interval;
  state total_energy: kg / (m * s ^ 2) on interval;
  variable velocity: m / s on interval;
  variable pressure: kg / (m * s ^ 2) on interval;
  parameter gamma: 1 = 1.4;

  relation velocity_definition on interval {
momentum - density * velocity = 0;
  }
  relation pressure_definition on interval {
pressure - (gamma - 1) * (total_energy - 0.5 * momentum * velocity) = 0;
  }
  relation mass on interval {
derivative(density) + div(scalar_flux(value = momentum)) = 0;
  }
  relation momentum_balance on interval {
derivative(momentum)
  + div(scalar_flux(value = momentum * velocity + pressure)) = 0;
  }
  relation energy on interval {
derivative(total_energy)
  + div(scalar_flux(value = velocity * (total_energy + pressure))) = 0;
  }
}
"#;

#[test]
fn recognizes_name_independent_exact_euler_meaning_and_lineage() {
    let model = recognize(SOURCE);
    assert_eq!(model.bounds(), [0.0, 1.0]);
    assert_eq!(model.boundaries().len(), 2);
    assert_eq!(model.conservative_fields().len(), 3);
    assert_eq!(model.primitive_fields().len(), 2);
    assert_eq!(model.balance_relations().len(), 3);
    assert_eq!(model.closure_relations().len(), 2);
    assert_eq!(model.flux_operator(), scalar_flux_lift().unwrap().digest());
    assert!(model.domain() != model.representation());
    assert!(model.gamma_parameter() != model.domain());
    assert_eq!(model.gamma(), 1.4);

    let renamed = SOURCE
        .replace("model Main", "model Renamed")
        .replace("density", "rho")
        .replace("momentum", "impulse")
        .replace("total_energy", "whole_energy")
        .replace("velocity", "speed")
        .replace("pressure", "thermodynamic_pressure")
        .replace("gamma", "heat_ratio")
        .replace("scalar_flux", "embed");
    let renamed = recognize(&renamed);
    assert_eq!(renamed.gamma(), model.gamma());

    let reordered = SOURCE.replace(
        "  state density: kg / m ^ 3 on interval;\n  state momentum: kg / (m ^ 2 * s) on interval;\n  state total_energy: kg / (m * s ^ 2) on interval;\n  variable velocity: m / s on interval;\n  variable pressure: kg / (m * s ^ 2) on interval;",
        "  variable pressure: kg / (m * s ^ 2) on interval;\n  variable velocity: m / s on interval;\n  state density: kg / m ^ 3 on interval;\n  state total_energy: kg / (m * s ^ 2) on interval;\n  state momentum: kg / (m ^ 2 * s) on interval;",
    );
    assert_ne!(reordered, SOURCE);
    assert_eq!(recognize(&reordered).gamma(), model.gamma());

    let different_initials = SOURCE
        .replace("kg / m ^ 3 = 1", "kg / m ^ 3 = 2")
        .replace(
            "kg / (m ^ 2 * s) = 0",
            "kg / (m ^ 2 * s) = 3[kg / (m ^ 2 * s)]",
        )
        .replace("kg / (m * s ^ 2) = 2.5", "kg / (m * s ^ 2) = 9");
    assert_eq!(recognize(&different_initials).gamma(), model.gamma());
}

#[test]
fn additive_orientation_does_not_change_admitted_equation_meaning() {
    let oriented = SOURCE
        .replace(
            "momentum - density * velocity = 0;",
            "momentum = density * velocity;",
        )
        .replace(
            "derivative(density) + div(scalar_flux(value = momentum)) = 0;",
            "derivative(density) = -div(scalar_flux(value = momentum));",
        )
        .replace(
            "derivative(total_energy)\n      + div(scalar_flux(value = velocity * (total_energy + pressure))) = 0;",
            "-derivative(total_energy)\n      - div(scalar_flux(value = velocity * (total_energy + pressure))) = 0;",
        );
    recognize(&oriented);
}

#[test]
fn converts_states_and_evaluates_flux_and_characteristic_bound() {
    let model = recognize(SOURCE);
    let primitive = EulerPrimitiveState1d::new(1.2, -2.5, 0.9);
    let conservative = model.primitive_to_conservative(primitive).unwrap();
    let round_trip = model.conservative_to_primitive(conservative).unwrap();
    for (actual, expected) in round_trip
        .components()
        .into_iter()
        .zip(primitive.components())
    {
        assert!((actual - expected).abs() < 16.0 * f64::EPSILON);
    }
    let flux = model.physical_flux(conservative).unwrap();
    assert!((flux[0] + 3.0).abs() < 16.0 * f64::EPSILON);
    assert!((flux[1] - 8.4).abs() < 16.0 * f64::EPSILON);
    assert!(model.sound_speed(conservative).unwrap() > 0.0);
    assert!(model.characteristic_speed_bound(conservative).unwrap() > 2.5);
    assert!(model.is_admissible(conservative));

    let overflowing_flux = model
        .primitive_to_conservative(EulerPrimitiveState1d::new(1.0e40, 1.0e100, 1.0e239))
        .unwrap();
    assert!(model.physical_flux(overflowing_flux).is_err());
}

#[test]
fn rejects_nonphysical_or_nonfinite_states() {
    let model = recognize(SOURCE);
    for state in [
        EulerConservativeState1d::new(0.0, 0.0, 1.0),
        EulerConservativeState1d::new(1.0, 2.0, 1.0),
        EulerConservativeState1d::new(1.0, f64::NAN, 3.0),
    ] {
        assert!(!model.is_admissible(state));
        assert!(model.physical_flux(state).is_err());
    }
    assert!(
        model
            .primitive_to_conservative(EulerPrimitiveState1d::new(1.0, 0.0, 0.0))
            .is_err()
    );
}

#[test]
fn rejects_wrong_closure_source_boundary_and_extra_relation() {
    let wrong_closure = SOURCE.replace("0.5 * momentum", "0.25 * momentum");
    assert!(try_recognize(&wrong_closure).is_err());

    let source_term = SOURCE.replace(
        "derivative(density) + div(scalar_flux(value = momentum)) = 0;",
        "derivative(density) + div(scalar_flux(value = momentum)) + derivative(density) = 0;",
    );
    assert!(try_recognize(&source_term).is_err());

    let boundary = SOURCE.replace("state density: kg / m ^ 3 on interval;", "state density: kg / m ^ 3 on interval in h1;").replace(
        "relation velocity_definition on interval {",
        "relation lower_law on lower { trace(density) = 0; }\n  relation velocity_definition on interval {",
    );
    assert!(try_recognize(&boundary).is_err());

    let extra = SOURCE.replace(
        "relation velocity_definition on interval {",
        "relation extra on interval { density - density = 0; }\n  relation velocity_definition on interval {",
    );
    assert!(try_recognize(&extra).is_err());

    let missing = SOURCE.replace(
        "  relation mass on interval {\nderivative(density) + div(scalar_flux(value = momentum)) = 0;\n  }\n",
        "",
    );
    assert!(try_recognize(&missing).is_err());

    let duplicate = SOURCE.replace(
        "  state momentum: kg / (m ^ 2 * s) on interval;",
        "  state momentum: kg / (m ^ 2 * s) on interval;\n  variable other_momentum: kg / (m ^ 2 * s) on interval;",
    );
    assert!(try_recognize(&duplicate).is_err());
}

#[test]
fn rejects_invalid_gamma_and_swapped_flux_lineage() {
    assert!(try_recognize(&SOURCE.replace("1.4", "1.0")).is_err());
    assert!(try_recognize(&SOURCE.replace("1.4", "0.5")).is_err());
    let swapped = SOURCE.replace(
        "scalar_flux(value = momentum * velocity + pressure)",
        "scalar_flux(value = momentum * velocity - pressure)",
    );
    assert!(try_recognize(&swapped).is_err());

    assert!(compile("nonfinite-gamma.eqi", &SOURCE.replace("1.4", "1e999")).is_err());
    assert!(
        compile(
            "wrong-density-dimension.eqi",
            &SOURCE.replace("density: kg / m ^ 3 on", "density: kg / m ^ 2 on"),
        )
        .is_err()
    );
}

fn recognize(source: &str) -> IdealGasEulerModel1d {
    try_recognize(source).expect("ordinary ideal-gas Euler source is recognized")
}

fn try_recognize(source: &str) -> Result<IdealGasEulerModel1d, Diagnostic> {
    let mut compiled = compile("ideal-gas-euler.eqi", source).expect("source compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    let program = KernelProgram::from_snapshot(&store.snapshot(), model)
        .expect("source commits as one Kernel program");
    recognize_ideal_gas_euler_1d(&program)
}
