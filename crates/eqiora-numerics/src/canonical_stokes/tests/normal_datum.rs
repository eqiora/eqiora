//! Native admission retains boundary orientation and optional authored carriers.
use super::*;
use crate::canonical_boundary::PhysicalBoundarySupport;
use crate::canonical_stokes::SteadyIncompressibleStokesModel2d;
use eqiora_core::{Id, RawId, entity::kinds};

struct Fixture {
    model: SteadyIncompressibleStokesModel2d,
    boundary: RawId,
    parameters: Vec<Id<kinds::Parameter>>,
    carrier: Option<(RawId, RawId)>,
}

fn source(carrier: bool, profile: &str) -> String {
    let fields = if carrier {
        "variable inlet_profile:m/s on fluid in h1;"
    } else {
        ""
    };
    let definition = if carrier {
        format!("relation inlet_definition on fluid{{inlet_profile-({profile})=0;}} ")
    } else {
        String::new()
    };
    let rhs = if carrier {
        "normal(isotropic_lift(inlet_profile))".to_owned()
    } else {
        format!("normal_vector()*({profile})")
    };
    SOURCE.replace("  parameter mu:",&format!("{fields} parameter inlet_speed:m/s=2; parameter height:m=2; {definition} parameter mu:"))
        .replace("relation x_lower_zero on x_lower { trace(velocity) = 0; }",&format!("relation x_lower_zero on x_lower {{ trace(velocity)=-({rhs}); }}"))
        .replace("relation x_upper_zero on x_upper { trace(velocity) = 0; }",&format!("relation x_upper_zero on x_upper {{ trace(velocity)={rhs}; }}"))
}

fn fixture(carrier: bool, profile: &str) -> Fixture {
    let (transaction, id, symbols) = compile("native-normal.eqi", &source(carrier, profile))
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), id).unwrap();
    let mut parameters = vec![
        symbols.get("inlet_speed").unwrap().downcast().unwrap(),
        symbols.get("height").unwrap().downcast().unwrap(),
    ];
    parameters.sort_by_key(Id::ulid);
    Fixture {
        model: lower_steady_incompressible_stokes_cartesian_2d(&program).unwrap(),
        boundary: symbols.get("x_lower").unwrap(),
        parameters,
        carrier: carrier.then(|| {
            (
                symbols.get("inlet_profile").unwrap(),
                symbols.get("inlet_definition").unwrap(),
            )
        }),
    }
}

fn close(actual: [f64; 2], expected: [f64; 2]) {
    for (a, b) in actual.into_iter().zip(expected) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
}

#[test]
fn parameter_only_and_coordinate_profiles_keep_nonaxis_normal_values_and_provenance() {
    for (profile, expected) in [
        ("inlet_speed*height/height", 2.),
        (
            "4*inlet_speed*(coordinate(1)+1[m])*(1[m]-coordinate(1))/height^2",
            1.5,
        ),
    ] {
        for carrier in [false, true] {
            let fixture = fixture(carrier, profile);
            let trace = fixture
                .model
                .prescribed_velocity_trace(&PhysicalBoundarySupport::CartesianSide {
                    axis: 0,
                    side: BoundarySide::Lower,
                })
                .unwrap();
            assert_eq!(trace.boundary(), fixture.boundary);
            assert_eq!(
                trace.coefficient_field().zip(trace.definition_relation()),
                fixture.carrier
            );
            let parameter_slices = fixture
                .model
                .normal_velocity_parameters()
                .collect::<Vec<_>>();
            assert_eq!(
                parameter_slices.len(),
                2,
                "both authored prescribed sides retain their inputs"
            );
            for parameters in parameter_slices {
                assert_eq!(parameters, fixture.parameters);
            }
            // The retained operator consumes the realization's whole supplied normal;
            // this is orientation/evaluation evidence, not a rotated-geometry solve.
            close(
                trace.value(Some([0.6, 0.8]), &[0., 0.5]).unwrap(),
                [-0.6 * expected, -0.8 * expected],
            );
            close(
                trace.value(Some([-0.6, -0.8]), &[0., 0.5]).unwrap(),
                [0.6 * expected, 0.8 * expected],
            );
        }
    }
}

#[test]
fn unknown_carrier_reaches_its_definition_gate_without_becoming_prescribed() {
    let source = source(true, "inlet_speed*height/height").replace(
        "relation inlet_definition on fluid{inlet_profile-(inlet_speed*height/height)=0;}",
        "",
    );
    let error =
        lower_steady_incompressible_stokes_cartesian_2d(&compile_program(&source)).unwrap_err();
    assert!(
        error
            .message()
            .contains("unknown-dependent prescribed datum"),
        "{error:?}"
    );
}

#[test]
fn transient_three_dimensional_normal_data_never_passes_through_a_two_component_tape() {
    for carrier in [false, true] {
        let fields = if carrier {
            "variable inlet_profile:m/s on fluid in h1; relation inlet_definition on fluid{inlet_profile-inlet_speed=0;}"
        } else {
            ""
        };
        let rhs = if carrier {
            "normal(isotropic_lift(inlet_profile))"
        } else {
            "normal_vector()*inlet_speed"
        };
        let source = transient_navier_stokes_source_3d()
            .replace(
                "  parameter mu:",
                &format!("parameter inlet_speed:m/s=2; {fields} parameter mu:"),
            )
            .replace(
                "relation x_lower_zero on x_lower { trace(velocity) = 0; }",
                &format!("relation x_lower_zero on x_lower{{trace(velocity)=-({rhs});}}"),
            )
            .replace(
                "relation x_upper_zero on x_upper { trace(velocity) = 0; }",
                &format!("relation x_upper_zero on x_upper{{trace(velocity)={rhs};}}"),
            );
        let model =
            lower_transient_incompressible_navier_stokes_cartesian::<3>(&compile_program(&source))
                .unwrap();
        let boundary = model
            .cartesian_boundary(0, BoundarySide::Lower)
            .unwrap()
            .boundary();
        let actual = model
            .prescribed_normal_velocity(boundary, [0., 0.6, 0.8], &[0., 0.5, 0.25])
            .unwrap()
            .unwrap();
        for (a, b) in actual.into_iter().zip([0., -1.2, -1.6]) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(
            model
                .prescribed_cartesian_normal_velocity(0, BoundarySide::Lower, &[0., 0.5, 0.25])
                .unwrap(),
            Some([2., 0., 0.])
        );
    }
}

#[test]
fn pressure_force_field_cannot_be_reclassified_as_a_velocity_carrier() {
    // The conversion gives the RHS correct velocity units, so this must reach
    // the carrier-role gate rather than fail ordinary dimensional typing.
    let source=SOURCE.replace("  parameter mu:","parameter conversion:m^2*s/kg=1; parameter mu:")
        .replace("relation x_lower_zero on x_lower { trace(velocity) = 0; }","relation x_lower_zero on x_lower{trace(velocity)=-normal_vector()*force_potential*conversion;}");
    let program = compile_program(&source);
    let error = lower_steady_incompressible_stokes_cartesian_2d(&program).unwrap_err();
    assert!(
        error.message().contains("real invariant velocity scalar"),
        "{error:?}"
    );
}

fn complete_vector_source(speed: f64) -> String {
    SOURCE
        .replace("  parameter mu:", &format!("parameter speed:m/s={speed}; parameter other_speed:m/s={speed}; parameter mu:"))
        .replace("relation x_lower_zero on x_lower { trace(velocity) = 0; }", "relation x_lower_zero on x_lower { trace(velocity)=speed*tensor_value(frame=fluid,components=[1,0]); }")
        .replace("relation x_upper_zero on x_upper { trace(velocity) = 0; }", "relation x_upper_zero on x_upper { trace(velocity)=other_speed*tensor_value(frame=fluid,components=[1,0]); }")
}

#[test]
fn complete_vector_datum_needs_no_normal_and_keeps_equal_valued_parameter_identity() {
    for speed in [0.7, 1.3] {
        let (transaction, id, symbols) =
            compile("complete-datum.eqi", &complete_vector_source(speed))
                .unwrap()
                .remove(0)
                .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), id).unwrap();
        let model = lower_steady_incompressible_stokes_cartesian_2d(&program).unwrap();
        let lower = model
            .prescribed_velocity_trace(&PhysicalBoundarySupport::CartesianSide {
                axis: 0,
                side: BoundarySide::Lower,
            })
            .unwrap();
        let upper = model
            .prescribed_velocity_trace(&PhysicalBoundarySupport::CartesianSide {
                axis: 0,
                side: BoundarySide::Upper,
            })
            .unwrap();
        assert!(!lower.requires_normal());
        assert_eq!(lower.boundary(), symbols.get("x_lower").unwrap());
        assert_eq!(lower.relation(), symbols.get("x_lower_zero").unwrap());
        assert_eq!(lower.coefficient_field(), None);
        close(lower.value(None, &[0., 0.5]).unwrap(), [speed, 0.]);
        close(upper.value(None, &[2., 0.5]).unwrap(), [speed, 0.]);
        assert!(
            !lower.same_law(upper),
            "equal values must not erase distinct Parameter IDs"
        );
        let inputs = model
            .normal_velocity_parameters()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        assert!(inputs.contains(&symbols.get("speed").unwrap().downcast().unwrap()));
        assert!(inputs.contains(&symbols.get("other_speed").unwrap().downcast().unwrap()));
    }
}

#[test]
fn prescribed_carrier_cannot_move_off_the_exact_parent_domain() {
    let source = source(true, "inlet_speed*height/height").replace(
        "variable inlet_profile:m/s on fluid",
        "variable inlet_profile:m/s on x_lower",
    );
    let diagnostics = compile("foreign-carrier.eqi", &source).unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|error| error.message().contains("differs from Relation scope")),
        "{diagnostics:?}"
    );
}

#[test]
fn complete_vector_datum_rejects_a_singular_coordinate_profile() {
    let source = complete_vector_source(0.7).replace(
        "=speed*tensor_value",
        "=(speed/(coordinate(1)/1[m]))*tensor_value",
    );
    let model = lower_steady_incompressible_stokes_cartesian_2d(&compile_program(&source)).unwrap();
    let key = PhysicalBoundarySupport::CartesianSide {
        axis: 0,
        side: BoundarySide::Lower,
    };
    assert!(model.prescribed_velocity(&key, None, &[0., 0.]).is_err());
    close(
        model
            .prescribed_velocity(&key, None, &[0., 0.5])
            .unwrap()
            .unwrap(),
        [1.4, 0.],
    );
}

#[test]
fn normal_datum_still_requires_an_authenticated_normal() {
    let fixture = fixture(false, "inlet_speed*height/height");
    let trace = fixture
        .model
        .prescribed_velocity_trace(&PhysicalBoundarySupport::CartesianSide {
            axis: 0,
            side: BoundarySide::Lower,
        })
        .unwrap();
    assert!(trace.requires_normal());
    assert!(
        trace
            .value(None, &[0., 0.5])
            .unwrap_err()
            .message()
            .contains("parent-outward normal")
    );
}

#[test]
fn complete_vector_datum_does_not_expand_transient_flow_admission() {
    let source = TRANSIENT_NAVIER_STOKES_SOURCE
        .replace("  parameter mu:", "parameter speed:m/s=0.7; parameter mu:")
        .replace("relation x_lower_zero on x_lower { trace(velocity) = 0; }", "relation x_lower_zero on x_lower { trace(velocity)=speed*tensor_value(frame=fluid,components=[1,0]); }");
    let error =
        lower_transient_incompressible_navier_stokes_cartesian::<2>(&compile_program(&source))
            .unwrap_err();
    assert!(
        error.message().contains("exact scoped boundary normal"),
        "{error:?}"
    );
}

fn complete_vector_expression_source(parameters: &[(&str, &str)], residual: &str) -> String {
    let declarations = parameters
        .iter()
        .map(|(name, value)| format!("parameter {name}:m/s={value};"))
        .collect::<String>();
    complete_vector_source(0.7)
        .replace("parameter speed:m/s=0.7;", &declarations)
        .replace(
            "relation x_lower_zero on x_lower { trace(velocity)=speed*tensor_value(frame=fluid,components=[1,0]); }",
            &format!("relation x_lower_zero on x_lower {{ {residual}; }}"),
        )
}

fn grouped_complete_vector_value(values: [&str; 3], datum: &str) -> [f64; 2] {
    let parameters = [("a", values[0]), ("b", values[1]), ("c", values[2])];
    let source =
        complete_vector_expression_source(&parameters, &format!("trace(velocity)={datum}"));
    let (transaction, id, symbols) = compile("grouped-complete-datum.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), id).unwrap();
    let model = lower_steady_incompressible_stokes_cartesian_2d(&program).unwrap();
    let retained = model
        .normal_velocity_parameters()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    // Keep each speed as a distinct dimensional Parameter, consumed by the
    // retained datum, rather than an arithmetic literal folded by the compiler.
    for (name, _) in parameters {
        let parameter: Id<kinds::Parameter> = symbols.get(name).unwrap().downcast().unwrap();
        assert!(
            retained.contains(&parameter),
            "datum omitted Parameter {name}"
        );
    }
    model
        .prescribed_velocity(
            &PhysicalBoundarySupport::CartesianSide {
                axis: 0,
                side: BoundarySide::Lower,
            },
            None,
            &[0., 0.5],
        )
        .unwrap()
        .unwrap()
}

#[test]
fn complete_vector_datum_keeps_nested_addition_and_subtraction_grouping() {
    let vector = "tensor_value(frame=fluid,components=[1,0])";
    for (datum, expected) in [
        (format!("a*{vector}-(b*{vector}-c*{vector})"), 6.),
        (format!("(a*{vector}-b*{vector})-c*{vector}"), 2.),
        (format!("a*{vector}-(b*{vector}+c*{vector})"), 2.),
        (format!("(a*{vector}-b*{vector})+c*{vector}"), 6.),
    ] {
        assert_eq!(
            grouped_complete_vector_value(["7", "3", "2"], &datum),
            [expected, 0.]
        );
    }
}

#[test]
fn complete_vector_datum_does_not_reassociate_finite_nonassociative_addition() {
    let vector = "tensor_value(frame=fluid,components=[1,0])";
    let datum = format!("a*{vector}+(b*{vector}+c*{vector})");
    // Binary64: b+c rounds back to b, so the authored a+(b+c) is zero.
    // Reassociating as (a+b)+c would instead produce one.
    let actual = grouped_complete_vector_value(["1e308", "-1e308", "1"], &datum);
    assert_eq!(actual, [0., 0.]);
    assert_ne!(actual[0], 1.);
}

#[test]
fn complete_vector_datum_preserves_grouping_that_avoids_intermediate_overflow() {
    let vector = "tensor_value(frame=fluid,components=[1,0])";
    let datum = format!("a*{vector}+(b*{vector}+c*{vector})");
    // The authored inner sum is zero. A left fold would overflow at a+b.
    let actual = grouped_complete_vector_value(["1e308", "1e308", "-1e308"], &datum);
    assert_eq!(actual, [1e308, 0.]);
    assert!(actual.into_iter().all(f64::is_finite));
}

#[test]
fn complete_vector_datum_rejects_duplicate_target_trace_and_interleaved_terms() {
    let vector = "tensor_value(frame=fluid,components=[1,0])";
    for residual in [
        format!("trace(velocity)+trace(velocity)=a*{vector}+b*{vector}"),
        format!("(trace(velocity)-a*{vector})+b*{vector}=0"),
    ] {
        let source = complete_vector_expression_source(&[("a", "1"), ("b", "2")], &residual);
        assert!(
            lower_steady_incompressible_stokes_cartesian_2d(&compile_program(&source)).is_err(),
            "unsupported target/datum grouping was admitted: {residual}"
        );
    }
}
