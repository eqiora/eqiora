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
