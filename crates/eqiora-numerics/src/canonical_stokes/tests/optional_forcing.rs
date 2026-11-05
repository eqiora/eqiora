//! Authored source presence, physical zero, and saved-model provenance.
use super::*;

#[test]
fn omitted_steady_force_has_no_source_identity_and_matches_physical_zero() {
    // An authored source cannot become absent merely because its gradient was dropped.
    let error = lower_steady_incompressible_stokes_cartesian_2d(&compile_program(
        &SOURCE.replace(" - grad(force_potential)", ""),
    ))
    .unwrap_err();
    assert!(error.message().contains("additional pressure-valued Field"));
    let explicit_zero = SOURCE.replace(
        "load_scale: kg / (m * s ^ 2) = 3",
        "load_scale: kg / (m * s ^ 2) = 0",
    );
    let omitted = SOURCE
        .replace("  variable force_potential: kg / (m * s ^ 2) on fluid;\n", "")
        .replace("  parameter load_scale: kg / (m * s ^ 2) = 3;\n", "")
        .replace("  parameter length_scale: m = 2;\n", "")
        .replace("  relation force on fluid {\n    force_potential - load_scale * coordinate(0) / length_scale = 0;\n  }\n", "")
        .replace(" - grad(force_potential)", "");
    let control =
        lower_steady_incompressible_stokes_cartesian_2d(&compile_program(&explicit_zero)).unwrap();
    let absent =
        lower_steady_incompressible_stokes_cartesian_2d(&replayed_program(&omitted)).unwrap();
    assert_eq!(absent.force_potential(), None);
    assert_eq!(absent.force_potential_definition(), None);
    assert_eq!(absent.force_potential_expression(), None);
    absent.replay_correspondence().unwrap();
    assert!(
        !absent.correspondence().entries.iter().any(|entry| matches!(
            entry.role,
            crate::form_compiler::vocabulary::MixedTermRole::SourceDefinition
                | crate::form_compiler::vocabulary::MixedTermRole::MomentumBodySource
        ))
    );
    let scales = physical_scales(1.);
    let nonzero =
        lower_steady_incompressible_stokes_cartesian_2d(&replayed_program(SOURCE)).unwrap();
    for point in [[0.2, 0.3], [0.8, 0.7]] {
        assert_eq!(
            crate::canonical_stokes::realization::dimensionless_body_force(&absent, scales)(point)
                .unwrap(),
            [0., 0.]
        );
        assert_eq!(
            crate::canonical_stokes::realization::dimensionless_body_force(&control, scales)(point)
                .unwrap(),
            [0., 0.]
        );
        // Authored potential is 3*x/2, so its physical gradient is (3/2,0).
        assert_eq!(
            crate::canonical_stokes::realization::dimensionless_body_force(&nonzero, scales)(point)
                .unwrap(),
            [1.5, 0.]
        );
    }
    for point in [[f64::NAN, 0.], [f64::INFINITY, 0.]] {
        assert!(
            crate::canonical_stokes::realization::dimensionless_body_force(&absent, scales)(point)
                .is_err()
        );
        assert!(
            crate::canonical_stokes::realization::dimensionless_body_force(&control, scales)(point)
                .is_err()
        );
    }
    // The normalized coordinate is finite, but conversion to physical x overflows.
    for model in [&absent, &control] {
        assert!(
            crate::canonical_stokes::realization::dimensionless_body_force(
                model,
                physical_scales(2.)
            )([f64::MAX, 0.])
            .is_err()
        );
    }
}

#[test]
fn omitted_transient_force_retains_unforced_roles_and_nonzero_loads() {
    let omitted = TRANSIENT_NAVIER_STOKES_SOURCE
        .replace(
            "  variable force_potential: kg / (m * s ^ 2) on fluid;\n",
            "",
        )
        .replace("  parameter zero_pressure: kg / (m * s ^ 2) = 0;\n", "")
        .replace(
            "  relation force on fluid {\n    force_potential - zero_pressure = 0;\n  }\n",
            "",
        )
        .replace("      - grad(force_potential)", "");
    let absent =
        crate::canonical_stokes::lower_transient_incompressible_navier_stokes_cartesian::<2>(
            &compile_program(&omitted),
        )
        .unwrap();
    let zero =
        crate::canonical_stokes::lower_transient_incompressible_navier_stokes_cartesian::<2>(
            &compile_program(TRANSIENT_NAVIER_STOKES_SOURCE),
        )
        .unwrap();
    assert_eq!(absent.force_potential(), None);
    assert_eq!(absent.force_potential_definition(), None);
    assert_eq!(absent.force_potential_expression(), None);
    let correspondence =
        crate::canonical_stokes::navier_stokes_integral_formulation::integral_conservative_correspondence(&absent);
    assert_eq!(correspondence.law.source, None);
    crate::canonical_stokes::navier_stokes_integral_formulation::replay_integral_conservative_correspondence(
        &correspondence,
        &absent,
    )
    .unwrap();
    let nonzero_source = TRANSIENT_NAVIER_STOKES_SOURCE
        .replace(
            "zero_pressure: kg / (m * s ^ 2) = 0",
            "zero_pressure: kg / (m * s ^ 2) = 3",
        )
        .replace(
            "force_potential - zero_pressure = 0",
            "force_potential - zero_pressure * coordinate(0) / 2[m] = 0",
        );
    let nonzero =
        crate::canonical_stokes::lower_transient_incompressible_navier_stokes_cartesian::<2>(
            &compile_program(&nonzero_source),
        )
        .unwrap();
    for point in [[0.2, 0.3], [0.8, 0.7]] {
        assert_eq!(absent.conservative_body_force(&point).unwrap(), [0., 0.]);
        assert_eq!(zero.conservative_body_force(&point).unwrap(), [0., 0.]);
        assert_eq!(nonzero.conservative_body_force(&point).unwrap(), [1.5, 0.]);
    }
    assert!(absent.conservative_body_force(&[0.]).is_err());
    assert!(absent.conservative_body_force(&[f64::NAN, 0.]).is_err());
}

fn replayed_program(source: &str) -> KernelProgram {
    let bytes = eqiora_artifact::ModelEnvelope::from_program(&compile_program(source))
        .unwrap()
        .canonical_json()
        .unwrap();
    eqiora_artifact::ModelEnvelope::from_json(&bytes, Default::default())
        .unwrap()
        .to_program()
        .unwrap()
}

fn physical_scales(
    length: f64,
) -> crate::canonical_stokes::realization::IncompressibleFlowScaleProfile2d {
    let unit = |powers| eqiora_core::DimExponents::from_integers(powers).unwrap();
    crate::canonical_stokes::realization::IncompressibleFlowScaleProfile2d::new(
        eqiora_core::DynQuantity::new(length, unit([0, 1, 0, 0, 0, 0, 0])),
        eqiora_core::DynQuantity::new(1., unit([0, 1, -1, 0, 0, 0, 0])),
        eqiora_core::DynQuantity::new(1., unit([1, -1, -2, 0, 0, 0, 0])),
    )
    .unwrap()
}
