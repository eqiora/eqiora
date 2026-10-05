//! Independently derived oracle for the frozen Cartesian Q1 elasticity form.
//!
//! Every scientific literal and tolerance below is owned by the registered
//! `numerics.compiled-cartesian-elasticity-q1-2d` evidence contract. This child
//! module calls the production-native private seam directly; it intentionally
//! contains no second elasticity formula or test-only adapter.

mod patch_assembly;
mod tape;

use std::{array, num::NonZeroUsize};

use eqiora_assembly::LocalContribution;
use eqiora_compiler::compile;
use eqiora_core::Diagnostic;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::{
    AffineGeometryMap, CartesianMesh, MeshEntity, MeshTopology, QuadratureRule, ReferenceCell,
    simplex_centroid_rule,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};

use super::*;
use crate::form_compiler::{MatrixSlot, WeakSign, WeakTermSlot};
use crate::spatial_expression::ScalarSpatialExpression;
use patch_assembly::{assemble_body_free_action, assemble_loaded_derived_form, center_vertex};

const SOURCE: &str =
    include_str!("../../../../../verify/solid/isotropic-elasticity-2d/models/linear-load.eqi");

const PARAMETERS: [f64; 3] = [3.0, 2.0, 1.0];
const STATE: [f64; 8] = [1.0, -2.0, 2.0, 0.5, 2.5, 1.5, 3.5, 4.0];
const STATE_DIRECTION: [f64; 8] = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
const PARAMETER_DIRECTION: [f64; 3] = [0.5, -1.0 / 3.0, 2.0];
const COTANGENT: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

const MATRIX: [f64; 64] = [
    11.0 / 3.0,
    5.0 / 4.0,
    -13.0 / 6.0,
    -1.0 / 4.0,
    1.0 / 3.0,
    1.0 / 4.0,
    -11.0 / 6.0,
    -5.0 / 4.0,
    5.0 / 4.0,
    11.0 / 3.0,
    1.0 / 4.0,
    1.0 / 3.0,
    -1.0 / 4.0,
    -13.0 / 6.0,
    -5.0 / 4.0,
    -11.0 / 6.0,
    -13.0 / 6.0,
    1.0 / 4.0,
    11.0 / 3.0,
    -5.0 / 4.0,
    -11.0 / 6.0,
    5.0 / 4.0,
    1.0 / 3.0,
    -1.0 / 4.0,
    -1.0 / 4.0,
    1.0 / 3.0,
    -5.0 / 4.0,
    11.0 / 3.0,
    5.0 / 4.0,
    -11.0 / 6.0,
    1.0 / 4.0,
    -13.0 / 6.0,
    1.0 / 3.0,
    -1.0 / 4.0,
    -11.0 / 6.0,
    5.0 / 4.0,
    11.0 / 3.0,
    -5.0 / 4.0,
    -13.0 / 6.0,
    1.0 / 4.0,
    1.0 / 4.0,
    -13.0 / 6.0,
    5.0 / 4.0,
    -11.0 / 6.0,
    -5.0 / 4.0,
    11.0 / 3.0,
    -1.0 / 4.0,
    1.0 / 3.0,
    -11.0 / 6.0,
    -5.0 / 4.0,
    1.0 / 3.0,
    1.0 / 4.0,
    -13.0 / 6.0,
    -1.0 / 4.0,
    11.0 / 3.0,
    5.0 / 4.0,
    -5.0 / 4.0,
    -11.0 / 6.0,
    -1.0 / 4.0,
    -13.0 / 6.0,
    1.0 / 4.0,
    1.0 / 3.0,
    5.0 / 4.0,
    11.0 / 3.0,
];
const LOAD: [f64; 8] = [
    1.0 / 16.0,
    0.0,
    1.0 / 16.0,
    0.0,
    1.0 / 16.0,
    0.0,
    1.0 / 16.0,
    0.0,
];
const RESIDUAL: [f64; 8] = [
    -217.0 / 16.0,
    -21.0,
    23.0 / 16.0,
    -9.0,
    -25.0 / 16.0,
    9.0,
    215.0 / 16.0,
    21.0,
];
const STATE_ACTION: [f64; 8] = [-7.0, 11.0, 1.0, 17.0, -1.0, -17.0, 7.0, -11.0];
const MU_ACTION: [f64; 8] = [-3.0, -5.5, -1.0, -1.5, 1.0, 1.5, 3.0, 5.5];
const LAMBDA_ACTION: [f64; 8] = [
    -9.0 / 4.0,
    -9.0 / 4.0,
    9.0 / 4.0,
    -9.0 / 4.0,
    -9.0 / 4.0,
    9.0 / 4.0,
    9.0 / 4.0,
    9.0 / 4.0,
];
const PRESSURE_GRADIENT_ACTION: [f64; 8] = [
    -1.0 / 16.0,
    0.0,
    -1.0 / 16.0,
    0.0,
    -1.0 / 16.0,
    0.0,
    -1.0 / 16.0,
    0.0,
];
const PARAMETER_ACTION: [f64; 8] = [
    -7.0 / 8.0,
    -2.0,
    -11.0 / 8.0,
    0.0,
    9.0 / 8.0,
    0.0,
    5.0 / 8.0,
    2.0,
];
const STATE_TRANSPOSE: [f64; 8] = [-21.0, -27.0, 3.0, -9.0, -3.0, 9.0, 21.0, 27.0];
const PARAMETER_TRANSPOSE: [f64; 3] = [56.0, 27.0, -1.0];

const STEP: f64 = 1.0 / 4096.0;
const ABSOLUTE_TOLERANCE: f64 = 1.0e-9;

#[test]
fn registered_evidence() {
    tape::structural_tape_falsifiers();
    private_seam_matches_the_frozen_contract();
    exact_local_contribution_and_differential_actions_match_the_dual_oracle();
    centered_finite_differences_confirm_every_forward_and_transpose_action();
    exact_values_reject_constitutive_order_load_and_transpose_mutants();
    witness_only_form_rejects_actions_and_realization_drift();
    derived_primal_requires_its_certificate_owned_load_before_realization_work();
    derivation_rejects_ambiguous_incomplete_foreign_and_mixed_roles();
    certificate_replay_rejects_stale_relation_node_and_test_trial_slot();
    ordinary_local_action_matches_the_affine_patch_oracle();
    loaded_homogeneous_patch_matches_the_separate_balance_oracle();
}

fn private_seam_matches_the_frozen_contract() {
    let _: fn(&KernelProgram) -> Result<DerivedCartesianQ1ElasticityForm2d, Diagnostic> =
        derive_cartesian_q1_elasticity_form_2d;
    let _: fn(&QuadratureRule) -> Result<AdmittedCartesianQ1ElasticityForm2d<'static>, Diagnostic> =
        compile_cartesian_q1_elasticity_form_2d;
    let _: for<'form, 'quadrature> fn(
        &'form DerivedCartesianQ1ElasticityForm2d,
        &'quadrature QuadratureRule,
    ) -> Result<
        AdmittedCartesianQ1ElasticityForm2d<'form>,
        Diagnostic,
    > = DerivedCartesianQ1ElasticityForm2d::admit_quadrature;
    fn assert_admitted_methods<'form>() {
        let _: fn(
            &AdmittedCartesianQ1ElasticityForm2d<'form>,
            &AffineGeometryMap,
            &QuadratureRule,
            f64,
            f64,
            Option<&ScalarSpatialExpression<f64>>,
        ) -> Result<LocalContribution<f64>, Diagnostic> =
            AdmittedCartesianQ1ElasticityForm2d::evaluate;
        let _: fn(
            &AdmittedCartesianQ1ElasticityForm2d<'form>,
            &AffineGeometryMap,
            &QuadratureRule,
            [f64; 3],
            &[f64; 8],
            &[f64; 8],
            [f64; 3],
            &[f64; 8],
        ) -> Result<CartesianElasticityDifferentialActions2d, Diagnostic> =
            AdmittedCartesianQ1ElasticityForm2d::evaluate_with_actions;
    }
    assert_admitted_methods();
    let _: fn(&CartesianElasticityDifferentialActions2d) -> &LocalContribution<f64> =
        CartesianElasticityDifferentialActions2d::contribution;
    let _: fn(&CartesianElasticityDifferentialActions2d) -> &[f64; 8] =
        CartesianElasticityDifferentialActions2d::residual;
    let _: fn(&CartesianElasticityDifferentialActions2d) -> &[f64; 8] =
        CartesianElasticityDifferentialActions2d::state_direction_action;
    let _: fn(&CartesianElasticityDifferentialActions2d) -> &[f64; 8] =
        CartesianElasticityDifferentialActions2d::parameter_direction_action;
    let _: fn(&CartesianElasticityDifferentialActions2d) -> &[f64; 8] =
        CartesianElasticityDifferentialActions2d::state_transpose_action;
    let _: fn(&CartesianElasticityDifferentialActions2d) -> &[f64; 3] =
        CartesianElasticityDifferentialActions2d::parameter_transpose_action;
}

fn exact_local_contribution_and_differential_actions_match_the_dual_oracle() {
    let program = compile_program(SOURCE);
    let model =
        crate::canonical_elasticity::lower_isotropic_elasticity_cartesian_2d(&program).unwrap();
    let quadrature = quadrature();
    let form = derive_cartesian_q1_elasticity_form_2d(&program).unwrap();
    let admitted = form.admit_quadrature(&quadrature).unwrap();
    let primal = admitted
        .evaluate(
            &geometry(),
            &quadrature,
            PARAMETERS[0],
            PARAMETERS[1],
            Some(model.load_potential_expression()),
        )
        .unwrap();
    let baseline = admitted
        .evaluate_with_actions(
            &geometry(),
            &quadrature,
            PARAMETERS,
            &STATE,
            &STATE_DIRECTION,
            PARAMETER_DIRECTION,
            &COTANGENT,
        )
        .unwrap();
    assert_absolute_slice(baseline.contribution().matrix(), &MATRIX);
    assert_absolute_slice(baseline.contribution().rhs(), &LOAD);
    assert_absolute_slice(primal.matrix(), baseline.contribution().matrix());
    assert_absolute_slice(primal.rhs(), baseline.contribution().rhs());
    assert_absolute_slice(baseline.residual(), &RESIDUAL);
    assert_absolute_slice(baseline.state_direction_action(), &STATE_ACTION);
    assert_absolute_slice(baseline.parameter_direction_action(), &PARAMETER_ACTION);
    assert_absolute_slice(baseline.state_transpose_action(), &STATE_TRANSPOSE);
    assert_absolute_slice(baseline.parameter_transpose_action(), &PARAMETER_TRANSPOSE);

    for (direction, expected) in [
        ([1.0, 0.0, 0.0], MU_ACTION),
        ([0.0, 1.0, 0.0], LAMBDA_ACTION),
        ([0.0, 0.0, 1.0], PRESSURE_GRADIENT_ACTION),
    ] {
        let coordinate = actions(PARAMETERS, STATE, STATE_DIRECTION, direction, COTANGENT);
        assert_absolute_slice(coordinate.parameter_direction_action(), &expected);
    }

    assert_absolute_scalar(dot(&COTANGENT, baseline.state_direction_action()), -60.0);
    assert_absolute_scalar(
        dot(&STATE_DIRECTION, baseline.state_transpose_action()),
        -60.0,
    );
    assert_absolute_scalar(dot(&COTANGENT, baseline.parameter_direction_action()), 17.0);
    assert_absolute_scalar(
        dot(&PARAMETER_DIRECTION, baseline.parameter_transpose_action()),
        17.0,
    );
}

fn centered_finite_differences_confirm_every_forward_and_transpose_action() {
    let baseline = actions(
        PARAMETERS,
        STATE,
        STATE_DIRECTION,
        PARAMETER_DIRECTION,
        COTANGENT,
    );
    let plus_state = array::from_fn(|index| STATE[index] + STEP * STATE_DIRECTION[index]);
    let minus_state = array::from_fn(|index| STATE[index] - STEP * STATE_DIRECTION[index]);
    let state_fd = centered_vector(
        residual(PARAMETERS, plus_state),
        residual(PARAMETERS, minus_state),
    );
    assert_absolute_slice(&state_fd, baseline.state_direction_action());

    let plus_parameters =
        array::from_fn(|index| PARAMETERS[index] + STEP * PARAMETER_DIRECTION[index]);
    let minus_parameters =
        array::from_fn(|index| PARAMETERS[index] - STEP * PARAMETER_DIRECTION[index]);
    let parameter_fd = centered_vector(
        residual(plus_parameters, STATE),
        residual(minus_parameters, STATE),
    );
    assert_absolute_slice(&parameter_fd, baseline.parameter_direction_action());

    let state_transpose_fd: [f64; 8] = array::from_fn(|coordinate| {
        let mut plus = STATE;
        let mut minus = STATE;
        plus[coordinate] += STEP;
        minus[coordinate] -= STEP;
        centered_scalar(
            dot(&COTANGENT, &residual(PARAMETERS, plus)),
            dot(&COTANGENT, &residual(PARAMETERS, minus)),
        )
    });
    assert_absolute_slice(&state_transpose_fd, baseline.state_transpose_action());

    let parameter_transpose_fd: [f64; 3] = array::from_fn(|coordinate| {
        let mut plus = PARAMETERS;
        let mut minus = PARAMETERS;
        plus[coordinate] += STEP;
        minus[coordinate] -= STEP;
        centered_scalar(
            dot(&COTANGENT, &residual(plus, STATE)),
            dot(&COTANGENT, &residual(minus, STATE)),
        )
    });
    assert_absolute_slice(
        &parameter_transpose_fd,
        baseline.parameter_transpose_action(),
    );
}

fn exact_values_reject_constitutive_order_load_and_transpose_mutants() {
    let baseline = actions(
        PARAMETERS,
        STATE,
        STATE_DIRECTION,
        PARAMETER_DIRECTION,
        COTANGENT,
    );

    // A full-gradient law cannot preserve the frozen matrix or the zero action
    // of the exact infinitesimal rotation in this DOF ordering.
    let rotation = [0.0, 0.0, 0.0, 0.5, -0.5, 0.0, -0.5, 0.5];
    let rotation_action = matrix_action(baseline.contribution().matrix(), &rotation);
    assert_absolute_slice(&rotation_action, &[0.0; 8]);

    for (name, parameters) in [
        ("swapped Lame contributions", [2.0, 3.0, 1.0]),
        ("halved shear contribution", [1.5, 2.0, 1.0]),
        ("dropped volumetric contribution", [3.0, 0.0, 1.0]),
        ("halved volumetric contribution", [3.0, 1.0, 1.0]),
    ] {
        let mutant = actions(
            parameters,
            STATE,
            STATE_DIRECTION,
            PARAMETER_DIRECTION,
            COTANGENT,
        );
        assert_rejected_numeric_mutant(mutant.contribution().matrix(), &MATRIX, name);
    }
    let half_shear = actions(
        [1.5, 2.0, 1.0],
        STATE,
        STATE_DIRECTION,
        PARAMETER_DIRECTION,
        COTANGENT,
    );
    let dropped_shear = baseline
        .contribution()
        .matrix()
        .iter()
        .zip(half_shear.contribution().matrix())
        .map(|(baseline, half)| 2.0 * half - baseline)
        .collect::<Vec<_>>();
    assert_rejected_numeric_mutant(&dropped_shear, &MATRIX, "dropped shear contribution");

    let component_permutation = [1, 0, 3, 2, 5, 4, 7, 6];
    let node_permutation = [0, 1, 4, 5, 2, 3, 6, 7];
    let component_major_permutation = [0, 2, 4, 6, 1, 3, 5, 7];
    for (name, permutation) in [
        ("swapped vector components", component_permutation),
        ("swapped scalar-node bit order", node_permutation),
        (
            "component-major local-DOF flattening",
            component_major_permutation,
        ),
    ] {
        let mutant = permute_matrix(baseline.contribution().matrix(), permutation);
        assert_rejected_numeric_mutant(&mutant, &MATRIX, name);
    }

    let reversed_load = baseline
        .contribution()
        .rhs()
        .iter()
        .map(|value| -*value)
        .collect::<Vec<_>>();
    assert_rejected_numeric_mutant(&reversed_load, &LOAD, "reversed load gradient");
    assert_rejected_numeric_mutant(
        baseline.state_direction_action(),
        &STATE_TRANSPOSE,
        "primal action returned as state transpose",
    );

    for (name, mutant) in [
        ("omitted mu coordinate", [0.0, 27.0, -1.0]),
        ("omitted lambda coordinate", [56.0, 0.0, -1.0]),
        ("omitted pressure-gradient coordinate", [56.0, 27.0, 0.0]),
        ("transposed mu/lambda coordinates", [27.0, 56.0, -1.0]),
    ] {
        assert_rejected_numeric_mutant(&mutant, &PARAMETER_TRANSPOSE, name);
    }
}

fn witness_only_form_rejects_actions_and_realization_drift() {
    let quadrature = quadrature();
    let witness = compile_cartesian_q1_elasticity_form_2d(&quadrature).unwrap();
    assert!(
        witness
            .evaluate_with_actions(
                &geometry(),
                &quadrature,
                PARAMETERS,
                &STATE,
                &STATE_DIRECTION,
                PARAMETER_DIRECTION,
                &COTANGENT,
            )
            .is_err(),
        "witness-data form admitted derivative actions without a certificate"
    );

    let derived = derived_form();
    let one_point = QuadratureRule::tensor_product_gauss_legendre(2, 1).unwrap();
    let three_point = QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap();
    let simplex = simplex_centroid_rule(2).unwrap();
    assert!(derived.admit_quadrature(&one_point).is_err());
    assert!(derived.admit_quadrature(&three_point).is_err());
    assert!(compile_cartesian_q1_elasticity_form_2d(&simplex).is_err());

    let generic = compile_cartesian_q1_elasticity_form_2d(&three_point).unwrap();
    let model = crate::canonical_elasticity::lower_isotropic_elasticity_cartesian_2d(
        &compile_program(SOURCE),
    )
    .unwrap();
    assert!(
        generic
            .evaluate(&geometry(), &three_point, 3.0, 2.0, None)
            .is_ok(),
        "generic witness form lost the body-free local-action envelope"
    );
    assert!(
        generic
            .evaluate(
                &geometry(),
                &three_point,
                3.0,
                2.0,
                Some(model.load_potential_expression()),
            )
            .is_ok(),
        "generic witness form lost accepted ScalarSpatialExpression loads"
    );

    let admitted = derived.admit_quadrature(&quadrature).unwrap();
    let wrong_geometry = AffineGeometryMap::new(
        ReferenceCell::hypercube(2).unwrap(),
        3,
        vec![0.25, 0.25, 0.0],
        vec![0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
    )
    .unwrap();
    assert!(
        admitted
            .evaluate_with_actions(
                &wrong_geometry,
                &quadrature,
                PARAMETERS,
                &STATE,
                &STATE_DIRECTION,
                PARAMETER_DIRECTION,
                &COTANGENT,
            )
            .is_err()
    );
}

fn derived_primal_requires_its_certificate_owned_load_before_realization_work() {
    let program = compile_program(SOURCE);
    let exact_model =
        crate::canonical_elasticity::lower_isotropic_elasticity_cartesian_2d(&program).unwrap();
    let foreign_source = SOURCE.replace("pressure_gradient", "foreign_pressure_gradient");
    let foreign_model = crate::canonical_elasticity::lower_isotropic_elasticity_cartesian_2d(
        &compile_program(&foreign_source),
    )
    .unwrap();
    let quadrature = quadrature();
    let form = derive_cartesian_q1_elasticity_form_2d(&program).unwrap();
    let admitted = form.admit_quadrature(&quadrature).unwrap();
    let wrong_geometry = AffineGeometryMap::new(
        ReferenceCell::hypercube(2).unwrap(),
        3,
        vec![0.25, 0.25, 0.0],
        vec![0.25, 0.0, 0.0, 0.25, 0.0, 0.0],
    )
    .unwrap();
    let wrong_quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 1).unwrap();

    for (name, potential) in [
        ("absent", None),
        ("foreign", Some(foreign_model.load_potential_expression())),
    ] {
        let error = admitted
            .evaluate(&wrong_geometry, &wrong_quadrature, 3.0, 2.0, potential)
            .expect_err("derived primal must reject a non-certificate load");
        assert!(
            error.message().contains("load") || error.message().contains("certificate"),
            "{name} potential reached realization validation first: {error:?}"
        );
    }

    assert!(
        admitted
            .evaluate(
                &geometry(),
                &quadrature,
                3.0,
                2.0,
                Some(exact_model.load_potential_expression()),
            )
            .is_ok()
    );
}

fn derivation_rejects_ambiguous_incomplete_foreign_and_mixed_roles() {
    let full_gradient = SOURCE.replace("symmetric_part(grad(displacement))", "grad(displacement)");
    assert_derivation_rejects(&full_gradient, "full-gradient constitutive law");

    let wrong_field_shape = SOURCE.replace("vector<m, 2>", "m");
    assert_ne!(wrong_field_shape, SOURCE);
    assert!(
        compile("compiled-cartesian-elasticity.eqi", &wrong_field_shape).is_err(),
        "wrong displacement field shape escaped semantic typing"
    );

    let altered_domain = SOURCE.replace("box(0, 1, 0, 1)", "box(0, 2, 0, 1)");
    assert_derivation_rejects(&altered_domain, "altered unit-square domain");

    let absent_coefficient =
        SOURCE.replace("      + lambda * isotropic_lift(div(displacement))\n", "");
    assert_derivation_rejects(&absent_coefficient, "absent Lamé coefficient role");

    let duplicated_coefficient = SOURCE.replace(
        "      + lambda * isotropic_lift(div(displacement))",
        concat!(
            "      + lambda * isotropic_lift(div(displacement))\n",
            "      + lambda * isotropic_lift(div(displacement))",
        ),
    );
    assert_derivation_rejects(&duplicated_coefficient, "duplicated Lamé coefficient role");

    let duplicate_unknown = SOURCE.replace(
        "variable displacement: vector<m, 2> on body;",
        concat!(
            "variable displacement: vector<m, 2> on body;\n",
            "  variable foreign_displacement: vector<m, 2> on body;",
        ),
    );
    assert_derivation_rejects(&duplicate_unknown, "duplicate displacement role");

    let foreign_parameter = SOURCE.replace(
        "parameter pressure_gradient: kg / (m ^ 2 * s ^ 2) = 1;",
        concat!(
            "parameter pressure_gradient: kg / (m ^ 2 * s ^ 2) = 1;\n",
            "  parameter foreign_parameter: 1 = 7;",
        ),
    );
    assert_derivation_rejects(&foreign_parameter, "foreign parameter role");

    let duplicate_volume = SOURCE.replace(
        "  relation x_lower_value on x_lower",
        concat!(
            "  relation foreign_balance on body {\n",
            "    -div(2 * mu * symmetric_part(grad(displacement))\n",
            "      + lambda * isotropic_lift(div(displacement)))\n",
            "      - grad(load_potential) = 0;\n",
            "  }\n\n",
            "  relation x_lower_value on x_lower",
        ),
    );
    assert_derivation_rejects(&duplicate_volume, "ambiguous volume relation");

    let incomplete_boundary = SOURCE.replace(
        "  relation y_upper_value on y_upper { trace(displacement) = 0; }\n",
        "",
    );
    assert_derivation_rejects(&incomplete_boundary, "incomplete boundary role");

    let duplicate_boundary = SOURCE.replace(
        "  relation y_upper_value on y_upper { trace(displacement) = 0; }",
        concat!(
            "  relation y_upper_value on y_upper { trace(displacement) = 0; }\n",
            "  relation y_upper_duplicate on y_upper { trace(displacement) = 0; }",
        ),
    );
    assert_derivation_rejects(&duplicate_boundary, "ambiguous boundary role");

    let mixed_boundary = SOURCE.replace(
        "relation x_upper_value on x_upper { trace(displacement) = 0; }",
        concat!(
            "relation x_upper_value on x_upper {\n",
            "    normal(2 * mu * symmetric_part(grad(displacement))\n",
            "      + lambda * isotropic_lift(div(displacement))) = 0;\n",
            "  }",
        ),
    );
    assert_derivation_rejects(&mixed_boundary, "mixed boundary role");

    let mut oversized_terms = vec!["0 * (pressure_gradient * coordinate(0))"; 4_097];
    oversized_terms.push("pressure_gradient * coordinate(0)");
    let oversized_load = balanced_sum(&oversized_terms);
    let oversized = SOURCE.replace("pressure_gradient * coordinate(0)", &oversized_load);
    assert_derivation_rejects(&oversized, "bounded compiler resources");
}

fn certificate_replay_rejects_stale_relation_node_and_test_trial_slot() {
    let quadrature = quadrature();

    let mut stale_relation = derived_form();
    let relation = stale_relation
        .certificate
        .entries
        .iter()
        .find(|entry| entry.relation != stale_relation.certificate.entries[0].relation)
        .expect("elasticity certificate owns volume and boundary Relations")
        .relation;
    stale_relation.certificate.entries[0].relation = relation;
    assert!(stale_relation.admit_quadrature(&quadrature).is_err());

    let foreign_source = SOURCE
        .replace("domain body", "domain foreign_body")
        .replace("boundary(body,", "boundary(foreign_body,")
        .replace(" on body", " on foreign_body");
    let foreign_form = derive_cartesian_q1_elasticity_form_2d(&compile_program(&foreign_source))
        .expect("renaming the exact Model preserves its mathematical shape");
    let mut foreign_relation = derived_form();
    foreign_relation.certificate.entries[0].relation = foreign_form.certificate.entries[0].relation;
    assert!(foreign_relation.admit_quadrature(&quadrature).is_err());

    let mut foreign_domain = derived_form();
    foreign_domain.domain = foreign_form.domain;
    assert!(foreign_domain.admit_quadrature(&quadrature).is_err());

    let mut stale_node = derived_form();
    let source_node = stale_node
        .certificate
        .entries
        .iter()
        .find(|entry| entry.source_node != stale_node.certificate.entries[0].source_node)
        .expect("elasticity certificate owns distinct derivation nodes")
        .source_node;
    stale_node.certificate.entries[0].source_node = source_node;
    assert!(stale_node.admit_quadrature(&quadrature).is_err());

    let mut slot_mutant = derived_form();
    let bilinear = slot_mutant
        .certificate
        .entries
        .iter()
        .position(|entry| matches!(entry.slot, WeakTermSlot::Bilinear { .. }))
        .expect("elasticity certificate owns a bilinear test/trial entry");
    slot_mutant.certificate.entries[bilinear].slot = WeakTermSlot::Bilinear {
        test: MatrixSlot::Trial,
        trial: MatrixSlot::Test,
    };
    assert!(slot_mutant.admit_quadrature(&quadrature).is_err());

    let mut sign_mutant = derived_form();
    let positive = sign_mutant
        .certificate
        .entries
        .iter()
        .position(|entry| entry.sign == WeakSign::Positive)
        .expect("elasticity certificate owns a positive weak term");
    sign_mutant.certificate.entries[positive].sign = WeakSign::Negative;
    assert!(sign_mutant.admit_quadrature(&quadrature).is_err());
}

fn ordinary_local_action_matches_the_affine_patch_oracle() {
    let mesh = CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0]], &[2, 2]).unwrap();
    let quadrature = quadrature();
    let action = crate::cartesian_elasticity::lower_cartesian_q1_linear_elasticity_local_action_2d(
        &mesh,
        PARAMETERS[0],
        PARAMETERS[1],
        &quadrature,
    )
    .unwrap();
    assert_eq!(action.entity_count(), mesh.entity_count(2).unwrap());
    assert_eq!(action.rows(), 8);
    assert_eq!(action.columns(), 8);
    assert_absolute_slice(&action.coefficients()[..64], &MATRIX);

    let system = assemble_body_free_action(&mesh, &action);
    let vertex_count = mesh.entity_count(0).unwrap();
    let center = center_vertex(&mesh);
    let displacement = (0..vertex_count)
        .flat_map(|vertex| {
            let point = mesh.vertex_coordinates(MeshEntity::new(0, vertex)).unwrap();
            [
                2.0 * point[0] + 3.0 * point[1] + 1.0,
                5.0 * point[0] + 7.0 * point[1] - 2.0,
            ]
        })
        .collect::<Vec<_>>();
    let field = crate::cartesian_elasticity::CartesianQ1VectorField2d::new(
        mesh.clone(),
        displacement.clone(),
    )
    .unwrap();
    let interpolation_error = field
        .error_norms(
            &|point: &[f64]| {
                (
                    [
                        2.0 * point[0] + 3.0 * point[1] + 1.0,
                        5.0 * point[0] + 7.0 * point[1] - 2.0,
                    ],
                    [[2.0, 3.0], [5.0, 7.0]],
                )
            },
            &quadrature,
        )
        .unwrap();
    assert_absolute_scalar(interpolation_error.l2(), 0.0);
    assert_absolute_scalar(interpolation_error.h1_seminorm(), 0.0);

    let elastic_force = system.matrix().multiply(&displacement).unwrap();
    assert_absolute_slice(&elastic_force[2 * center..2 * center + 2], &[0.0, 0.0]);

    // The frozen symmetric strain [[2, 4], [4, 7]] and stress
    // [[30, 24], [24, 60]] are observed through their four exact boundary
    // resultants; no constitutive formula is reimplemented in the oracle.
    let mut side_resultants = [[0.0; 2]; 4];
    let mut assembled_resultant = [0.0; 2];
    for vertex in 0..vertex_count {
        let point = mesh.vertex_coordinates(MeshEntity::new(0, vertex)).unwrap();
        let force = &elastic_force[2 * vertex..2 * vertex + 2];
        assembled_resultant[0] += force[0];
        assembled_resultant[1] += force[1];
        for (side, on_side) in [
            point[0] == 0.0,
            point[0] == 1.0,
            point[1] == 0.0,
            point[1] == 1.0,
        ]
        .into_iter()
        .enumerate()
        {
            if on_side {
                side_resultants[side][0] += force[0];
                side_resultants[side][1] += force[1];
            }
        }
    }
    for (actual, expected) in
        side_resultants
            .iter()
            .zip([[-30.0, -24.0], [30.0, 24.0], [-24.0, -60.0], [24.0, 60.0]])
    {
        assert_absolute_slice(actual, &expected);
    }
    assert_absolute_slice(&assembled_resultant, &[0.0, 0.0]);
}

fn loaded_homogeneous_patch_matches_the_separate_balance_oracle() {
    let program = compile_program(SOURCE);
    let model =
        crate::canonical_elasticity::lower_isotropic_elasticity_cartesian_2d(&program).unwrap();
    let mesh = CartesianMesh::uniform(&[[0.0, 1.0], [0.0, 1.0]], &[2, 2]).unwrap();
    let quadrature = quadrature();
    let system = assemble_loaded_derived_form(&program, &model, &mesh, &quadrature);
    let center = center_vertex(&mesh);
    let center_dofs = [2 * center, 2 * center + 1];
    let matrix = system.matrix();
    assert_absolute_slice(
        &[
            matrix.entry(center_dofs[0], center_dofs[0]).unwrap_or(0.0),
            matrix.entry(center_dofs[0], center_dofs[1]).unwrap_or(0.0),
            matrix.entry(center_dofs[1], center_dofs[0]).unwrap_or(0.0),
            matrix.entry(center_dofs[1], center_dofs[1]).unwrap_or(0.0),
        ],
        &[44.0 / 3.0, 0.0, 0.0, 44.0 / 3.0],
    );
    assert_absolute_slice(
        &[system.rhs()[center_dofs[0]], system.rhs()[center_dofs[1]]],
        &[1.0 / 4.0, 0.0],
    );

    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-15,
        1.0e-15,
        NonZeroUsize::new(64).unwrap(),
    )
    .unwrap();
    let solution = crate::cartesian_elasticity::solve_cartesian_q1_linear_elasticity_2d(
        &mesh,
        model.shear_modulus(),
        model.first_lame_parameter(),
        model.load_potential_expression(),
        &quadrature,
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan),
    )
    .unwrap();
    assert_absolute_slice(
        solution.displacement().vertex_values(center).unwrap(),
        &[3.0 / 176.0, 0.0],
    );
    assert_absolute_slice(&solution.boundary_reaction(), &[-1.0, 0.0]);
    assert_absolute_slice(&solution.integrated_body_force(), &[1.0, 0.0]);
    assert_absolute_slice(
        &[
            solution.boundary_reaction()[0] + solution.integrated_body_force()[0],
            solution.boundary_reaction()[1] + solution.integrated_body_force()[1],
        ],
        &[0.0, 0.0],
    );
}

fn balanced_sum(terms: &[&str]) -> String {
    if let [term] = terms {
        return (*term).to_owned();
    }
    let middle = terms.len() / 2;
    format!(
        "({} + {})",
        balanced_sum(&terms[..middle]),
        balanced_sum(&terms[middle..]),
    )
}

fn actions(
    parameters: [f64; 3],
    state: [f64; 8],
    state_direction: [f64; 8],
    parameter_direction: [f64; 3],
    cotangent: [f64; 8],
) -> CartesianElasticityDifferentialActions2d {
    let form = derived_form();
    let quadrature = quadrature();
    form.admit_quadrature(&quadrature)
        .unwrap()
        .evaluate_with_actions(
            &geometry(),
            &quadrature,
            parameters,
            &state,
            &state_direction,
            parameter_direction,
            &cotangent,
        )
        .unwrap()
}

fn residual(parameters: [f64; 3], state: [f64; 8]) -> [f64; 8] {
    *actions(parameters, state, [0.0; 8], [0.0; 3], [0.0; 8]).residual()
}

fn derived_form() -> DerivedCartesianQ1ElasticityForm2d {
    derive_cartesian_q1_elasticity_form_2d(&compile_program(SOURCE)).unwrap()
}

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled = compile("compiled-cartesian-elasticity.eqi", source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn assert_derivation_rejects(source: &str, mutation: &str) {
    assert_ne!(source, SOURCE, "{mutation} did not change its input");
    let program = compile_program(source);
    let error = derive_cartesian_q1_elasticity_form_2d(&program)
        .expect_err("structural mutant must fail closed");
    assert!(
        error.message().contains("form compiler"),
        "{mutation} escaped the compiler gate with an unrelated diagnostic: {error:?}"
    );
}

fn geometry() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::hypercube(2).unwrap(),
        2,
        vec![0.25, 0.25],
        vec![0.25, 0.0, 0.0, 0.25],
    )
    .unwrap()
}

fn quadrature() -> QuadratureRule {
    QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap()
}

fn centered_vector<const N: usize>(plus: [f64; N], minus: [f64; N]) -> [f64; N] {
    array::from_fn(|index| centered_scalar(plus[index], minus[index]))
}

fn centered_scalar(plus: f64, minus: f64) -> f64 {
    (plus - minus) / (2.0 * STEP)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn matrix_action(matrix: &[f64], input: &[f64; 8]) -> [f64; 8] {
    array::from_fn(|row| {
        matrix[row * 8..(row + 1) * 8]
            .iter()
            .zip(input)
            .map(|(entry, input)| entry * input)
            .sum()
    })
}

fn permute_matrix(matrix: &[f64], permutation: [usize; 8]) -> [f64; 64] {
    array::from_fn(|index| {
        let row = index / 8;
        let column = index % 8;
        matrix[permutation[row] * 8 + permutation[column]]
    })
}

fn assert_absolute_scalar(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= ABSOLUTE_TOLERANCE,
        "actual={actual:e}, expected={expected:e}, error={:e}, tolerance={ABSOLUTE_TOLERANCE:e}",
        (actual - expected).abs(),
    );
}

fn assert_absolute_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().copied().zip(expected.iter().copied()) {
        assert_absolute_scalar(actual, expected);
    }
}

fn assert_rejected_numeric_mutant(actual: &[f64], expected: &[f64], mutation: &str) {
    assert_eq!(actual.len(), expected.len());
    let maximum_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        maximum_error > ABSOLUTE_TOLERANCE,
        "{mutation} survived the exact oracle: error={maximum_error:e}"
    );
}
