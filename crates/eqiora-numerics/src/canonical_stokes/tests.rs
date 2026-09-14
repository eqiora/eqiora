use eqiora_compiler::compile;
use eqiora_core::diagnostic::codes;
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::BoundarySide;
use eqiora_sem::KernelProgram;

use super::{
    lower_steady_incompressible_stokes_cartesian_2d,
    lower_transient_incompressible_navier_stokes_cartesian_2d,
    lower_transient_incompressible_navier_stokes_cartesian_3d,
};
use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};
use crate::form_compiler::vocabulary::{
    BoundaryTreatment, FormulationKind, IntegralConservativeRule, MixedFormulationRule,
    MixedTermRole, MixedTermSign,
};

const SOURCE: &str = r#"
model steady_stokes() {
  domain fluid = box(0, 2, -1, 1);
  domain x_lower = boundary(fluid, axis = 0, side = lower);
  domain x_upper = boundary(fluid, axis = 0, side = upper);
  domain y_lower = boundary(fluid, axis = 1, side = lower);
  domain y_upper = boundary(fluid, axis = 1, side = upper);

  variable velocity: vector<m / s, 2> on fluid;
  variable pressure: kg / (m * s ^ 2) on fluid;
  variable force_potential: kg / (m * s ^ 2) on fluid;
  parameter mu: kg / (m * s) = 2.5;
  parameter load_scale: kg / (m * s ^ 2) = 3;
  parameter length_scale: m = 2;

  relation force on fluid {
    force_potential - load_scale * coordinate(0) / length_scale = 0;
  }
  relation momentum on fluid {
    -div(
      2 * mu * symmetric_part(grad(velocity))
      - isotropic_lift(pressure)
    ) - grad(force_potential) = 0;
  }
  relation incompressibility on fluid { div(velocity) = 0; }

  relation x_lower_zero on x_lower { trace(velocity) = 0; }
  relation x_upper_zero on x_upper { trace(velocity) = 0; }
  relation y_lower_zero on y_lower { trace(velocity) = 0; }
  relation y_upper_zero on y_upper { trace(velocity) = 0; }
}
"#;

const PORT_COMPONENTS: &str = r#"
public connector VelocityTractionBoundary {
  trace velocity: m / s;
  flux traction: kg / (m * s ^ 2);
  shape spatial_vector;
  frame spatial;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}

public component NewtonianBoundary2d(
  support body: volume(ambient_dimension = 2),
  support face: boundary(parent = body),
  variable velocity: vector<m / s, 2> on body,
  variable pressure: kg / (m * s ^ 2) on body,
  parameter dynamic_viscosity: kg / (m * s),
  port mechanical: VelocityTractionBoundary over face
) {

  relation interface on face {
    trace(velocity) - mechanical.velocity = 0;
    normal(
      2 * dynamic_viscosity * symmetric_part(grad(velocity))
      - isotropic_lift(pressure)
    ) - mechanical.traction = 0;
  }
}

public component NormalPressureTraction2d(
  support body: volume(ambient_dimension = 2),
  support face: boundary(parent = body),
  variable pressure: kg / (m * s ^ 2) on body,
  port mechanical: VelocityTractionBoundary over face
) {

  relation prescribed_traction on face {
    mechanical.traction - normal(isotropic_lift(pressure)) = 0;
  }
}
"#;

const TRANSIENT_NAVIER_STOKES_SOURCE: &str = r#"
public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2]
  = component(left, 0) * component(right, 1);

model transient_navier_stokes() {
  domain fluid = box(0, 2, -1, 1);
  domain x_lower = boundary(fluid, axis = 0, side = lower);
  domain x_upper = boundary(fluid, axis = 0, side = upper);
  domain y_lower = boundary(fluid, axis = 1, side = lower);
  domain y_upper = boundary(fluid, axis = 1, side = upper);

  state velocity: vector<m / s, 2> on fluid;
  variable pressure: kg / (m * s ^ 2) on fluid;
  variable force_potential: kg / (m * s ^ 2) on fluid;
  parameter rho: kg / m ^ 3 = 1.25;
  parameter mu: kg / (m * s) = 0.125;
  parameter zero_pressure: kg / (m * s ^ 2) = 0;

  relation force on fluid {
    force_potential - zero_pressure = 0;
  }
  relation momentum on fluid {
    rho * derivative(velocity)
      + div(rho * outer_product(left = velocity, right = velocity))
      - div(
        2 * mu * symmetric_part(grad(velocity))
        - isotropic_lift(pressure)
      )
      - grad(force_potential) = 0;
  }
  relation incompressibility on fluid { div(velocity) = 0; }

  relation x_lower_zero on x_lower { trace(velocity) = 0; }
  relation x_upper_zero on x_upper { trace(velocity) = 0; }
  relation y_lower_zero on y_lower { trace(velocity) = 0; }
  relation y_upper_zero on y_upper { trace(velocity) = 0; }
}
"#;

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled = compile("steady-stokes.eqi", source).expect("source compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("model validates")
}

fn assert_rejected(source: &str) {
    assert_eq!(
        lower_steady_incompressible_stokes_cartesian_2d(&compile_program(source))
            .expect_err("altered model must fail closed")
            .code(),
        codes::INVALID_SPATIAL_LOWERING
    );
}

#[test]
fn steady_stokes_directional_certificate_rejects_structural_mutants() {
    let program = compile_program(SOURCE);
    let model = lower_steady_incompressible_stokes_cartesian_2d(&program)
        .expect("canonical steady Stokes lowers");
    let common = model.common();
    let certified = common.correspondence().entries.clone();
    super::mixed_certificate::check_model(&program, common, &certified, None)
        .expect("live DAG matches the admitted certificate");

    let mut missing = certified.clone();
    missing.pop();
    assert!(super::mixed_certificate::check_model(&program, common, &missing, None).is_err());

    let mut duplicate = certified.clone();
    duplicate.push(duplicate[0]);
    assert!(super::mixed_certificate::check_model(&program, common, &duplicate, None).is_err());

    let mut reversed = certified.clone();
    let momentum = reversed
        .iter_mut()
        .find(|entry| entry.role == MixedTermRole::MomentumBodySource)
        .expect("momentum source entry");
    momentum.source_sign = match momentum.source_sign {
        MixedTermSign::Positive => MixedTermSign::Negative,
        MixedTermSign::Negative => MixedTermSign::Positive,
    };
    assert!(super::mixed_certificate::check_model(&program, common, &reversed, None).is_err());

    let mut wrong_assumption = certified.clone();
    wrong_assumption[0].assumptions = &["numerical-agreement-implies-equivalence"];
    assert!(
        super::mixed_certificate::check_model(&program, common, &wrong_assumption, None).is_err()
    );
    let definition =
        super::support::typed_relation(&program, common.force_potential_definition()).unwrap();
    let dag = definition.expression();
    let field_node=dag.nodes().iter().position(|node|matches!(node,eqiora_schema::kernel::ExprNode::Symbol(eqiora_schema::kernel::SymbolRef::Field(id)) if id.erase()==common.force_potential())).unwrap();
    let mut self_authenticating = certified.clone();
    let entry = self_authenticating
        .iter_mut()
        .find(|e| e.role == MixedTermRole::SourceDefinition)
        .unwrap();
    entry.source_node = dag.node_id(field_node as u32).unwrap();
    entry.source_sign = MixedTermSign::Positive;
    entry.produced_sign = MixedTermSign::Positive;
    assert!(
        super::mixed_certificate::check_model(&program, common, &self_authenticating, None)
            .is_err()
    );

    let mut crosswired = certified;
    crosswired
        .iter_mut()
        .find(|entry| entry.role == MixedTermRole::ContinuityConstraint)
        .expect("continuity entry")
        .test = Some(common.velocity());
    assert!(super::mixed_certificate::check_model(&program, common, &crosswired, None).is_err());

    let mut inward = common.correspondence().entries.clone();
    inward
        .iter_mut()
        .find(|entry| entry.role == MixedTermRole::BoundaryLaw)
        .expect("boundary entry")
        .normal = crate::form_compiler::vocabulary::MixedNormalOrientation::ParentOutward;
    assert!(super::mixed_certificate::check_model(&program, common, &inward, None).is_err());
}

fn assert_transient_navier_stokes_rejected(source: &str) {
    assert_eq!(
        lower_transient_incompressible_navier_stokes_cartesian_2d(&compile_program(source))
            .expect_err("altered transient flow must fail closed")
            .code(),
        codes::INVALID_SPATIAL_LOWERING
    );
}

fn transient_navier_stokes_source_3d() -> String {
    TRANSIENT_NAVIER_STOKES_SOURCE
        .replace("vector<m / s, 2>", "vector<m / s, 3>")
        .replace(
            "domain fluid = box(0, 2, -1, 1);",
            "domain fluid = box(0, 2, -1, 1, -2, 2);",
        )
        .replace(
            "  state velocity:",
            "  domain z_lower = boundary(fluid, axis = 2, side = lower);\n  domain z_upper = boundary(fluid, axis = 2, side = upper);\n  state velocity:",
        )
        .replace(
            "  relation y_upper_zero on y_upper { trace(velocity) = 0; }",
            "  relation y_upper_zero on y_upper { trace(velocity) = 0; }\n  relation z_lower_zero on z_lower { trace(velocity) = 0; }\n  relation z_upper_zero on z_upper { trace(velocity) = 0; }",
        )
}

fn direct_normal_pressure_relation(field: &str, operator: char) -> String {
    format!(
        "relation x_upper_zero on x_upper {{\n    normal(\n      2 * mu * symmetric_part(grad(velocity))\n      - isotropic_lift(pressure)\n    ) {operator} normal(isotropic_lift({field})) = 0;\n  }}"
    )
}

fn source_with_normal_pressure(operator: char) -> String {
    SOURCE
        .replace(
            "variable force_potential: kg / (m * s ^ 2) on fluid;",
            "variable force_potential: kg / (m * s ^ 2) on fluid;\n  variable ambient_pressure: kg / (m * s ^ 2) on fluid;",
        )
        .replace(
            "parameter length_scale: m = 2;",
            "parameter length_scale: m = 2;\n  parameter ambient_pressure_value: kg / (m * s ^ 2) = 4.5;",
        )
        .replace(
            "  relation force on fluid {",
            "  relation ambient_pressure_definition on fluid {\n    ambient_pressure - ambient_pressure_value = 0;\n  }\n  relation force on fluid {",
        )
        .replace(
            "relation x_upper_zero on x_upper { trace(velocity) = 0; }",
            &direct_normal_pressure_relation("ambient_pressure", operator),
        )
}

fn port_closed_normal_pressure_source(terminal_operator: char) -> String {
    let direct = source_with_normal_pressure('+');
    let instances = r#"instance fluid_boundary: NewtonianBoundary2d(
    body = fluid,
    face = x_upper,
    velocity = velocity,
    pressure = pressure,
    dynamic_viscosity = mu
  );
  instance ambient_boundary: NormalPressureTraction2d(
    body = fluid,
    face = x_upper,
    pressure = ambient_pressure
  );
  connect fluid_boundary.mechanical, ambient_boundary.mechanical;"#;
    format!(
        "{}\n{}",
        PORT_COMPONENTS.replace(
            "mechanical.traction - normal(isotropic_lift(pressure))",
            &format!("mechanical.traction {terminal_operator} normal(isotropic_lift(pressure))"),
        ),
        direct.replace(
            &direct_normal_pressure_relation("ambient_pressure", '+'),
            instances
        )
    )
}

#[test]
fn lowers_exact_meaning_without_numerical_choices() {
    let program = compile_program(SOURCE);
    let model = lower_steady_incompressible_stokes_cartesian_2d(&program)
        .expect("exact canonical Stokes model lowers");

    assert_eq!(model.bounds(), &[[0.0, 2.0], [-1.0, 1.0]]);
    assert_eq!(model.dynamic_viscosity(), 2.5);
    assert_eq!(
        model
            .dynamic_viscosity_expression()
            .parameter_fields()
            .len(),
        1
    );
    assert!(
        (model
            .force_potential_expression()
            .evaluate(&[0.4, 0.0])
            .unwrap()
            - 0.6)
            .abs()
            < 1.0e-15
    );
    assert_eq!(
        model.force_potential_expression().parameter_fields().len(),
        2
    );
    assert_ne!(model.velocity(), model.pressure());
    assert_ne!(model.pressure(), model.force_potential());
    assert_ne!(
        model.force_potential_definition(),
        model.momentum_relation()
    );
    assert_ne!(
        model.momentum_relation(),
        model.incompressibility_relation()
    );
    let correspondence = model.common.correspondence();
    assert_eq!(
        correspondence.formulation.kind,
        FormulationKind::MixedGalerkin
    );
    assert_eq!(
        correspondence.formulation.boundary_treatment,
        BoundaryTreatment::ExplicitTraceFluxLaws
    );
    assert_eq!(correspondence.formulation.velocity_trial, model.velocity());
    assert_eq!(correspondence.formulation.velocity_test, model.velocity());
    assert_eq!(correspondence.formulation.pressure_trial, model.pressure());
    assert_eq!(correspondence.formulation.pressure_test, model.pressure());
    assert_eq!(correspondence.law.source, model.force_potential());
    assert_eq!(
        correspondence.law.source_definition,
        model.force_potential_definition()
    );
    assert_eq!(
        correspondence.formulation.rules,
        [
            MixedFormulationRule::MomentumTestPairing,
            MixedFormulationRule::StressDivergenceByParts,
            MixedFormulationRule::PressureVelocityCoupling,
            MixedFormulationRule::ContinuityConstraintPairing,
            MixedFormulationRule::SourcePairing,
            MixedFormulationRule::ExplicitBoundaryLaw,
        ]
    );
    assert_eq!(correspondence.law.boundary_relations.len(), 4);
    model
        .common
        .replay_correspondence()
        .expect("recognized mixed correspondence replays exactly");
    let mut stale = model.common.clone();
    stale.correspondence.law.boundary_relations.pop();
    assert!(stale.replay_correspondence().is_err());
    for axis in 0..2 {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            assert!(matches!(
                model
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("complete Stokes boundary inventory")
                    .disposition(),
                PhysicalBoundaryDisposition::TraceZero
            ));
        }
    }
    assert!(
        model
            .boundary_inventory()
            .boundary(2, BoundarySide::Lower)
            .is_none()
    );
    assert_eq!(model.common().boundary_relations().len(), 4);
    assert!(
        model
            .common()
            .boundary_relations()
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    );
    for binding in model.common().boundary_relations() {
        assert!(program.edges().iter().any(|edge| {
            edge.kind() == EdgeKind::AppliesOn
                && edge.from() == binding.relation()
                && edge.to() == binding.boundary()
        }));
    }
}

#[test]
fn admits_additive_stokes_orientations_without_rewriting_the_model() {
    let additive = SOURCE
        .replace(
            "force_potential - load_scale * coordinate(0) / length_scale = 0;",
            "force_potential + -(load_scale * coordinate(0) / length_scale) = 0;",
        )
        .replace(
            ") - grad(force_potential) = 0;",
            ") + -grad(force_potential) = 0;",
        );
    let additive_program = compile_program(&additive);
    let additive_identity = (additive_program.model(), additive_program.revision());
    let additive_model = lower_steady_incompressible_stokes_cartesian_2d(&additive_program)
        .expect("additively grouped Stokes roles lower");
    assert_eq!(
        (additive_program.model(), additive_program.revision()),
        additive_identity
    );
    assert_eq!(additive_model.dynamic_viscosity(), 2.5);
    assert!(
        (additive_model
            .force_potential_expression()
            .evaluate(&[0.4, 0.0])
            .unwrap()
            - 0.6)
            .abs()
            < 1.0e-15
    );

    let reversed = SOURCE
        .replace(
            "force_potential - load_scale * coordinate(0) / length_scale = 0;",
            "load_scale * coordinate(0) / length_scale - force_potential = 0;",
        )
        .replace("-div(\n      2 * mu", "div(\n      2 * mu")
        .replace(
            ") - grad(force_potential) = 0;",
            ") + grad(force_potential) = 0;",
        )
        .replace("div(velocity) = 0;", "-div(velocity) = 0;")
        .replace("trace(velocity) = 0;", "-trace(velocity) = 0;");
    lower_steady_incompressible_stokes_cartesian_2d(&compile_program(&reversed))
        .expect("whole-equation Stokes sign reversal lowers");
}

#[test]
fn rejects_inconsistent_or_duplicate_additive_stokes_roles() {
    assert_rejected(&SOURCE.replace(
        ") - grad(force_potential) = 0;",
        ") + grad(force_potential) = 0;",
    ));
    assert_rejected(&SOURCE.replace(
        ") - grad(force_potential) = 0;",
        ") - grad(force_potential) - grad(force_potential) = 0;",
    ));
}

#[test]
fn retains_exact_flux_zero_meaning_without_claiming_a_realization() {
    let source = SOURCE.replace(
        "relation x_upper_zero on x_upper { trace(velocity) = 0; }",
        "relation x_upper_zero on x_upper {\n    normal(\n      2 * mu * symmetric_part(grad(velocity))\n      - isotropic_lift(pressure)\n    ) = 0;\n  }",
    );
    let model = lower_steady_incompressible_stokes_cartesian_2d(&compile_program(&source))
        .expect("exact zero Newtonian traction remains canonical meaning");
    assert!(matches!(
        model
            .boundary_inventory()
            .boundary(0, BoundarySide::Upper)
            .expect("x-upper inventory")
            .disposition(),
        PhysicalBoundaryDisposition::FluxZero
    ));
    let pressure = model
        .normal_pressure(0, BoundarySide::Upper)
        .expect("zero traction is exactly zero normal pressure");
    assert_eq!(pressure.coefficient_field(), None);
    assert_eq!(pressure.definition_relation(), None);
    assert_eq!(pressure.expression().constant_value(), Some(0.0));
}

#[test]
fn lowers_direct_normal_pressure_without_leaking_it_into_shared_boundary_meaning() {
    let model = lower_steady_incompressible_stokes_cartesian_2d(&compile_program(
        &source_with_normal_pressure('+'),
    ))
    .expect("direct static-pressure boundary lowers");
    let disposition = model
        .boundary_inventory()
        .boundary(0, BoundarySide::Upper)
        .expect("x-upper inventory")
        .disposition();
    let PhysicalBoundaryDisposition::Prescribed(law) = disposition else {
        panic!("normal pressure must remain one generic prescribed law");
    };
    assert_eq!(law.quantity(), PhysicalBoundaryQuantity::Flux);
    let pressure = model
        .normal_pressure(0, BoundarySide::Upper)
        .expect("normal-pressure tape");
    assert!(pressure.coefficient_field().is_some());
    assert!(pressure.definition_relation().is_some());
    assert_eq!(pressure.expression().constant_value(), Some(4.5));
}

#[test]
fn lowers_natural_equality_normal_pressure_orientation() {
    let source = source_with_normal_pressure('+').replace(
        ") + normal(isotropic_lift(ambient_pressure)) = 0;",
        ") = -normal(isotropic_lift(ambient_pressure));",
    );
    let model = lower_steady_incompressible_stokes_cartesian_2d(&compile_program(&source))
        .expect("natural equality pressure orientation lowers");
    assert_eq!(
        model
            .normal_pressure(0, BoundarySide::Upper)
            .expect("normal-pressure tape")
            .expression()
            .constant_value(),
        Some(4.5)
    );
}

#[test]
fn port_closed_normal_pressure_has_the_same_stokes_tape_and_checks_terminal_sign() {
    let model = lower_steady_incompressible_stokes_cartesian_2d(&compile_program(
        &port_closed_normal_pressure_source('-'),
    ))
    .expect("port-closed static-pressure boundary lowers");
    let law = model
        .normal_pressure(0, BoundarySide::Upper)
        .expect("normal-pressure tape");
    assert_eq!(law.expression().constant_value(), Some(4.5));
    assert!(matches!(
        model
            .boundary_inventory()
            .boundary(0, BoundarySide::Upper)
            .expect("x-upper inventory")
            .disposition(),
        PhysicalBoundaryDisposition::Prescribed(law)
            if law.quantity() == PhysicalBoundaryQuantity::Flux
    ));

    assert_rejected(&port_closed_normal_pressure_source('+'));
}

#[test]
fn rejects_normal_pressure_sign_and_semantic_field_aliases() {
    assert_rejected(&source_with_normal_pressure('-'));

    for field in ["pressure", "force_potential"] {
        let source = SOURCE.replace(
            "relation x_upper_zero on x_upper { trace(velocity) = 0; }",
            &direct_normal_pressure_relation(field, '+'),
        );
        assert_rejected(&source);
    }
}

#[test]
fn fails_closed_on_operator_boundary_and_model_drift() {
    assert_rejected(&SOURCE.replace(
        "2 * mu * symmetric_part(grad(velocity))",
        "mu * symmetric_part(grad(velocity))",
    ));
    assert_rejected(&SOURCE.replace("- isotropic_lift(pressure)", "+ isotropic_lift(pressure)"));
    assert_rejected(&SOURCE.replace(
        "relation incompressibility on fluid { div(velocity) = 0; }",
        "",
    ));
    assert_rejected(&SOURCE.replace("trace(velocity) = 0;", "trace(pressure) = 0;"));
    assert_rejected(&SOURCE.replace(
        "parameter mu: kg / (m * s) = 2.5;",
        "parameter unused: 1 = 7;\n  parameter mu: kg / (m * s) = 2.5;",
    ));
    assert_rejected(&SOURCE.replace(
        "parameter mu: kg / (m * s) = 2.5;",
        "parameter mu: kg / (m * s) = 0;",
    ));
}

#[test]
fn lowers_exact_fixed_domain_transient_navier_stokes_meaning() {
    let program = compile_program(TRANSIENT_NAVIER_STOKES_SOURCE);
    let model = lower_transient_incompressible_navier_stokes_cartesian_2d(&program)
        .expect("exact conservative transient flow lowers");

    assert_eq!(model.bounds(), &[[0.0, 2.0], [-1.0, 1.0]]);
    assert_eq!(model.mass_density(), 1.25);
    assert_eq!(model.dynamic_viscosity(), 0.125);
    assert_eq!(model.mass_density_expression().parameter_fields().len(), 1);
    assert_eq!(
        model
            .dynamic_viscosity_expression()
            .parameter_fields()
            .len(),
        1
    );
    assert_eq!(
        model.force_potential_expression().constant_value(),
        Some(0.0)
    );
    assert_eq!(
        model.conservative_body_force(&[0.3, -0.2]).unwrap(),
        [0.0; 2]
    );
    assert_ne!(model.velocity(), model.pressure());
    assert_ne!(model.pressure(), model.force_potential());
    assert_ne!(
        model.force_potential_definition(),
        model.momentum_relation()
    );
    assert_ne!(
        model.momentum_relation(),
        model.incompressibility_relation()
    );
    for axis in 0..2 {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            assert!(matches!(
                model
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("complete transient-flow boundary inventory")
                    .disposition(),
                PhysicalBoundaryDisposition::TraceZero
            ));
        }
    }
    assert_eq!(model.boundary_relations().len(), 4);

    let projected = model.common_projection();
    let mixed = projected.mixed_galerkin_correspondence();
    assert_eq!(mixed.formulation.kind, FormulationKind::MixedGalerkin);
    assert_eq!(mixed.formulation.velocity_trial, model.velocity());
    assert_eq!(mixed.formulation.pressure_trial, model.pressure());
    let mut stale_mixed = mixed.clone();
    stale_mixed.law.boundary_relations.pop();
    assert!(
        projected
            .replay_mixed_galerkin_correspondence(&stale_mixed)
            .is_err()
    );

    let correspondence = super::integral_conservative_correspondence(&model);
    assert_eq!(
        correspondence.formulation.kind,
        FormulationKind::IntegralConservative
    );
    assert_eq!(correspondence.formulation.domain, model.domain());
    assert_eq!(
        correspondence.formulation.momentum_unknown,
        model.velocity()
    );
    assert_eq!(correspondence.formulation.pressure_role, model.pressure());
    assert_eq!(
        correspondence.formulation.boundary_treatment,
        BoundaryTreatment::ExplicitTraceFluxLaws
    );
    assert_eq!(
        correspondence.formulation.rules,
        [
            IntegralConservativeRule::ArbitrarySubdomainBalance,
            IntegralConservativeRule::TransientStorageIntegral,
            IntegralConservativeRule::PhysicalMomentumFlux,
            IntegralConservativeRule::PhysicalStressFlux,
            IntegralConservativeRule::BodySourceIntegral,
            IntegralConservativeRule::IncompressibilityFluxBalance,
            IntegralConservativeRule::ExplicitBoundaryLaw,
        ]
    );
    assert_eq!(correspondence.law.boundary_relations.len(), 4);

    let mut stale = correspondence.clone();
    stale.law.boundary_relations.pop();
    assert!(
        super::navier_stokes_integral_formulation::replay_integral_conservative_correspondence(
            &stale, &model
        )
        .is_err()
    );
}

#[test]
fn lowers_the_same_transient_flow_meaning_in_three_dimensions() {
    let source = transient_navier_stokes_source_3d();
    let model =
        lower_transient_incompressible_navier_stokes_cartesian_3d(&compile_program(&source))
            .expect("exact 3D conservative transient flow lowers");

    assert_eq!(model.bounds(), &[[0.0, 2.0], [-1.0, 1.0], [-2.0, 2.0]]);
    assert_eq!(
        model.conservative_body_force(&[0.2, -0.3, 0.4]).unwrap(),
        [0.0; 3]
    );
    for axis in 0..3 {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            assert!(matches!(
                model
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("complete 3D transient-flow boundary inventory")
                    .disposition(),
                PhysicalBoundaryDisposition::TraceZero
            ));
        }
    }
    assert_eq!(model.boundary_relations().len(), 6);
}

#[test]
fn three_dimensional_transient_flow_rejects_dimension_and_boundary_drift() {
    assert!(
        lower_transient_incompressible_navier_stokes_cartesian_3d(&compile_program(
            TRANSIENT_NAVIER_STOKES_SOURCE,
        ))
        .is_err()
    );

    let incomplete = transient_navier_stokes_source_3d()
        .replace(
            "  domain z_upper = boundary(fluid, axis = 2, side = upper);\n",
            "",
        )
        .replace(
            "  relation z_upper_zero on z_upper { trace(velocity) = 0; }\n",
            "",
        );
    assert!(
        lower_transient_incompressible_navier_stokes_cartesian_3d(&compile_program(&incomplete))
            .is_err()
    );
}

#[test]
fn transient_navier_stokes_requires_exact_velocity_pair_and_density_identity() {
    assert_transient_navier_stokes_rejected(&TRANSIENT_NAVIER_STOKES_SOURCE.replace(
        "outer_product(left = velocity, right = velocity)",
        "outer_product(left = velocity, right = velocity + velocity)",
    ));

    let distinct_density = TRANSIENT_NAVIER_STOKES_SOURCE
        .replace(
            "parameter mu: kg / (m * s) = 0.125;",
            "parameter rho_flux: kg / m ^ 3 = 1.25;\n  parameter mu: kg / (m * s) = 0.125;",
        )
        .replace(
            "div(rho * outer_product(left = velocity, right = velocity))",
            "div(rho_flux * outer_product(left = velocity, right = velocity))",
        );
    assert_transient_navier_stokes_rejected(&distinct_density);

    assert_transient_navier_stokes_rejected(
        &TRANSIENT_NAVIER_STOKES_SOURCE.replace("kg / m ^ 3 = 1.25", "kg / m ^ 3 = 0"),
    );
    assert_transient_navier_stokes_rejected(
        &TRANSIENT_NAVIER_STOKES_SOURCE.replace("kg / (m * s) = 0.125", "kg / (m * s) = -0.125"),
    );

    let spatial_density = TRANSIENT_NAVIER_STOKES_SOURCE
        .replace(
            "parameter mu: kg / (m * s) = 0.125;",
            "parameter inverse_length: 1 / m = 0.01;\n  parameter mu: kg / (m * s) = 0.125;",
        )
        .replace(
            "rho * derivative(velocity)",
            "(rho * (1 + inverse_length * coordinate(0))) * derivative(velocity)",
        )
        .replace(
            "div(rho * outer_product(left = velocity, right = velocity))",
            "div((rho * (1 + inverse_length * coordinate(0))) * outer_product(left = velocity, right = velocity))",
        );
    assert_transient_navier_stokes_rejected(&spatial_density);
}

#[test]
fn transient_navier_stokes_rejects_hidden_ale_velocity() {
    assert_transient_navier_stokes_rejected(
        &TRANSIENT_NAVIER_STOKES_SOURCE
            .replace(
                "variable pressure: kg / (m * s ^ 2) on fluid;",
                "variable mesh_velocity: vector<m / s, 2> on fluid;\n  variable pressure: kg / (m * s ^ 2) on fluid;",
            )
            .replace(
                "outer_product(left = velocity, right = velocity)",
                "outer_product(left = velocity - mesh_velocity, right = velocity)",
            ),
    );
}

#[test]
fn transient_navier_stokes_normalizes_only_explicit_whole_sign_reversal() {
    let marker = "  relation momentum on fluid {";
    let relation = TRANSIENT_NAVIER_STOKES_SOURCE
        .find(marker)
        .expect("momentum Relation marker");
    let residual_start = relation + marker.len();
    let residual_end = residual_start
        + TRANSIENT_NAVIER_STOKES_SOURCE[residual_start..]
            .find("= 0;")
            .expect("momentum residual terminator");
    let residual = TRANSIENT_NAVIER_STOKES_SOURCE[residual_start..residual_end].trim();
    let reversed = format!(
        "{}\n    -({residual}) {}",
        &TRANSIENT_NAVIER_STOKES_SOURCE[..residual_start],
        &TRANSIENT_NAVIER_STOKES_SOURCE[residual_end..]
    );

    lower_transient_incompressible_navier_stokes_cartesian_2d(&compile_program(&reversed))
        .expect("explicit negation of the complete residual is normalized");
}
