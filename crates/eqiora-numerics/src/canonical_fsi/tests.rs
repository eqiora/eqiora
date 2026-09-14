use eqiora_compiler::compile;
use eqiora_core::diagnostic::codes;
use eqiora_graph::{EdgeKind, GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::BoundarySide;
use eqiora_sem::KernelProgram;

use super::{
    lower_ale_fsi_cartesian_2d, lower_ale_fsi_cartesian_3d, lower_fixed_reference_fsi_cartesian_2d,
};
use crate::canonical_boundary::PhysicalBoundaryDisposition;

const SOURCE: &str = r#"
public connector VelocityTractionBoundary {
  trace velocity: m / s;
  flux traction: kg / (m * s ^ 2);
  shape spatial_vector;
  frame spatial;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}

public component NewtonianInterface2d(
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

public component ElasticInterface2d(
  support body: volume(ambient_dimension = 2),
  support face: boundary(parent = body),
  variable displacement: vector<m, 2> on body,
  variable velocity: vector<m / s, 2> on body,
  parameter mu: kg / (m * s ^ 2),
  parameter lambda: kg / (m * s ^ 2),
  port mechanical: VelocityTractionBoundary over face
) {

  relation interface on face {
    trace(velocity) - mechanical.velocity = 0;
    normal(
      2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))
    ) - mechanical.traction = 0;
  }
}

model Main() {
  domain fluid = box(0, 1, 0, 1);
  domain fluid_x_lower = boundary(fluid, axis = 0, side = lower);
  domain fluid_x_upper = boundary(fluid, axis = 0, side = upper);
  domain fluid_y_lower = boundary(fluid, axis = 1, side = lower);
  domain fluid_y_upper = boundary(fluid, axis = 1, side = upper);
  domain solid = box(1, 2, 0, 1);
  domain solid_x_lower = boundary(solid, axis = 0, side = lower);
  domain solid_x_upper = boundary(solid, axis = 0, side = upper);
  domain solid_y_lower = boundary(solid, axis = 1, side = lower);
  domain solid_y_upper = boundary(solid, axis = 1, side = upper);


  state fluid_velocity: vector<m / s, 2> on fluid;
  variable pressure: kg / (m * s ^ 2) on fluid;
  variable fluid_load: kg / (m * s ^ 2) on fluid;
  state displacement: vector<m, 2> on solid;
  state solid_velocity: vector<m / s, 2> on solid;
  variable solid_load: kg / (m * s ^ 2) on solid;

  parameter fluid_density: kg / m ^ 3 = 2;
  parameter viscosity: kg / (m * s) = 0.5;
  parameter solid_density: kg / m ^ 3 = 3;
  parameter mu: kg / (m * s ^ 2) = 4;
  parameter lambda: kg / (m * s ^ 2) = 5;
  parameter zero_pressure: kg / (m * s ^ 2) = 0;

  relation fluid_load_definition on fluid { fluid_load - zero_pressure = 0; }
  relation fluid_momentum on fluid {
    fluid_density * derivative(fluid_velocity)
      - div(
        2 * viscosity * symmetric_part(grad(fluid_velocity))
        - isotropic_lift(pressure)
      )
      - grad(fluid_load) = 0;
  }
  relation incompressibility on fluid { div(fluid_velocity) = 0; }

  relation solid_load_definition on solid { solid_load - zero_pressure = 0; }
  relation kinematics on solid {
    derivative(displacement) - solid_velocity = 0;
  }
  relation solid_momentum on solid {
    solid_density * derivative(solid_velocity)
      - div(
        2 * mu * symmetric_part(grad(displacement))
        + lambda * isotropic_lift(div(displacement))
      )
      - grad(solid_load) = 0;
  }

  relation fluid_x_lower_zero on fluid_x_lower { trace(fluid_velocity) = 0; }
  relation fluid_y_lower_zero on fluid_y_lower { trace(fluid_velocity) = 0; }
  relation fluid_y_upper_zero on fluid_y_upper { trace(fluid_velocity) = 0; }
  relation solid_x_upper_zero on solid_x_upper { trace(solid_velocity) = 0; }
  relation solid_y_lower_zero on solid_y_lower { trace(solid_velocity) = 0; }
  relation solid_y_upper_zero on solid_y_upper { trace(solid_velocity) = 0; }

  instance fluid_interface: NewtonianInterface2d(
    body = fluid,
    face = fluid_x_upper,
    velocity = fluid_velocity,
    pressure = pressure,
    dynamic_viscosity = viscosity
  );
  instance solid_interface: ElasticInterface2d(
    body = solid,
    face = solid_x_lower,
    displacement = displacement,
    velocity = solid_velocity,
    mu = mu,
    lambda = lambda
  );
  connect fluid_interface.mechanical, solid_interface.mechanical;
}
"#;

fn compile_program(source: &str) -> KernelProgram {
    let mut compiled = compile("fixed-reference-fsi.eqi", source).expect("source compiles");
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).expect("transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("model validates")
}

fn assert_rejected(source: &str) {
    assert_eq!(
        lower_fixed_reference_fsi_cartesian_2d(&compile_program(source))
            .expect_err("altered model must fail closed")
            .code(),
        codes::INVALID_SPATIAL_LOWERING
    );
}

fn ale_source() -> String {
    format!(
        "public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2]\n  = component(left, 0) * component(right, 1);\n{}",
        SOURCE.replace(
            "fluid_density * derivative(fluid_velocity)\n      - div(",
            "fluid_density * derivative(fluid_velocity)\n      + div(fluid_density * outer_product(left = fluid_velocity, right = fluid_velocity))\n      - div(",
        )
    )
}

fn ale_source_3d() -> String {
    ale_source()
        .replace("vector<m / s, 2>", "vector<m / s, 3>")
        .replace("vector<m, 2>", "vector<m, 3>")
        .replace("ambient_dimension = 2", "ambient_dimension = 3")
        .replace(
            "domain fluid = box(0, 1, 0, 1);",
            "domain fluid = box(0, 1, 0, 1, 0, 1);",
        )
        .replace(
            "domain solid = box(1, 2, 0, 1);",
            "domain solid = box(1, 2, 0, 1, 0, 1);",
        )
        .replace(
            "  domain fluid_y_upper = boundary(fluid, axis = 1, side = upper);",
            "  domain fluid_y_upper = boundary(fluid, axis = 1, side = upper);\n  domain fluid_z_lower = boundary(fluid, axis = 2, side = lower);\n  domain fluid_z_upper = boundary(fluid, axis = 2, side = upper);",
        )
        .replace(
            "  domain solid_y_upper = boundary(solid, axis = 1, side = upper);",
            "  domain solid_y_upper = boundary(solid, axis = 1, side = upper);\n  domain solid_z_lower = boundary(solid, axis = 2, side = lower);\n  domain solid_z_upper = boundary(solid, axis = 2, side = upper);",
        )
        .replace(
            "  relation fluid_y_upper_zero on fluid_y_upper { trace(fluid_velocity) = 0; }",
            "  relation fluid_y_upper_zero on fluid_y_upper { trace(fluid_velocity) = 0; }\n  relation fluid_z_lower_zero on fluid_z_lower { trace(fluid_velocity) = 0; }\n  relation fluid_z_upper_zero on fluid_z_upper { trace(fluid_velocity) = 0; }",
        )
        .replace(
            "  relation solid_y_upper_zero on solid_y_upper { trace(solid_velocity) = 0; }",
            "  relation solid_y_upper_zero on solid_y_upper { trace(solid_velocity) = 0; }\n  relation solid_z_lower_zero on solid_z_lower { trace(solid_velocity) = 0; }\n  relation solid_z_upper_zero on solid_z_upper { trace(solid_velocity) = 0; }",
        )
}

#[test]
fn recognizes_exact_package_neutral_fixed_reference_meaning() {
    let program = compile_program(SOURCE);
    let model = lower_fixed_reference_fsi_cartesian_2d(&program)
        .expect("exact fixed-reference FSI semantics lower");
    let fluids = model.fluids().collect::<Vec<_>>();
    let [fluid] = fluids.as_slice() else {
        panic!("fixture must lower one exact fluid Domain")
    };
    let solids = model.solids().collect::<Vec<_>>();
    let [solid] = solids.as_slice() else {
        panic!("fixture must lower one exact solid Domain")
    };
    let interfaces = model.interfaces().collect::<Vec<_>>();
    let [interface] = interfaces.as_slice() else {
        panic!("fixture must lower one exact Connection")
    };
    let fluid_endpoint = interface
        .endpoint(fluid.domain())
        .expect("Connection owns the fluid Domain endpoint");
    let solid_endpoint = interface
        .endpoint(solid.continuum().domain())
        .expect("Connection owns the solid Domain endpoint");

    assert_eq!(fluid.bounds(), &[[0.0, 1.0], [0.0, 1.0]]);
    assert_eq!(solid.continuum().bounds(), &[[1.0, 2.0], [0.0, 1.0]]);
    assert_eq!(fluid.mass_density(), 2.0);
    assert_eq!(fluid.dynamic_viscosity(), 0.5);
    assert_eq!(solid.mass_density(), 3.0);
    assert_eq!(solid.continuum().shear_modulus(), 4.0);
    assert_eq!(solid.continuum().first_lame_parameter(), 5.0);
    assert_eq!(interface.axis(), 0);
    assert_eq!(fluid_endpoint.side(), BoundarySide::Upper);
    assert_eq!(solid_endpoint.side(), BoundarySide::Lower);
    assert_ne!(fluid_endpoint.port(), solid_endpoint.port());
    assert_eq!(
        fluid.conservative_body_force(&[0.5, 0.5]).unwrap(),
        [0.0; 2]
    );
    assert_eq!(
        solid
            .continuum()
            .conservative_body_force(&[1.5, 0.5])
            .unwrap(),
        [0.0; 2]
    );
    assert!(matches!(
        fluid
            .boundary_inventory()
            .boundary(0, BoundarySide::Upper)
            .expect("fluid interface entry")
            .disposition(),
        PhysicalBoundaryDisposition::PortBinding { .. }
    ));
    assert!(matches!(
        solid
            .continuum()
            .boundary_inventory()
            .boundary(0, BoundarySide::Lower)
            .expect("solid interface entry")
            .disposition(),
        PhysicalBoundaryDisposition::PortBinding { .. }
    ));
    assert_ne!(
        fluid.force_potential_definition(),
        fluid.momentum_relation()
    );
    assert_ne!(
        fluid.momentum_relation(),
        fluid.incompressibility_relation()
    );
    assert_ne!(
        solid.continuum().load_definition_relation(),
        solid.kinematic_relation()
    );
    assert_ne!(
        solid.kinematic_relation(),
        solid.continuum().equilibrium_relation()
    );
    for bindings in [
        fluid.boundary_relations(),
        solid.continuum().boundary_relations(),
    ] {
        assert!(!bindings.is_empty());
        assert!(bindings.windows(2).all(|pair| pair[0] < pair[1]));
        for binding in bindings {
            assert!(program.edges().iter().any(|edge| {
                edge.kind() == EdgeKind::AppliesOn
                    && edge.from() == binding.relation()
                    && edge.to() == binding.boundary()
            }));
        }
    }
}

#[test]
fn recognizes_conservative_transient_fsi_without_adding_ale_meaning() {
    let source = ale_source();
    let program = compile_program(&source);
    let model = lower_ale_fsi_cartesian_2d(&program)
        .expect("conservative transient-fluid FSI semantics lower");

    assert_eq!(model.fluid().bounds(), &[[0.0, 1.0], [0.0, 1.0]]);
    assert_eq!(
        model.solid().continuum().bounds(),
        &[[1.0, 2.0], [0.0, 1.0]]
    );
    assert_eq!(model.fluid().mass_density(), 2.0);
    assert_eq!(model.fluid().dynamic_viscosity(), 0.5);
    assert_eq!(model.interface().axis(), 0);
    assert_eq!(
        model
            .interface()
            .endpoint(model.fluid().domain())
            .unwrap()
            .side(),
        BoundarySide::Upper
    );
    assert_eq!(
        model
            .interface()
            .endpoint(model.solid().continuum().domain())
            .unwrap()
            .side(),
        BoundarySide::Lower
    );
    assert!(lower_fixed_reference_fsi_cartesian_2d(&program).is_err());
    assert!(lower_ale_fsi_cartesian_2d(&compile_program(SOURCE)).is_err());
}

#[test]
fn recognizes_the_same_conservative_ale_fsi_meaning_in_three_dimensions() {
    let source = ale_source_3d();
    let model = lower_ale_fsi_cartesian_3d(&compile_program(&source))
        .expect("exact 3D conservative ALE FSI semantics lower");

    assert_eq!(model.fluid().bounds(), &[[0.0, 1.0]; 3]);
    assert_eq!(
        model.solid().continuum().bounds(),
        &[[1.0, 2.0], [0.0, 1.0], [0.0, 1.0]]
    );
    assert_eq!(model.interface().axis(), 0);
    assert_eq!(
        model
            .interface()
            .endpoint(model.fluid().domain())
            .unwrap()
            .side(),
        BoundarySide::Upper
    );
    assert_eq!(
        model
            .interface()
            .endpoint(model.solid().continuum().domain())
            .unwrap()
            .side(),
        BoundarySide::Lower
    );
    assert_eq!(
        model.fluid().conservative_body_force(&[0.2; 3]).unwrap(),
        [0.0; 3]
    );
    assert_eq!(
        model
            .solid()
            .continuum()
            .conservative_body_force(&[1.2, 0.2, 0.2])
            .unwrap(),
        [0.0; 3]
    );
}

#[test]
fn three_dimensional_ale_fsi_rejects_dimension_and_boundary_drift() {
    assert!(lower_ale_fsi_cartesian_3d(&compile_program(&ale_source())).is_err());

    let incomplete = ale_source_3d()
        .replace(
            "  domain solid_z_upper = boundary(solid, axis = 2, side = upper);\n",
            "",
        )
        .replace(
            "  relation solid_z_upper_zero on solid_z_upper { trace(solid_velocity) = 0; }\n",
            "",
        );
    assert!(lower_ale_fsi_cartesian_3d(&compile_program(&incomplete)).is_err());
}

#[test]
fn rejects_a_steady_fluid_that_omits_inertia() {
    assert_rejected(&SOURCE.replace(
        "fluid_density * derivative(fluid_velocity)\n      - div(",
        "-div(",
    ));
}

#[test]
fn rejects_noncoincident_interface_geometry() {
    let altered = SOURCE.replace(
        "domain solid = box(1, 2, 0, 1);",
        "domain solid = box(1.25, 2.25, 0, 1);",
    );
    let diagnostic = compile("fixed-reference-fsi.eqi", &altered)
        .expect_err("noncoincident physical interface must fail before lowering");
    assert_eq!(diagnostic[0].code(), codes::LANGUAGE_TYPE_ERROR);
}

#[test]
fn rejects_meaning_outside_the_closed_two_domain_network() {
    assert_rejected(&SOURCE.replace(
        "relation incompressibility on fluid { div(fluid_velocity) = 0; }",
        "relation incompressibility on fluid { div(fluid_velocity) = 0; }\n  relation hidden on fluid { pressure = 0; }",
    ));
}
