use std::num::NonZeroUsize;

use eqiora::meshing::{CellId, MeshEntity, MeshQualityGate, SimplicialMesh};
use eqiora::realization::{
    CoupledFieldwiseRealizationRequest, MeshArtifactReference, RealizationCapabilities,
    RealizationRevision, SemanticRevision, resolve_coupled_fieldwise,
};
use eqiora::solver::{
    LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan,
};
use eqiora::{DimExponents, DynQuantity};
use eqiora_numerics::fsi::{
    FixedReferenceFsiPartition, FixedReferenceFsiScaleProfile2d, FixedReferenceFsiState,
    finalize_resolved_fixed_reference_fsi_step_2d, fixed_reference_fsi_plan_2d,
    fixed_reference_fsi_requirements_2d, lower_fixed_reference_fsi_cartesian_2d,
};

const TWO_REGION_SOURCE: &str =
    include_str!("../../../verify/fsi/fixed-reference-monolithic-step-2d/models/direct.eqi");

#[test]
fn ordinary_three_domain_two_connection_model_lowers_as_one_network() {
    let document = eqiora::api::ModelDocument::compile("solid-fluid-solid.eqi", &network_source())
        .expect("ordinary three-Domain Model compiles");
    let model = lower_fixed_reference_fsi_cartesian_2d(document.program())
        .expect("three-Domain/two-Connection FSI network lowers");

    assert_eq!(model.fluids().count(), 1);
    assert_eq!(model.solids().count(), 2);
    assert_eq!(model.interfaces().count(), 2);
    let mut shear_moduli = model
        .solids()
        .map(|solid| solid.continuum().shear_modulus().to_bits())
        .collect::<Vec<_>>();
    shear_moduli.sort_unstable();
    assert_eq!(shear_moduli, [2.0_f64.to_bits(), 7.0_f64.to_bits()]);
}

#[test]
fn ordinary_three_domain_two_connection_run_closes_prestrained_energy() {
    let document = eqiora::api::ModelDocument::compile("solid-fluid-solid.eqi", &network_source())
        .expect("ordinary three-Domain Model compiles");
    let model = lower_fixed_reference_fsi_cartesian_2d(document.program())
        .expect("three-Domain/two-Connection FSI network lowers");
    let mesh = network_mesh();
    let mesh_reference = MeshArtifactReference::from_sha256([0x35; 32]);
    let plan = fixed_reference_fsi_plan_2d(
        &model,
        mesh_reference,
        quantity(0.05, [0, 0, 1, 0, 0, 0, 0]),
        FixedReferenceFsiScaleProfile2d::new(
            quantity(3.0, [0, 1, 0, 0, 0, 0, 0]),
            quantity(0.5, [0, 1, -1, 0, 0, 0, 0]),
            quantity(4.0, [1, -1, -2, 0, 0, 0, 0]),
        )
        .unwrap(),
        reference_solver(),
    )
    .expect("plural exact FSI plan");
    let partition = FixedReferenceFsiPartition::<2>::new(
        &mesh,
        model
            .fluids()
            .map(|fluid| {
                (
                    fluid.domain().downcast().unwrap(),
                    cells_in_x_range(&mesh, fluid.bounds()[0]),
                )
            })
            .chain(model.solids().map(|solid| {
                (
                    solid.continuum().domain().downcast().unwrap(),
                    cells_in_x_range(&mesh, solid.continuum().bounds()[0]),
                )
            })),
        plan.spatial().trace_quotients(),
    )
    .expect("three exact Domain inventories replay both Connections");
    let previous = prestrained_network_state(document.program(), &model, &plan, &mesh, &partition);
    let resolved = resolve_coupled_fieldwise(
        &CoupledFieldwiseRealizationRequest::explicit(
            document.program().model(),
            SemanticRevision::new(model.semantic_revision()),
            RealizationRevision::new(1),
            plan,
        ),
        fixed_reference_fsi_requirements_2d(&model),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .expect("ordinary host realization admits the plural FSI network");
    let accepted = finalize_resolved_fixed_reference_fsi_step_2d(
        &model,
        &resolved,
        mesh_reference,
        &mesh,
        &partition,
        &previous,
    )
    .expect("plural fixed-reference operator finalizes")
    .solve(&REFERENCE_LINEAR_SOLVER)
    .expect("plural fixed-reference Run satisfies its acceptance");

    assert_eq!(accepted.state().fields().count(), 6);
    assert!(accepted.numerical_evidence().residual_norm() < 1.0e-9);
    assert!(
        accepted
            .numerical_evidence()
            .energy_balance()
            .defect()
            .abs()
            < 1.0e-9
    );
    assert_eq!(
        accepted
            .numerical_evidence()
            .interface_actions()
            .iter()
            .map(|action| action.connection().erase())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        2
    );
    let maximum_velocity = model
        .solids()
        .flat_map(|solid| {
            accepted
                .state()
                .coefficients(solid.velocity().downcast().unwrap())
                .unwrap()
                .map(|(_, _, _, value)| value.abs())
        })
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_velocity > 1.0e-10,
        "prestrain must drive the accepted Run"
    );
    let mut interface_displacement = model
        .solids()
        .map(|solid| {
            let x = if solid.continuum().bounds()[0][0] == 0.0 {
                1.0
            } else {
                2.0
            };
            let vertex = mesh
                .vertices()
                .iter()
                .position(|point| point.as_slice() == [x, 0.5])
                .unwrap();
            let displacement = accepted
                .state()
                .coefficients(solid.continuum().displacement().downcast().unwrap())
                .unwrap()
                .find(|(entity, slot, component, _)| {
                    *entity == MeshEntity::new(0, vertex) && *slot == 0 && *component == 0
                })
                .unwrap()
                .3;
            (solid.continuum().shear_modulus(), displacement.abs())
        })
        .collect::<Vec<_>>();
    interface_displacement.sort_by_key(|(shear, _)| shear.to_bits());
    assert!(
        (interface_displacement[0].1 - interface_displacement[1].1).abs() > 1.0e-10,
        "distinct Region-local prestrains must not collapse to one solid state"
    );
}

fn prestrained_network_state(
    program: &eqiora::sem::KernelProgram,
    model: &eqiora_numerics::fsi::FixedReferenceFsiCartesianModel2d,
    plan: &eqiora::realization::CoupledFieldwiseRealizationPlan,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
) -> FixedReferenceFsiState<2> {
    let vector = |domain, value: [f64; 2]| {
        partition
            .domain_vertices(domain)
            .unwrap()
            .iter()
            .flat_map(move |vertex| {
                value
                    .into_iter()
                    .enumerate()
                    .map(move |(component, value)| {
                        (MeshEntity::new(0, vertex.index()), 0, component, value)
                    })
            })
            .collect::<Vec<_>>()
    };
    let fluid = model.fluids().next().unwrap();
    let fluid_domain = fluid.domain().downcast().unwrap();
    let mut fluid_velocity = vector(fluid_domain, [0.0; 2]);
    for cell in partition.domain_cells(fluid_domain).unwrap() {
        for component in 0..2 {
            fluid_velocity.push((MeshEntity::new(2, cell.index()), 0, component, 0.0));
        }
    }
    let mut fields = vec![
        (fluid.velocity().downcast().unwrap(), fluid_velocity),
        (
            fluid.pressure().downcast().unwrap(),
            partition
                .domain_vertices(fluid_domain)
                .unwrap()
                .iter()
                .map(|vertex| (MeshEntity::new(0, vertex.index()), 0, 0, 0.0))
                .collect(),
        ),
    ];
    for solid in model.solids() {
        let domain = solid.continuum().domain().downcast().unwrap();
        let mut displacement = vector(domain, [0.0; 2]);
        let interface_x = if solid.continuum().bounds()[0][0] == 0.0 {
            (1.0, 0.02)
        } else {
            (2.0, -0.01)
        };
        for (entity, _, component, value) in &mut displacement {
            if *component == 0 && mesh.vertices()[entity.index()].as_slice() == [interface_x.0, 0.5]
            {
                *value = interface_x.1;
            }
        }
        fields.push((
            solid.velocity().downcast().unwrap(),
            vector(domain, [0.0; 2]),
        ));
        fields.push((
            solid.continuum().displacement().downcast().unwrap(),
            displacement,
        ));
    }
    FixedReferenceFsiState::new(program, plan, mesh, partition, fields)
        .expect("complete exact plural prestrained State")
}

fn cells_in_x_range(mesh: &SimplicialMesh, bounds: [f64; 2]) -> Vec<CellId> {
    mesh.cells()
        .iter()
        .enumerate()
        .filter_map(|(index, cell)| {
            let x = cell
                .iter()
                .map(|&vertex| mesh.vertices()[vertex][0])
                .sum::<f64>()
                / 3.0;
            (bounds[0] < x && x < bounds[1]).then_some(CellId::new(index))
        })
        .collect()
}

fn network_mesh() -> SimplicialMesh {
    let vertices = [0.0, 0.5, 1.0]
        .into_iter()
        .flat_map(|y| [0.0, 1.0, 2.0, 3.0].into_iter().map(move |x| vec![x, y]))
        .collect();
    let mut cells = Vec::new();
    for row in 0..2 {
        for column in 0..3 {
            let lower_left = row * 4 + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + 4;
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
}

fn quantity(value: f64, dimensions: [i32; 7]) -> DynQuantity {
    DynQuantity::new(value, DimExponents::from_integers(dimensions).unwrap())
}

fn reference_solver() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(20_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

fn network_source() -> String {
    let declarations = TWO_REGION_SOURCE
        .split_once("model Main()")
        .expect("fixture declarations precede Main")
        .0;
    format!("{declarations}{NETWORK_MODEL}")
}

const NETWORK_MODEL: &str = r#"
model Main() {
  domain solid_left = box(0, 1, 0, 1);
  domain solid_left_x_lower = boundary(solid_left, axis = 0, side = lower);
  domain solid_left_x_upper = boundary(solid_left, axis = 0, side = upper);
  domain solid_left_y_lower = boundary(solid_left, axis = 1, side = lower);
  domain solid_left_y_upper = boundary(solid_left, axis = 1, side = upper);

  domain fluid = box(1, 2, 0, 1);
  domain fluid_x_lower = boundary(fluid, axis = 0, side = lower);
  domain fluid_x_upper = boundary(fluid, axis = 0, side = upper);
  domain fluid_y_lower = boundary(fluid, axis = 1, side = lower);
  domain fluid_y_upper = boundary(fluid, axis = 1, side = upper);

  domain solid_right = box(2, 3, 0, 1);
  domain solid_right_x_lower = boundary(solid_right, axis = 0, side = lower);
  domain solid_right_x_upper = boundary(solid_right, axis = 0, side = upper);
  domain solid_right_y_lower = boundary(solid_right, axis = 1, side = lower);
  domain solid_right_y_upper = boundary(solid_right, axis = 1, side = upper);

  state fluid_velocity: vector<m / s, 2> on fluid;
  variable fluid_pressure: kg / (m * s ^ 2) on fluid;
  variable fluid_load_potential: kg / (m * s ^ 2) on fluid;
  parameter fluid_density: kg / m ^ 3 = 2;
  parameter fluid_viscosity: kg / (m * s) = 0.5;
  parameter zero_pressure: kg / (m * s ^ 2) = 0;
  relation fluid_load_definition on fluid { fluid_load_potential - zero_pressure = 0; }
  relation fluid_momentum on fluid {
    fluid_density * derivative(fluid_velocity)
      - div(2 * fluid_viscosity * symmetric_part(grad(fluid_velocity))
        - isotropic_lift(fluid_pressure)) - grad(fluid_load_potential) = 0;
  }
  relation incompressibility on fluid { div(fluid_velocity) = 0; }

  state solid_left_displacement: vector<m, 2> on solid_left;
  state solid_left_velocity: vector<m / s, 2> on solid_left;
  variable solid_left_load_potential: kg / (m * s ^ 2) on solid_left;
  parameter solid_left_density: kg / m ^ 3 = 3;
  parameter solid_left_mu: kg / (m * s ^ 2) = 2;
  parameter solid_left_lambda: kg / (m * s ^ 2) = 1;
  relation solid_left_load_definition on solid_left {
    solid_left_load_potential - zero_pressure = 0;
  }
  relation solid_left_kinematics on solid_left {
    derivative(solid_left_displacement) - solid_left_velocity = 0;
  }
  relation solid_left_momentum on solid_left {
    solid_left_density * derivative(solid_left_velocity)
      - div(2 * solid_left_mu * symmetric_part(grad(solid_left_displacement))
        + solid_left_lambda * isotropic_lift(div(solid_left_displacement)))
      - grad(solid_left_load_potential) = 0;
  }

  state solid_right_displacement: vector<m, 2> on solid_right;
  state solid_right_velocity: vector<m / s, 2> on solid_right;
  variable solid_right_load_potential: kg / (m * s ^ 2) on solid_right;
  parameter solid_right_density: kg / m ^ 3 = 3;
  parameter solid_right_mu: kg / (m * s ^ 2) = 7;
  parameter solid_right_lambda: kg / (m * s ^ 2) = 1;
  relation solid_right_load_definition on solid_right {
    solid_right_load_potential - zero_pressure = 0;
  }
  relation solid_right_kinematics on solid_right {
    derivative(solid_right_displacement) - solid_right_velocity = 0;
  }
  relation solid_right_momentum on solid_right {
    solid_right_density * derivative(solid_right_velocity)
      - div(2 * solid_right_mu * symmetric_part(grad(solid_right_displacement))
        + solid_right_lambda * isotropic_lift(div(solid_right_displacement)))
      - grad(solid_right_load_potential) = 0;
  }

  instance fluid_boundary: NewtonianMechanicalInterface2d(
    body = fluid,
    exterior = boundaries(fluid_x_lower, fluid_x_upper, fluid_y_lower, fluid_y_upper),
    velocity = fluid_velocity, pressure = fluid_pressure,
    dynamic_viscosity = fluid_viscosity
  );
  instance solid_left_boundary: ElastodynamicMechanicalInterface2d(
    body = solid_left,
    exterior = boundaries(solid_left_x_lower, solid_left_x_upper,
      solid_left_y_lower, solid_left_y_upper),
    displacement = solid_left_displacement, velocity = solid_left_velocity,
    mu = solid_left_mu, lambda = solid_left_lambda
  );
  instance solid_right_boundary: ElastodynamicMechanicalInterface2d(
    body = solid_right,
    exterior = boundaries(solid_right_x_lower, solid_right_x_upper,
      solid_right_y_lower, solid_right_y_upper),
    displacement = solid_right_displacement, velocity = solid_right_velocity,
    mu = solid_right_mu, lambda = solid_right_lambda
  );

  instance solid_left_x_lower_zero: ZeroVelocity2d(body = solid_left, face = solid_left_x_lower);
  instance solid_left_y_lower_zero: ZeroVelocity2d(body = solid_left, face = solid_left_y_lower);
  instance solid_left_y_upper_zero: ZeroVelocity2d(body = solid_left, face = solid_left_y_upper);
  instance fluid_y_lower_zero: ZeroVelocity2d(body = fluid, face = fluid_y_lower);
  instance fluid_y_upper_zero: ZeroVelocity2d(body = fluid, face = fluid_y_upper);
  instance solid_right_x_upper_zero: ZeroVelocity2d(body = solid_right, face = solid_right_x_upper);
  instance solid_right_y_lower_zero: ZeroVelocity2d(body = solid_right, face = solid_right_y_lower);
  instance solid_right_y_upper_zero: ZeroVelocity2d(body = solid_right, face = solid_right_y_upper);

  connect solid_left_boundary.mechanical[boundary = solid_left_x_lower], solid_left_x_lower_zero.mechanical;
  connect solid_left_boundary.mechanical[boundary = solid_left_y_lower], solid_left_y_lower_zero.mechanical;
  connect solid_left_boundary.mechanical[boundary = solid_left_y_upper], solid_left_y_upper_zero.mechanical;
  connect fluid_boundary.mechanical[boundary = fluid_y_lower], fluid_y_lower_zero.mechanical;
  connect fluid_boundary.mechanical[boundary = fluid_y_upper], fluid_y_upper_zero.mechanical;
  connect solid_right_boundary.mechanical[boundary = solid_right_x_upper], solid_right_x_upper_zero.mechanical;
  connect solid_right_boundary.mechanical[boundary = solid_right_y_lower], solid_right_y_lower_zero.mechanical;
  connect solid_right_boundary.mechanical[boundary = solid_right_y_upper], solid_right_y_upper_zero.mechanical;
  connect solid_left_boundary.mechanical[boundary = solid_left_x_upper],
    fluid_boundary.mechanical[boundary = fluid_x_lower];
  connect fluid_boundary.mechanical[boundary = fluid_x_upper],
    solid_right_boundary.mechanical[boundary = solid_right_x_lower];
}
"#;
