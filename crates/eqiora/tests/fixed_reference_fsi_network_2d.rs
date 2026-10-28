#[path = "fixed_reference_fsi_network_2d/source.rs"]
mod source;
use source::Material;

use std::num::NonZeroUsize;

use eqiora::meshing::{CellId, MeshEntity, MeshQualityGate, SimplicialMesh};
use eqiora::realization::{
    CoupledFieldwiseRealizationRequest, DiscretizationMethod, MeshArtifactReference, MeshKind,
    RealizationCapabilities, RealizationRevision, SemanticRevision, SpatialDimensionSupport,
    TargetCapabilities, VectorLayoutKind, resolve_coupled_fieldwise,
};
use eqiora::solver::{
    LinearSolver, LinearSolverBackend, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER,
    ReductionPolicy, SolverPlan,
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
    run_network(
        &network_source(),
        0,
        reference_solver(),
        &REFERENCE_LINEAR_SOLVER,
    );
}

#[test]
fn two_fluid_pressure_spaces_close_through_one_prestrained_solid() {
    let source = source::network(
        TWO_REGION_SOURCE,
        &[
            (
                "fluid_left",
                Material::Fluid {
                    density: 2.0,
                    viscosity: 0.5,
                },
            ),
            (
                "solid",
                Material::Solid {
                    density: 3.0,
                    mu: 2.0,
                    lambda: 1.0,
                },
            ),
            (
                "fluid_right",
                Material::Fluid {
                    density: 5.0,
                    viscosity: 1.25,
                },
            ),
        ],
    );
    run_network(&source, 1, reference_solver(), &REFERENCE_LINEAR_SOLVER);
}

#[test]
fn native_network_solver_substitution_preserves_physical_acceptance() {
    let source = network_source();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        reference_solver().relative_tolerance(),
        reference_solver().absolute_tolerance(),
        reference_solver().maximum_iterations(),
    )
    .unwrap();
    run_network(&source, 0, solver, &REFERENCE_LINEAR_SOLVER);
}

fn run_network(
    source: &str,
    prestrain_component: usize,
    solver: SolverPlan,
    backend: &dyn LinearSolverBackend,
) {
    let document = eqiora::api::ModelDocument::compile("three-domain-network.eqi", source)
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
        solver,
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
    let mut previous = prestrained_network_state(
        document.program(),
        &model,
        &plan,
        &mesh,
        &partition,
        prestrain_component,
    );
    let resolved = resolve_coupled_fieldwise(
        &CoupledFieldwiseRealizationRequest::explicit(
            document.program().model(),
            SemanticRevision::new(model.semantic_revision()),
            RealizationRevision::new(1),
            plan,
        ),
        fixed_reference_fsi_requirements_2d(&model),
        &RealizationCapabilities::cartesian_product(
            [DiscretizationMethod::ContinuousGalerkin],
            [(
                MeshKind::ImportedAffineSimplicial,
                SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
            )],
            [VectorLayoutKind::Replicated],
            backend.capabilities(),
            TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
        )
        .unwrap(),
    )
    .expect("ordinary host realization admits the plural FSI network");
    assert_eq!(resolved.plan().solver(), solver);
    // Continue from each accepted physical State through the same exact policy.
    for _ in 0..2 {
        let accepted = finalize_resolved_fixed_reference_fsi_step_2d(
            &model,
            &resolved,
            mesh_reference,
            &mesh,
            &partition,
            &previous,
        )
        .expect("plural fixed-reference operator finalizes")
        .solve(backend)
        .expect("plural fixed-reference Run satisfies its acceptance");

        assert_eq!(
            accepted
                .numerical_evidence()
                .solve_report()
                .solver_provider(),
            backend.provider()
        );
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
        if model.fluids().count() == 2 {
            assert_eq!(model.solids().count(), 1);
            let pressures = model
                .fluids()
                .map(|fluid| fluid.pressure())
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(pressures.len(), 2);
            for fluid in model.fluids() {
                let maximum = accepted
                    .state()
                    .coefficients(fluid.velocity().downcast().unwrap())
                    .unwrap()
                    .map(|(_, _, _, value)| value.abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    maximum > 1.0e-10,
                    "each fluid must respond to the coupled solid prestrain"
                );
            }
        } else {
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
        previous = accepted.state().clone();
    }
}

fn prestrained_network_state(
    program: &eqiora::sem::KernelProgram,
    model: &eqiora_numerics::fsi::FixedReferenceFsiCartesianModel2d,
    plan: &eqiora::realization::CoupledFieldwiseRealizationPlan,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    prestrain_component: usize,
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
    let mut fields = Vec::new();
    for fluid in model.fluids() {
        let fluid_domain = fluid.domain().downcast().unwrap();
        let mut fluid_velocity = vector(fluid_domain, [0.0; 2]);
        for cell in partition.domain_cells(fluid_domain).unwrap() {
            for component in 0..2 {
                fluid_velocity.push((MeshEntity::new(2, cell.index()), 0, component, 0.0));
            }
        }
        fields.push((fluid.velocity().downcast().unwrap(), fluid_velocity));
        fields.push((
            fluid.pressure().downcast().unwrap(),
            partition
                .domain_vertices(fluid_domain)
                .unwrap()
                .iter()
                .map(|vertex| (MeshEntity::new(0, vertex.index()), 0, 0, 0.0))
                .collect(),
        ));
    }
    for solid in model.solids() {
        let domain = solid.continuum().domain().downcast().unwrap();
        let mut displacement = vector(domain, [0.0; 2]);
        for (entity, _, component, value) in &mut displacement {
            if *component == prestrain_component {
                let point = mesh.vertices()[entity.index()].as_slice();
                if point == [1.0, 0.5] {
                    *value = 0.02;
                }
                if point == [2.0, 0.5] {
                    *value = -0.01;
                }
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
    source::network(
        TWO_REGION_SOURCE,
        &[
            (
                "solid_left",
                Material::Solid {
                    density: 3.0,
                    mu: 2.0,
                    lambda: 1.0,
                },
            ),
            (
                "fluid",
                Material::Fluid {
                    density: 2.0,
                    viscosity: 0.5,
                },
            ),
            (
                "solid_right",
                Material::Solid {
                    density: 3.0,
                    mu: 7.0,
                    lambda: 1.0,
                },
            ),
        ],
    )
}
