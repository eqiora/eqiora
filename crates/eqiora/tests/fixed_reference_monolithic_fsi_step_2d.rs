use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora::artifact::{
    AffineTriangleMeshCellsV1, GeometryMeshCorrespondenceEnvelopeV1,
    MeshProductionLineageEnvelopeV1, ModelEnvelope,
};
use eqiora::geometry::{GeometryGraph, PlanarTopologyHandle};
use eqiora::realization::{SolveRoot, TransformationNode};
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonBackwardEuler, CommonInitialField, CommonInitialValues,
    CommonMethodRequest, CommonScopedSpatialPolicy, CommonSolvePolicy, CommonSpatialPolicy,
    IncompressibleScalingRequest2d, common::PhysicalBoundaryDisposition,
    fsi::FixedReferenceFsiCartesianModel2d, fsi::lower_fixed_reference_fsi_cartesian_2d,
    resolve_common_plan,
};
use support::fixed_reference_fsi::{
    direct_document, exact_spatial_witness, execute_initial_step, packaged_document,
};

mod support;

macro_rules! only_fluid {
    ($model:expr) => {
        $model
            .fluids()
            .next()
            .expect("fixture owns one fluid Region")
    };
}
macro_rules! only_solid {
    ($model:expr) => {
        $model
            .solids()
            .next()
            .expect("fixture owns one solid Region")
    };
}
macro_rules! only_interface {
    ($model:expr) => {
        $model
            .interfaces()
            .next()
            .expect("fixture owns one FSI Connection")
    };
}

#[derive(Debug, PartialEq)]
struct SemanticObservation {
    fluid_bounds: [[u64; 2]; 2],
    solid_bounds: [[u64; 2]; 2],
    fluid_density: u64,
    fluid_viscosity: u64,
    solid_density: u64,
    solid_mu: u64,
    solid_lambda: u64,
    interface_axis: usize,
    fluid_side: eqiora::kernel::BoundarySide,
    solid_side: eqiora::kernel::BoundarySide,
}

#[test]
fn direct_and_exact_packages_share_one_fixed_reference_fsi_meaning() {
    let direct = direct_document();
    let packaged = packaged_document();

    let direct_model = lower_fixed_reference_fsi_cartesian_2d(direct.program())
        .expect("direct fixed-reference FSI semantics lower");
    let packaged_model = lower_fixed_reference_fsi_cartesian_2d(packaged.model().program())
        .expect("exact-package fixed-reference FSI semantics lower");
    assert_eq!(observe(&direct_model), observe(&packaged_model));
    let direct_spatial = exact_spatial_witness(direct.program(), &direct_model);
    let packaged_spatial = exact_spatial_witness(packaged.model().program(), &packaged_model);
    assert_eq!(direct_spatial, packaged_spatial);

    for model in [&direct_model, &packaged_model] {
        let interface = only_interface!(model);
        let fluid_side = interface.endpoint(only_fluid!(model).domain()).unwrap();
        let solid_side = interface
            .endpoint(only_solid!(model).continuum().domain())
            .unwrap();
        assert_eq!(interface.axis(), 0);
        assert_ne!(fluid_side.boundary(), solid_side.boundary());
        assert_ne!(fluid_side.port(), solid_side.port());
        assert_eq!(
            only_fluid!(model)
                .conservative_body_force(&[0.25, 0.5])
                .unwrap(),
            [0.0; 2]
        );
        assert_eq!(
            only_solid!(model)
                .continuum()
                .load_potential_expression()
                .evaluate(&[1.5, 0.5])
                .unwrap(),
            0.0
        );
        assert_eq!(live_boundary_count(model), 2);
    }
}

#[test]
fn fixed_reference_monolithic_fsi_step_2d() {
    let direct = direct_document();
    let packaged = packaged_document();
    let direct_model = lower_fixed_reference_fsi_cartesian_2d(direct.program())
        .expect("direct fixed-reference FSI semantics lower");
    let packaged_model = lower_fixed_reference_fsi_cartesian_2d(packaged.model().program())
        .expect("exact-package fixed-reference FSI semantics lower");
    let direct = execute_initial_step(direct.program(), &direct_model);
    let packaged = execute_initial_step(packaged.model().program(), &packaged_model);

    assert_eq!(direct.operator, direct.replayed_operator);
    assert_eq!(packaged.operator, packaged.replayed_operator);
    assert!(direct.physical_operator.agrees(&packaged.physical_operator));
    direct
        .physical_operator
        .rejects_wrong_coordinates(&packaged.physical_operator);
    direct.physical_operator.assert_residual_compatible(
        &packaged.physical_operator,
        &direct.solution,
        &packaged.solution,
    );
    for execution in [&direct, &packaged] {
        let solution = &execution.solution;
        let fields = execution.fields;
        assert!(matches!(
            solution.realization_graph().root(),
            SolveRoot::Linear(_)
        ));
        assert!(matches!(
            solution.realization_graph().transformations(),
            [
                TransformationNode::BackwardEulerElimination { .. },
                TransformationNode::ConformingTraceQuotient { .. }
            ]
        ));
        let evidence = solution.numerical_evidence();
        let fluid_velocity = vector_coefficients(solution, fields.fluid_velocity);
        let solid_velocity = vector_coefficients(solution, fields.solid_velocity);
        let interface_midpoint = fluid_velocity
            .keys()
            .filter_map(|entity| (entity.dimension() == 0).then_some(*entity))
            .find(|vertex| {
                solid_velocity.contains_key(vertex)
                    && fluid_velocity[vertex]
                        .into_iter()
                        .any(|component| component.abs() > 1.0e-10)
            })
            .expect("the exact shared interface has nonzero motion");
        assert_eq!(
            fluid_velocity[&interface_midpoint],
            solid_velocity[&interface_midpoint]
        );
        assert!(evidence.pressure_constant_action_norm() > 1.0e-10);
        assert!(evidence.residual_norm() < 1.0e-9);
        assert!(evidence.continuity_residual_norm() < 1.0e-9);
        assert!(evidence.kinematic_residual_norm() < 1.0e-14);
        assert_eq!(evidence.interface_velocity_jump_norm(), 0.0);
        assert!(!evidence.interface_actions().is_empty());
        assert!(evidence.interface_action_imbalance_norm() < 1.0e-9);
        assert!(evidence.energy_balance().defect().abs() < 1.0e-9);
        assert_eq!(
            fluid_velocity
                .keys()
                .filter(|entity| entity.dimension() == 2)
                .count(),
            4
        );
        assert_eq!(
            vector_coefficients(solution, fields.solid_displacement)
                .keys()
                .filter(|entity| entity.dimension() == 0)
                .count(),
            6
        );
    }
}

#[test]
fn common_plan_matches_independent_two_step_scientific_composition() {
    let direct = direct_document();
    let independent_model = lower_fixed_reference_fsi_cartesian_2d(direct.program())
        .expect("independent fixed-reference FSI meaning lowers");
    let independent_spatial =
        support::fixed_reference_fsi::spatial_context(direct.program(), &independent_model);
    let independent_execution = support::fixed_reference_fsi::execution_context(
        direct.program(),
        &independent_model,
        &independent_spatial,
    );
    let independent_first = support::fixed_reference_fsi::solve_step(
        &independent_model,
        &independent_spatial,
        &independent_execution,
        &support::fixed_reference_fsi::prestrained_state(
            direct.program(),
            &independent_spatial,
            &independent_execution,
        ),
    );
    let independent_second = support::fixed_reference_fsi::solve_step(
        &independent_model,
        &independent_spatial,
        &independent_execution,
        &support::fixed_reference_fsi::state_from_solution(
            &independent_spatial,
            &independent_first.solution,
        ),
    );

    // Geometry selection names deliberately encode no equation or Cartesian role.
    let graph = GeometryGraph::new();
    let fluid = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let solid = graph.rectangle([1.0, 2.0], [0.0, 1.0]).unwrap();
    let fluid_edges = fluid.boundaries();
    let solid_edges = solid.boundaries();
    let partition = graph
        .partition(&fluid, &solid, [fluid_edges[1], solid_edges[0]])
        .unwrap();
    let geometry = graph
        .build(
            &partition,
            &BTreeMap::from([
                ("patch-z".to_owned(), vec![fluid.region().into()]),
                (
                    "edge-9".to_owned(),
                    vec![PlanarTopologyHandle::from(fluid_edges[0])],
                ),
                (
                    "edge-2".to_owned(),
                    vec![PlanarTopologyHandle::from(fluid_edges[1])],
                ),
                (
                    "edge-7".to_owned(),
                    vec![PlanarTopologyHandle::from(fluid_edges[2])],
                ),
                (
                    "edge-4".to_owned(),
                    vec![PlanarTopologyHandle::from(fluid_edges[3])],
                ),
                ("patch-a".to_owned(), vec![solid.region().into()]),
                (
                    "edge-8".to_owned(),
                    vec![PlanarTopologyHandle::from(solid_edges[0])],
                ),
                (
                    "edge-1".to_owned(),
                    vec![PlanarTopologyHandle::from(solid_edges[1])],
                ),
                (
                    "edge-6".to_owned(),
                    vec![PlanarTopologyHandle::from(solid_edges[2])],
                ),
                (
                    "edge-3".to_owned(),
                    vec![PlanarTopologyHandle::from(solid_edges[3])],
                ),
            ]),
        )
        .unwrap();
    let compile_parameters = &[
        (
            "fluid_density",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(2.0)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "fluid_viscosity",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(0.5)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "solid_density",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(3.0)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "solid_mu",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(4.0)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "solid_lambda",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(2.0)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "zero_pressure",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(0.0)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
    ];
    let mut compile_bindings = vec![
        (
            "fluid",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("patch-z").unwrap(),
                parent: None,
            },
        ),
        (
            "solid",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("patch-a").unwrap(),
                parent: None,
            },
        ),
        (
            "fluid_x_lower",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("edge-9").unwrap(),
                parent: Some(geometry.entity_set("patch-z").unwrap()),
            },
        ),
        (
            "fluid_x_upper",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("edge-2").unwrap(),
                parent: Some(geometry.entity_set("patch-z").unwrap()),
            },
        ),
        (
            "fluid_y_lower",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("edge-7").unwrap(),
                parent: Some(geometry.entity_set("patch-z").unwrap()),
            },
        ),
        (
            "fluid_y_upper",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("edge-4").unwrap(),
                parent: Some(geometry.entity_set("patch-z").unwrap()),
            },
        ),
        (
            "solid_x_lower",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("edge-8").unwrap(),
                parent: Some(geometry.entity_set("patch-a").unwrap()),
            },
        ),
        (
            "solid_x_upper",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("edge-1").unwrap(),
                parent: Some(geometry.entity_set("patch-a").unwrap()),
            },
        ),
        (
            "solid_y_lower",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("edge-6").unwrap(),
                parent: Some(geometry.entity_set("patch-a").unwrap()),
            },
        ),
        (
            "solid_y_upper",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("edge-3").unwrap(),
                parent: Some(geometry.entity_set("patch-a").unwrap()),
            },
        ),
    ];
    compile_bindings.extend(compile_parameters.iter().map(|(name, value)| {
        (
            *name,
            eqiora::compiler::StaticBindingValue::Expression(value),
        )
    }));
    let common_document = eqiora::api::ModelDocument::compile_selected(
        "fixed-reference-fsi.eqi",
        include_str!("../../../examples/fixed-reference-fsi.eqi"),
        "FixedReferenceFsi2d",
        &compile_bindings,
    )
    .expect("component-only FSI compiles against exact Geometry");
    let common_model = ModelEnvelope::from_program(common_document.program()).unwrap();
    let policy = AffineTriangleMeshCellsV1::new([2, 2]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_adjacent_rectangle_partition_affine_triangles(
            &geometry,
            policy.cells(),
        )
        .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_rectangle_v1_resources(
        policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let resources = AuthenticatedCommonMesh::adjacent_partition(
        geometry,
        mesh.clone(),
        correspondence,
        production,
    )
    .unwrap();
    let model_digest = common_model.digest().unwrap();
    let fluid_domain = common_document.aliases()["fluid"].downcast().unwrap();
    let solid_domain = common_document.aliases()["solid"].downcast().unwrap();
    let scoped = CommonMethodRequest::Scoped(vec![
        CommonScopedSpatialPolicy::new(
            model_digest.clone(),
            fluid_domain,
            CommonSpatialPolicy::MiniP1,
        ),
        CommonScopedSpatialPolicy::new(model_digest.clone(), solid_domain, CommonSpatialPolicy::P1),
    ]);
    let requested = CommonSolvePolicy::Linear(
        eqiora_numerics::CommonLinearRequest::exact(
            eqiora::solver::SolverPlan::new(
                eqiora::solver::LinearSolver::MinimumResidual,
                1.0e-11,
                1.0e-13,
                NonZeroUsize::new(20_000).unwrap(),
            )
            .unwrap()
            .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
            .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
            eqiora::solver::REFERENCE_SOLVER_PROVIDER,
        )
        .unwrap(),
    );
    let common_plans = [
        (
            "manual legacy scaling",
            Some(IncompressibleScalingRequest2d::from_si(Some(2.0), Some(0.5), Some(4.0)).unwrap()),
        ),
        ("automatic scaling", None),
    ]
    .map(|(label, scaling)| {
        let plan = resolve_common_plan(
            &common_model,
            resources.clone(),
            scoped.clone(),
            requested,
            scaling,
            Some(CommonBackwardEuler::from_seconds(0.05).unwrap()),
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap()
        .as_fsi()
        .cloned()
        .expect("fixture retains its admitted fsi Plan");
        (label, plan)
    });
    let fields = [
        common_document.aliases()["definition.fluid_velocity"]
            .downcast()
            .unwrap(),
        common_document.aliases()["definition.fluid_pressure"]
            .downcast()
            .unwrap(),
        common_document.aliases()["definition.solid_velocity"]
            .downcast()
            .unwrap(),
        common_document.aliases()["definition.solid_displacement"]
            .downcast()
            .unwrap(),
    ];
    let fluid_vertices = common_plans[0].1.field_vertex_indices(fields[0]).unwrap();
    let fluid_cells = common_plans[0]
        .1
        .field_domain_cell_indices(fields[0])
        .unwrap();
    let solid_vertices = common_plans[0].1.field_vertex_indices(fields[2]).unwrap();
    let solid_displacement = solid_vertices
        .iter()
        .map(|&vertex| {
            if mesh.mesh().vertices()[vertex].as_slice() == [1.0, 0.5] {
                [0.02, 0.0]
            } else {
                [0.0; 2]
            }
        })
        .collect::<Vec<_>>();
    let common_states = common_plans
        .iter()
        .map(|(label, common_plan)| {
            let common_initial = common_plan
                .initial_state(
                    0.0,
                    vec![
                        CommonInitialField::new(
                            model_digest.clone(),
                            fields[0],
                            Some(CommonInitialValues::Vector2(
                                vec![[0.0; 2]; fluid_vertices.len()].into_boxed_slice(),
                            )),
                            Some(CommonInitialValues::Vector2(
                                vec![[0.0; 2]; fluid_cells.len()].into_boxed_slice(),
                            )),
                        )
                        .unwrap(),
                        CommonInitialField::new(
                            model_digest.clone(),
                            fields[1],
                            Some(CommonInitialValues::Scalar(
                                vec![0.0; fluid_vertices.len()].into_boxed_slice(),
                            )),
                            None,
                        )
                        .unwrap(),
                        CommonInitialField::new(
                            model_digest.clone(),
                            fields[2],
                            Some(CommonInitialValues::Vector2(
                                vec![[0.0; 2]; solid_vertices.len()].into_boxed_slice(),
                            )),
                            None,
                        )
                        .unwrap(),
                        CommonInitialField::new(
                            model_digest.clone(),
                            fields[3],
                            Some(CommonInitialValues::Vector2(
                                solid_displacement.clone().into_boxed_slice(),
                            )),
                            None,
                        )
                        .unwrap(),
                    ],
                )
                .unwrap();
            let common_first = common_plan
                .advance(&common_initial, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            let common_second = common_plan
                .advance(&common_first, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            (*label, common_plan, common_first, common_second)
        })
        .collect::<Vec<_>>();

    let coordinate_key = |point: &[f64]| {
        point
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let cell_key = |mesh: &eqiora::meshing::SimplicialMesh, cell: usize| {
        let mut vertices = mesh.cells()[cell]
            .iter()
            .map(|&vertex| coordinate_key(&mesh.vertices()[vertex]))
            .collect::<Vec<_>>();
        vertices.sort();
        vertices
    };
    let assert_vectors_close =
        |label: &str, left: &BTreeMap<Vec<u64>, [f64; 2]>, right: &BTreeMap<Vec<u64>, [f64; 2]>| {
            assert_eq!(
                left.keys().collect::<Vec<_>>(),
                right.keys().collect::<Vec<_>>(),
                "{label} support"
            );
            for (key, left) in left {
                let right = right[key];
                assert!(
                    left.iter()
                        .copied()
                        .zip(right)
                        .all(|(left, right)| (left - right).abs() < 1.0e-9),
                    "{label} differs at {key:?}: {left:?} vs {right:?}"
                );
            }
        };
    let assert_scalars_close =
        |label: &str, left: &BTreeMap<Vec<u64>, f64>, right: &BTreeMap<Vec<u64>, f64>| {
            assert_eq!(
                left.keys().collect::<Vec<_>>(),
                right.keys().collect::<Vec<_>>(),
                "{label} support"
            );
            for (key, left) in left {
                let right = right[key];
                assert!(
                    (left - right).abs() < 1.0e-9,
                    "{label} differs at {key:?}: {left:?} vs {right:?}"
                );
            }
        };
    let independent_fields = independent_first.fields;
    for (scaling, _common_plan, common_first, common_second) in &common_states {
        for (common, independent) in [
            (common_first, &independent_first.solution),
            (common_second, &independent_second.solution),
        ] {
            let common_velocity = [fields[0], fields[2]]
                .into_iter()
                .flat_map(|field| vector_state_coefficients(common.fsi_fields().unwrap(), field))
                .filter(|(entity, _)| entity.dimension() == 0)
                .map(|(entity, value)| {
                    (
                        coordinate_key(&mesh.mesh().vertices()[entity.index()]),
                        value,
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let independent_velocity = [
                independent_fields.fluid_velocity,
                independent_fields.solid_velocity,
            ]
            .into_iter()
            .flat_map(|field| vector_coefficients(independent, field))
            .filter(|(entity, _)| entity.dimension() == 0)
            .map(|(entity, value)| {
                (
                    coordinate_key(&independent_spatial.mesh.vertices()[entity.index()]),
                    value,
                )
            })
            .collect::<BTreeMap<_, _>>();
            assert_vectors_close(
                &format!("{scaling}: shared fluid/solid vertex velocity"),
                &common_velocity,
                &independent_velocity,
            );

            let common_bubbles = vector_state_coefficients(common.fsi_fields().unwrap(), fields[0])
                .into_iter()
                .filter(|(entity, _)| entity.dimension() == 2)
                .map(|(cell, value)| (cell_key(mesh.mesh(), cell.index()), value))
                .collect::<BTreeMap<_, _>>();
            let independent_bubbles =
                vector_coefficients(independent, independent_fields.fluid_velocity)
                    .into_iter()
                    .filter(|(entity, _)| entity.dimension() == 2)
                    .map(|(cell, value)| (cell_key(&independent_spatial.mesh, cell.index()), value))
                    .collect::<BTreeMap<_, _>>();
            assert_eq!(
                common_bubbles.keys().collect::<Vec<_>>(),
                independent_bubbles.keys().collect::<Vec<_>>(),
                "MINI fluid cell bubble support"
            );
            for (key, left) in &common_bubbles {
                let right = independent_bubbles[key];
                assert!(
                    left.iter()
                        .copied()
                        .zip(right)
                        .all(|(left, right)| (left - right).abs() < 1.0e-9),
                    "MINI fluid cell bubbles differ at {key:?}: {left:?} vs {right:?}"
                );
            }

            let common_pressure =
                scalar_state_coefficients(common.fsi_fields().unwrap(), fields[1])
                    .into_iter()
                    .filter(|(entity, _)| entity.dimension() == 0)
                    .map(|(vertex, value)| {
                        (
                            coordinate_key(&mesh.mesh().vertices()[vertex.index()]),
                            value,
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
            let independent_pressure =
                scalar_coefficients(independent, independent_fields.fluid_pressure)
                    .into_iter()
                    .filter(|(entity, _)| entity.dimension() == 0)
                    .map(|(vertex, value)| {
                        (
                            coordinate_key(&independent_spatial.mesh.vertices()[vertex.index()]),
                            value,
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
            assert_scalars_close(
                "gauge-free fluid pressure",
                &common_pressure,
                &independent_pressure,
            );

            let common_displacement =
                vector_state_coefficients(common.fsi_fields().unwrap(), fields[3])
                    .into_iter()
                    .filter(|(entity, _)| entity.dimension() == 0)
                    .map(|(entity, value)| {
                        (
                            coordinate_key(&mesh.mesh().vertices()[entity.index()]),
                            value,
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
            let independent_displacement =
                vector_coefficients(independent, independent_fields.solid_displacement)
                    .into_iter()
                    .filter(|(entity, _)| entity.dimension() == 0)
                    .map(|(entity, value)| {
                        (
                            coordinate_key(&independent_spatial.mesh.vertices()[entity.index()]),
                            value,
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
            assert_vectors_close(
                &format!("{scaling}: solid displacement"),
                &common_displacement,
                &independent_displacement,
            );
        }
        assert_eq!(
            [common_first.time_s(), common_second.time_s()],
            [0.05, 0.10]
        );
    }
}

fn observe(model: &FixedReferenceFsiCartesianModel2d) -> SemanticObservation {
    let interface = only_interface!(model);
    let fluid_side = interface.endpoint(only_fluid!(model).domain()).unwrap();
    let solid_side = interface
        .endpoint(only_solid!(model).continuum().domain())
        .unwrap();
    SemanticObservation {
        fluid_bounds: only_fluid!(model)
            .bounds()
            .map(|axis| axis.map(f64::to_bits)),
        solid_bounds: only_solid!(model)
            .continuum()
            .bounds()
            .map(|axis| axis.map(f64::to_bits)),
        fluid_density: only_fluid!(model).mass_density().to_bits(),
        fluid_viscosity: only_fluid!(model).dynamic_viscosity().to_bits(),
        solid_density: only_solid!(model).mass_density().to_bits(),
        solid_mu: only_solid!(model).continuum().shear_modulus().to_bits(),
        solid_lambda: only_solid!(model)
            .continuum()
            .first_lame_parameter()
            .to_bits(),
        interface_axis: interface.axis(),
        fluid_side: fluid_side.side(),
        solid_side: solid_side.side(),
    }
}

fn live_boundary_count(model: &FixedReferenceFsiCartesianModel2d) -> usize {
    [
        only_fluid!(model).boundary_inventory(),
        only_solid!(model).continuum().boundary_inventory(),
    ]
    .into_iter()
    .flat_map(|inventory| {
        [0, 1].into_iter().flat_map(move |axis| {
            [
                eqiora::kernel::BoundarySide::Lower,
                eqiora::kernel::BoundarySide::Upper,
            ]
            .into_iter()
            .map(move |side| inventory.boundary(axis, side).expect("complete boundary"))
        })
    })
    .filter(|entry| {
        matches!(
            entry.disposition(),
            PhysicalBoundaryDisposition::PortBinding { .. }
        )
    })
    .count()
}

fn vector_coefficients(
    solution: &eqiora_numerics::fsi::ResolvedFixedReferenceFsiSolution2d,
    field: eqiora::Id<eqiora::kinds::Field>,
) -> BTreeMap<eqiora::meshing::MeshEntity, [f64; 2]> {
    let mut values = BTreeMap::<_, [Option<f64>; 2]>::new();
    for (entity, slot, component, value) in solution
        .state()
        .coefficients(field)
        .expect("accepted exact Field")
    {
        assert_eq!(slot, 0);
        values.entry(entity).or_insert([None; 2])[component] = Some(value);
    }
    values
        .into_iter()
        .map(|(entity, components)| {
            (
                entity,
                components.map(|value| value.expect("complete vector component")),
            )
        })
        .collect()
}

fn vector_state_coefficients(
    state: &eqiora_numerics::fsi::FixedReferenceFsiState<2>,
    field: eqiora::Id<eqiora::kinds::Field>,
) -> BTreeMap<eqiora::meshing::MeshEntity, [f64; 2]> {
    let mut values = BTreeMap::<_, [Option<f64>; 2]>::new();
    for (entity, slot, component, value) in state.coefficients(field).expect("accepted exact Field")
    {
        assert_eq!(slot, 0);
        values.entry(entity).or_insert([None; 2])[component] = Some(value);
    }
    values
        .into_iter()
        .map(|(entity, components)| {
            (
                entity,
                components.map(|value| value.expect("complete vector component")),
            )
        })
        .collect()
}

fn scalar_coefficients(
    solution: &eqiora_numerics::fsi::ResolvedFixedReferenceFsiSolution2d,
    field: eqiora::Id<eqiora::kinds::Field>,
) -> BTreeMap<eqiora::meshing::MeshEntity, f64> {
    solution
        .state()
        .coefficients(field)
        .expect("accepted exact Field")
        .map(|(entity, slot, component, value)| {
            assert_eq!((slot, component), (0, 0));
            (entity, value)
        })
        .collect()
}

fn scalar_state_coefficients(
    state: &eqiora_numerics::fsi::FixedReferenceFsiState<2>,
    field: eqiora::Id<eqiora::kinds::Field>,
) -> BTreeMap<eqiora::meshing::MeshEntity, f64> {
    state
        .coefficients(field)
        .expect("accepted exact Field")
        .map(|(entity, slot, component, value)| {
            assert_eq!((slot, component), (0, 0));
            (entity, value)
        })
        .collect()
}
