use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    CartesianMeshCellsV2, CartesianMeshEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1,
    MeshProductionLineageEnvelopeV1, ModelEnvelope,
};
use eqiora::geometry::{CanonicalGeometryV1, GeometryGraph, PlanarTopologyHandle};
use eqiora::meshing::{MeshEntity, MeshTopology};
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonElasticityPlan, CommonResult, CommonSolvePolicy,
    CommonSpatialPolicy, resolve_common_plan,
};
use serde_json::{Value, json};

const SOURCE: &str = r#"public component MixedBoundaryElasticity2d(
  support body: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = body),
  support x_upper: boundary(parent = body),
  support y_lower: boundary(parent = body),
  support y_upper: boundary(parent = body),
  parameter mu: kg / (m * s ^ 2),
  parameter lambda: kg / (m * s ^ 2),
  parameter length_scale: m
) {

  variable displacement: vector<m, 2> on body;
  variable load_potential: kg / (m * s ^ 2) on body;
  relation load on body {
    load_potential - 2 * mu * coordinate(0) / length_scale = 0;
  }
  relation balance on body {
    -div(
      2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))
    ) - grad(load_potential) = 0;
  }
  relation x_lower_fixed on x_lower { trace(displacement) = 0; }
  relation x_upper_free on x_upper {
    normal(2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
  relation y_lower_free on y_lower {
    normal(2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
  relation y_upper_free on y_upper {
    normal(2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
}"#;
const CELLS_PER_AXIS: usize = 16;
const VERTICES_PER_AXIS: usize = CELLS_PER_AXIS + 1;

struct Accepted {
    document: ModelDocument,
    geometry: CanonicalGeometryV1,
    mesh: CartesianMeshEnvelopeV1,
    correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    plan: CommonElasticityPlan,
    result: CommonResult,
}

fn accepted() -> Accepted {
    accepted_source(SOURCE)
}

fn accepted_source(source: &str) -> Accepted {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    let geometry = graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("body".to_owned(), vec![rectangle.region().into()]),
                (
                    "x_lower".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[0])],
                ),
                (
                    "x_upper".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[1])],
                ),
                (
                    "y_lower".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[2])],
                ),
                (
                    "y_upper".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[3])],
                ),
            ]),
        )
        .unwrap();
    let compile_parameters = &[
        (
            "mu",
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
            "lambda",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(0.0)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "length_scale",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(1.0)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
    ];
    let mut compile_bindings = vec![
        (
            "body",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("body").unwrap(),
                parent: None,
            },
        ),
        (
            "x_lower",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("x_lower").unwrap(),
                parent: Some(geometry.entity_set("body").unwrap()),
            },
        ),
        (
            "x_upper",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("x_upper").unwrap(),
                parent: Some(geometry.entity_set("body").unwrap()),
            },
        ),
        (
            "y_lower",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("y_lower").unwrap(),
                parent: Some(geometry.entity_set("body").unwrap()),
            },
        ),
        (
            "y_upper",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("y_upper").unwrap(),
                parent: Some(geometry.entity_set("body").unwrap()),
            },
        ),
    ];
    compile_bindings.extend(compile_parameters.iter().map(|(name, value)| {
        (
            *name,
            eqiora::compiler::StaticBindingValue::Expression(value),
        )
    }));
    let document = ModelDocument::compile_selected(
        "mixed-boundary-elasticity.eqi",
        source,
        "MixedBoundaryElasticity2d",
        &compile_bindings,
    )
    .unwrap();
    let cells = CartesianMeshCellsV2::new([CELLS_PER_AXIS; 2]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
            &geometry,
            cells.cells().try_into().unwrap(),
        )
        .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner = AuthenticatedCommonMesh::structured_cartesian(
        geometry.clone(),
        mesh.clone(),
        correspondence.clone(),
        production,
    )
    .unwrap();
    let solver = CommonSolvePolicy::Linear(
        eqiora_numerics::CommonLinearRequest::exact(
            eqiora::solver::SolverPlan::new(
                eqiora::solver::LinearSolver::ConjugateGradient,
                1.0e-10,
                1.0e-12,
                NonZeroUsize::new(10_000).unwrap(),
            )
            .unwrap()
            .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
            .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
            eqiora::solver::REFERENCE_SOLVER_PROVIDER,
        )
        .unwrap(),
    );
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = resolve_common_plan(
        &model,
        owner,
        CommonSpatialPolicy::Q1,
        solver,
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
    .as_elasticity()
    .cloned()
    .expect("fixture retains its admitted elasticity Plan");
    let result = plan
        .run_result(&eqiora::solver::REFERENCE_LINEAR_SOLVER)
        .unwrap();
    Accepted {
        document,
        geometry,
        mesh,
        correspondence,
        plan,
        result,
    }
}

#[test]
fn elastic_energy_is_observed_from_the_accepted_displacement_gradient() {
    let source = SOURCE.replace(
        "  relation load on body {",
        r#"  observable energy: N = integral(
    mu * contract(symmetric_part(grad(displacement)),
                  symmetric_part(grad(displacement)), axes = ((0, 0), (1, 1))),
    measure(body));
  observable boundary_work: N = integral(
    contract(normal(2 * mu * symmetric_part(grad(displacement))),
             trace(displacement), axes = ((0, 0),)), measure(x_upper));
  relation load on body {"#,
    );
    let accepted = accepted_source(&source);
    let model = ModelEnvelope::from_program(accepted.document.program()).unwrap();
    let energy = accepted
        .document
        .program()
        .nodes()
        .find_map(|node| match node {
            eqiora::kernel::KernelNode::Observable(value)
                if matches!(
                    value.reduction(),
                    eqiora::kernel::ObservableReduction::SpatialIntegral {
                        measure: eqiora::kernel::ObservableMeasure::Volume,
                        ..
                    }
                ) =>
            {
                Some(value.id())
            }
            _ => None,
        })
        .unwrap();
    let quadrature = eqiora::meshing::QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
    let observed = accepted
        .result
        .observe(&model, energy, Some(&quadrature))
        .unwrap();
    // Independent Q1 solution: u_x interpolates x - x^2/2, u_y = 0.
    // Its cellwise derivative is 1 - x_mid. Midpoint summation gives
    // integral 3*(1-x_mid)^2 dx dy = 1 - h^2/4 = 1023/1024 N.
    // This is energy per out-of-plane thickness, not three-dimensional energy.
    let expected = 1.0 - 1.0 / (4.0 * (CELLS_PER_AXIS * CELLS_PER_AXIS) as f64);
    assert!((observed.value().component(0).unwrap().0 - expected).abs() < 1.0e-8);
    assert_eq!(
        observed.value().value_type().dimension(),
        eqiora::DimExponents::from_integers([1, 1, -2, 0, 0, 0, 0]).unwrap(),
    );
    let displacement = accepted
        .document
        .program()
        .nodes()
        .find_map(|node| match node {
            eqiora::kernel::KernelNode::Field(value) if !value.value_type().shape().is_scalar() => {
                Some(value)
            }
            _ => None,
        })
        .unwrap();
    let (_, values, _) = accepted.result.field_block(0, 0).unwrap();
    for (vertex, value) in values.as_chunks::<2>().0.iter().enumerate() {
        let coordinates = accepted
            .mesh
            .mesh()
            .vertex_coordinates(MeshEntity::new(0, vertex))
            .unwrap();
        let x = coordinates[0];
        assert!((value[0] - (x - x * x / 2.0)).abs() < 2e-11);
        assert!(value[1].abs() < 2e-11);
    }
    let coefficients = (0..VERTICES_PER_AXIS * VERTICES_PER_AXIS)
        .flat_map(|vertex| {
            accepted
                .mesh
                .mesh()
                .vertex_coordinates(MeshEntity::new(0, vertex))
                .unwrap()
        })
        .map(|value| eqiora::DynQuantity::new(value, displacement.dimension()))
        .collect::<Vec<_>>();
    let direction = accepted
        .result
        .observable_state_tangent([(displacement.id(), coefficients.clone())])
        .unwrap();
    let action = accepted
        .result
        .observe_state_jvp(&model, energy, &quadrature, &direction)
        .unwrap();
    // eta=(x,y): 2*mu*epsilon(u):epsilon(eta) integrates to 6*integral(1-x)=3 N.
    // This State direction is unconstrained; it is not a claimed equilibrium variation.
    assert!((action.real_scalar_value().unwrap().value() - 3.0).abs() < 1.0e-8);
    assert!(
        accepted
            .result
            .observable_state_tangent([(
                displacement.id(),
                coefficients[..coefficients.len() - 1].to_vec()
            )])
            .is_err()
    );
    let boundary_work = accepted
        .document
        .program()
        .nodes()
        .find_map(|node| match node {
            eqiora::kernel::KernelNode::Observable(value) if value.id() != energy => {
                Some(value.id())
            }
            _ => None,
        })
        .unwrap();
    let face = eqiora::meshing::QuadratureRule::gauss_legendre(2).unwrap();
    let work = accepted
        .result
        .observe(&model, boundary_work, Some(&face))
        .unwrap();
    // The Q1 recovered traction is 3h Pa, not the exactly zero natural Law datum.
    // With trace u_x=1/2 m, its boundary pairing is 3h/2 = 3/32 N.
    assert!((work.value().component(0).unwrap().0 - 3.0 / 32.0).abs() < 1e-8);
    let work_action = accepted
        .result
        .observe_state_jvp(&model, boundary_work, &face, &direction)
        .unwrap();
    // Delta traction=6 Pa and eta_x=1 m on this face: 6/2 + 3h = 51/16 N.
    assert!((work_action.component(0).unwrap().0 - 51.0 / 16.0).abs() < 1e-8);
    assert!(
        accepted
            .result
            .observe(&model, boundary_work, Some(&quadrature))
            .is_err()
    );
    let bytes = accepted.result.to_bytes().unwrap();
    let replayed = CommonResult::from_bytes(&bytes, accepted.result.plan()).unwrap();
    assert_eq!(
        replayed.observe(&model, energy, Some(&quadrature)).unwrap(),
        observed
    );
}

#[test]
fn common_elasticity_output_closes_exact_plan_and_mesh_lineage() {
    let accepted = accepted();
    let mesh = accepted.mesh.mesh();
    assert_eq!(mesh.entity_count(0), Some(289));
    assert_eq!(mesh.entity_count(2), Some(256));
    assert_eq!(mesh.axis_coordinates(0), Some(axis().as_slice()));
    assert_eq!(mesh.axis_coordinates(1), Some(axis().as_slice()));

    for i in 0..VERTICES_PER_AXIS {
        for j in 0..VERTICES_PER_AXIS {
            let vertex = 17 * i + j;
            assert_eq!(
                mesh.vertex_coordinates(MeshEntity::new(0, vertex)),
                Some(vec![i as f64 / 16.0, j as f64 / 16.0]),
            );
        }
    }
    for i in 0..CELLS_PER_AXIS {
        for j in 0..CELLS_PER_AXIS {
            let cell = 16 * i + j;
            let lower = 17 * i + j;
            assert_eq!(
                mesh.entity_vertices(MeshEntity::new(2, cell))
                    .unwrap()
                    .into_iter()
                    .map(|vertex| vertex.index())
                    .collect::<Vec<_>>(),
                [lower, lower + 17, lower + 1, lower + 18],
            );
        }
    }

    assert_eq!(accepted.result.plan().identity(), accepted.plan.identity());
    assert_eq!(
        accepted.plan.model_digest(),
        accepted.document.digest().unwrap()
    );
    assert_eq!(accepted.plan.cells(), [CELLS_PER_AXIS; 2]);
    assert_eq!(
        accepted.plan.geometry_digest(),
        accepted
            .geometry
            .digest_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    accepted
        .correspondence
        .validate_against_planar_rectangle_v2_cartesian(
            &accepted.geometry,
            &accepted.mesh,
            [CELLS_PER_AXIS; 2],
        )
        .unwrap();

    let (association, displacement, shape) = accepted.result.field_block(0, 0).unwrap();
    assert_eq!(association, "vertex");
    assert_eq!(shape, [289, 2]);
    assert_eq!(displacement.len(), 578);
    assert!(displacement.iter().all(|value| value.is_finite()));
    let (constrained_reaction, integrated_body_force, assembly_counts, exact_bounds) =
        accepted.result.elasticity_observation().unwrap();
    assert_eq!(exact_bounds, [[0.0, 1.0], [0.0, 1.0]]);
    assert!(constrained_reaction.into_iter().all(f64::is_finite));
    assert!(integrated_body_force.into_iter().all(f64::is_finite));
    assert!(assembly_counts.into_iter().all(|count| count > 0));
    assert!(
        accepted.result.solve_true_residual_norm(None).unwrap()
            <= accepted.result.solve_residual_target(None).unwrap()
    );

    let result_bytes = accepted.result.to_bytes().unwrap();
    let replayed = CommonResult::from_bytes(&result_bytes, accepted.result.plan()).unwrap();
    assert_eq!(replayed.to_bytes().unwrap(), result_bytes);
    assert_eq!(
        replayed.field_block(0, 0).unwrap(),
        accepted.result.field_block(0, 0).unwrap()
    );
    assert_eq!(
        replayed.elasticity_observation().unwrap(),
        accepted.result.elasticity_observation().unwrap()
    );

    let mut noncanonical = result_bytes;
    noncanonical.push(b'\n');
    assert!(CommonResult::from_bytes(&noncanonical, accepted.result.plan()).is_err());
}

#[test]
fn cartesian_mesh_and_correspondence_round_trip_canonically_and_reject_mutants() {
    let accepted = accepted();
    let mesh_bytes = accepted.mesh.canonical_json().unwrap();
    assert_top_level_key_order(
        &mesh_bytes,
        &[
            "schema",
            "encoding",
            "dimension",
            "scalar",
            "cell_family",
            "axes",
            "vertex_order",
            "cell_order",
            "local_node_order",
        ],
    );
    let mesh_json: Value = serde_json::from_slice(&mesh_bytes).unwrap();
    assert_eq!(mesh_json["axes"], json!([axis(), axis()]));
    let decoded = CartesianMeshEnvelopeV1::from_json(&mesh_bytes, Default::default()).unwrap();
    assert_eq!(decoded, accepted.mesh);

    let mut wrong_axis = mesh_json.clone();
    wrong_axis["axes"][0].as_array_mut().unwrap().reverse();
    assert!(
        CartesianMeshEnvelopeV1::from_json(
            &serde_json::to_vec(&wrong_axis).unwrap(),
            Default::default(),
        )
        .is_err()
    );

    let correspondence_bytes = accepted.correspondence.canonical_json().unwrap();
    let decoded =
        GeometryMeshCorrespondenceEnvelopeV1::from_json(&correspondence_bytes, Default::default())
            .unwrap();
    assert_eq!(decoded, accepted.correspondence);
    decoded
        .validate_against_planar_rectangle_v2_cartesian(
            &accepted.geometry,
            &accepted.mesh,
            [CELLS_PER_AXIS; 2],
        )
        .unwrap();
}

fn axis() -> Vec<f64> {
    (0..=CELLS_PER_AXIS)
        .map(|index| index as f64 / CELLS_PER_AXIS as f64)
        .collect()
}

fn assert_top_level_key_order(bytes: &[u8], keys: &[&str]) {
    let text = std::str::from_utf8(bytes).unwrap();
    let positions = keys
        .iter()
        .map(|key| text.find(&format!("\"{key}\"")).unwrap())
        .collect::<Vec<_>>();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
}
