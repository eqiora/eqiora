use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    ModelEnvelope,
};
use eqiora::geometry::{GeometryGraph, PlanarTopologyHandle};
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearPlan, CommonSolvePolicy, CommonSpatialPolicy,
};

pub(crate) const COMPONENT: &str = r#"
public component DifferentiatedPoisson(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
  parameter diffusion: 1,
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2,
  parameter boundary_offset: 1
) {

  variable potential: 1 on square;
  relation balance on square {
    -div(diffusion * grad(potential))
      - source_scale * math.sin(wave_number * coordinate(0))
        * math.sin(wave_number * coordinate(1)) = 0;
  }
  relation x_lower_value on x_lower { trace(potential) - boundary_offset = 0; }
  relation x_upper_value on x_upper { trace(potential) - boundary_offset = 0; }
  relation y_lower_value on y_lower { trace(potential) - boundary_offset = 0; }
  relation y_upper_value on y_upper { trace(potential) - boundary_offset = 0; }
}
"#;

pub(crate) fn document_and_plan(spatial: CommonSpatialPolicy) -> (ModelDocument, CommonLinearPlan) {
    document_and_plan_with_source(spatial, COMPONENT)
}

pub(crate) fn document_and_plans() -> (ModelDocument, CommonLinearPlan, CommonLinearPlan) {
    document_and_plans_with_source(COMPONENT)
}

pub(crate) fn plan_for_document(
    document: &ModelDocument,
    cells: usize,
    spatial: CommonSpatialPolicy,
) -> CommonLinearPlan {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    let geometry = graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("square".to_owned(), vec![rectangle.region().into()]),
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
    let cells = CartesianMeshCellsV2::new([cells, cells]).unwrap();
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
    let owner =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    let solver = CommonSolvePolicy::Linear(
        eqiora_numerics::CommonLinearRequest::exact(
            eqiora::solver::SolverPlan::new(
                if spatial == CommonSpatialPolicy::CellCenteredTpfa {
                    eqiora::solver::LinearSolver::ConjugateGradient
                } else {
                    eqiora::solver::LinearSolver::BiConjugateGradientStabilized
                },
                1.0e-12,
                1.0e-14,
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
    eqiora_numerics::ResolvedCommonPlan::resolve(
        &model,
        owner,
        spatial,
        solver,
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
    .as_linear()
    .cloned()
    .expect("fixture retains its admitted scalar Plan")
}

pub(crate) fn document_and_plan_with_source(
    spatial: CommonSpatialPolicy,
    source: &str,
) -> (ModelDocument, CommonLinearPlan) {
    let (document, q1, tpfa) = document_and_plans_with_source(source);
    let plan = match spatial {
        CommonSpatialPolicy::Q1 => q1,
        CommonSpatialPolicy::CellCenteredTpfa => tpfa,
        CommonSpatialPolicy::TetrahedralEdge
        | CommonSpatialPolicy::TetrahedralFace
        | CommonSpatialPolicy::P1
        | CommonSpatialPolicy::MiniP1
        | CommonSpatialPolicy::CellCentered => {
            panic!("this Cartesian fixture does not admit the requested policy")
        }
    };
    (document, plan)
}

fn document_and_plans_with_source(
    source: &str,
) -> (ModelDocument, CommonLinearPlan, CommonLinearPlan) {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    let geometry = graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("square".to_owned(), vec![rectangle.region().into()]),
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
            "diffusion",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(1.0)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "wave_number",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(std::f64::consts::PI)
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "source_scale",
            eqiora::language::SourceAstFactory::expression(
                eqiora::language::ExprKind::Number(
                    eqiora::language::DecimalLiteral::from_f64(2.0 * std::f64::consts::PI.powi(2))
                        .expect("finite fixture literal"),
                ),
                eqiora::language::TextRange::new(0, 0),
            )
            .unwrap(),
        ),
        (
            "boundary_offset",
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
            "square",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("square").unwrap(),
                parent: None,
            },
        ),
        (
            "x_lower",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("x_lower").unwrap(),
                parent: Some(geometry.entity_set("square").unwrap()),
            },
        ),
        (
            "x_upper",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("x_upper").unwrap(),
                parent: Some(geometry.entity_set("square").unwrap()),
            },
        ),
        (
            "y_lower",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("y_lower").unwrap(),
                parent: Some(geometry.entity_set("square").unwrap()),
            },
        ),
        (
            "y_upper",
            eqiora::compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("y_upper").unwrap(),
                parent: Some(geometry.entity_set("square").unwrap()),
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
        "differentiated-poisson.eqi",
        source,
        "DifferentiatedPoisson",
        &compile_bindings,
    )
    .unwrap();
    let cells = CartesianMeshCellsV2::new([12, 12]).unwrap();
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
    let owner =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    let solver = |spatial| {
        CommonSolvePolicy::Linear(
            eqiora_numerics::CommonLinearRequest::exact(
                eqiora::solver::SolverPlan::new(
                    if spatial == CommonSpatialPolicy::CellCenteredTpfa {
                        eqiora::solver::LinearSolver::ConjugateGradient
                    } else {
                        eqiora::solver::LinearSolver::BiConjugateGradientStabilized
                    },
                    1.0e-12,
                    1.0e-14,
                    NonZeroUsize::new(10_000).unwrap(),
                )
                .unwrap()
                .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
                .with_reduction(eqiora::solver::ReductionPolicy::Reproducible),
                eqiora::solver::REFERENCE_SOLVER_PROVIDER,
            )
            .unwrap(),
        )
    };
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let resolve = |owner, spatial| {
        eqiora_numerics::ResolvedCommonPlan::resolve(
            &model,
            owner,
            spatial,
            solver(spatial),
            None,
            None,
            &REFERENCE_LINEAR_SOLVER,
            None,
        )
        .unwrap()
        .as_linear()
        .cloned()
        .expect("fixture retains its admitted scalar Plan")
    };
    let q1 = resolve(owner.clone(), CommonSpatialPolicy::Q1);
    let tpfa = resolve(owner, CommonSpatialPolicy::CellCenteredTpfa);
    (document, q1, tpfa)
}
